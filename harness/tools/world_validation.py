#!/usr/bin/env python3
"""Schema-3 world validation shared by live tools and design checks.

Structure and references are checked here; runtime bindings and physical
correctness still require actual native execution and evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

SCHEMA_PATH = Path(__file__).with_name("world-state-v3.schema.json")


@dataclass(frozen=True)
class Issue:
    path: str
    message: str


class ContractLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently losing a world hypothesis."""


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    value = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML mapping keys must be strings")
        if key in value:
            raise ValueError(f"Duplicate YAML key: {key}")
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


ContractLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _walk(value: Any, path="$", ancestors=frozenset(), depth=0, budget=None):
    if budget is None:
        budget = [50_000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 50:
        raise ValueError("Document exceeds the structural size/depth limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite number")
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ValueError(f"{path}: recursive YAML aliases are unsupported")
        ancestors = ancestors | {id(value)}
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{path}: value must be JSON-compatible")
    yield path, value
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, child in children:
        child_path = f"{path}.{key}" if isinstance(value, dict) else f"{path}[{key}]"
        yield from _walk(child, child_path, ancestors, depth + 1, budget)


def load_world(path: Path):
    if path.stat().st_size > 512_000:
        raise ValueError("World exceeds the 512000-byte limit")
    value = yaml.load(path.read_text(), Loader=ContractLoader)
    list(_walk(value))
    return value


def load_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return schema


def _pointer(value, pointer):
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Expected an RFC 6901 JSON Pointer")
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON Pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("Array pointer requires an existing index")
            index = int(token)
            if index >= len(value):
                raise ValueError("JSON Pointer index does not exist")
            value = value[index]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ValueError("JSON Pointer target does not exist")
    return value


def validate_world(value, schema=None):
    try:
        nodes = list(_walk(value))
    except ValueError as exc:
        return [Issue("$", str(exc))]
    schema = load_schema() if schema is None else schema
    structural = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.json_path))
    if structural:
        return [Issue(error.json_path, error.message) for error in structural]
    return _semantic_checks(value, nodes)


def _semantic_checks(world, nodes):
    issues = []

    def fail(path, message):
        issues.append(Issue(path, message))

    def ref(identity, mapping, path, label):
        if not isinstance(identity, str) or identity not in mapping:
            fail(path, f"Unknown {label}: {identity}")
            return False
        return True

    sources = world.get("sources", {})
    objects = world.get("objects", {})
    phenomena = world.get("phenomena", {})
    solvers = world.get("solvers", {})
    couplings = world.get("couplings", {})
    cameras = world.get("cameras", {})
    contexts = world.get("contexts", {})
    clocks = world.get("timeline", {}).get("clocks", {})
    semantic = {**objects, **phenomena, **cameras}
    for identity in set(objects) & (set(phenomena) | set(cameras)) | (set(phenomena) & set(cameras)):
        fail("$", f"Semantic ID must be unique across objects, phenomena and cameras: {identity}")

    # Claims remain revisable; only a claim explicitly marked observed requires
    # an available source. An example or missing video is not an observation.
    for path, node in nodes:
        if not isinstance(node, dict):
            continue
        if (isinstance(node.get("real"), dict) and "unit" in node["real"]
                and isinstance(node.get("simulation"), dict) and "unit" in node["simulation"]):
            if node["real"].get("unit") != node["simulation"].get("unit"):
                fail(path, "Real and simulation values of a parameter must use the same canonical unit")
            interval = node.get("plausible_range")
            if interval is not None and interval[0] > interval[1]:
                fail(path + ".plausible_range", "Plausible range minimum exceeds maximum")
        references = node.get("evidence", [])
        if isinstance(references, (str, dict)):
            references = [references]
        if not isinstance(references, list):
            fail(path + ".evidence", "Evidence must be a source reference or list")
            continue
        for index, evidence in enumerate(references):
            identity = evidence if isinstance(evidence, str) else evidence.get("source") if isinstance(evidence, dict) else None
            location = f"{path}.evidence[{index}]"
            if ref(identity, sources, location, "evidence source") and node.get("status") == "observed":
                if sources[identity].get("availability") != "available":
                    fail(location, "Observed claims require an available source, not a design fixture or missing media")

    for identity, source in sources.items():
        path = f"$.sources.{identity}"
        if source.get("camera"):
            if ref(source["camera"], cameras, path + ".camera", "camera"):
                camera = cameras[source["camera"]]
                if camera.get("role") == "source" and camera.get("source") != identity:
                    fail(path + ".camera", "Source camera points back to a different source")
        if source.get("clock"):
            ref(source["clock"], clocks, path + ".clock", "clock")
        previous_end = None
        shot_ids = set()
        for index, shot in enumerate(source.get("shots", [])):
            location = f"{path}.shots[{index}]"
            ref(shot["camera"], cameras, location + ".camera", "camera")
            if shot["id"] in shot_ids:
                fail(location + ".id", "Shot IDs must be unique within a source")
            shot_ids.add(shot["id"])
            start, end = shot["interval_s"]
            if start >= end:
                fail(location + ".interval_s", "Shot interval must have positive duration")
            if previous_end is not None and start < previous_end:
                fail(location + ".interval_s", "Shots must be ordered and non-overlapping")
            previous_end = end
            camera = cameras.get(shot["camera"], {})
            if camera.get("source") not in (None, identity):
                fail(location + ".camera", "Shot camera belongs to a different source")

    known_mappings = []
    for index, mapping in enumerate(world.get("timeline", {}).get("source_mappings", [])):
        path = f"$.timeline.source_mappings[{index}]"
        ref(mapping["source"], sources, path + ".source", "source")
        for key in ("source_clock", "world_clock"):
            ref(mapping[key], clocks, path + "." + key, "clock")
        if mapping["world_clock"] != "world":
            fail(path + ".world_clock", "The target world clock must be timeline.clocks.world")
        source_clock = sources.get(mapping["source"], {}).get("clock")
        if source_clock is not None and source_clock != mapping["source_clock"]:
            fail(path + ".source_clock", "Mapping clock differs from the declared source clock")
        for key in ("source_interval_s", "world_interval_s"):
            interval = mapping.get(key)
            if interval is not None and interval[0] >= interval[1]:
                fail(path + "." + key, "Mapped interval must have positive duration")
        source_interval, world_interval = mapping.get("source_interval_s"), mapping.get("world_interval_s")
        if (source_interval is not None and world_interval is not None
                and source_interval[1] > source_interval[0] and world_interval[1] > world_interval[0]):
            slope = (world_interval[1] - world_interval[0]) / (source_interval[1] - source_interval[0])
            offset = world_interval[0] - slope * source_interval[0]
            for previous in known_mappings:
                old, old_slope, old_offset = previous
                if old["source"] != mapping["source"] or old["source_clock"] != mapping["source_clock"]:
                    continue
                a = max(old["source_interval_s"][0], source_interval[0])
                b = min(old["source_interval_s"][1], source_interval[1])
                if a < b and any(abs((slope - old_slope) * t + offset - old_offset) > 1e-9 for t in (a, b)):
                    fail(path, "Overlapping source mappings assign conflicting world times")
            known_mappings.append((mapping, slope, offset))

    for identity, obj in objects.items():
        path = f"$.objects.{identity}"
        physics = obj.get("physics", {})
        if physics.get("solver"):
            ref(physics["solver"], solvers, path + ".physics.solver", "solver")
        if physics.get("collision") == "enabled" and obj.get("geometry", {}).get("collision", {}).get("kind") == "none":
            fail(path + ".geometry.collision", "Enabled collision cannot use an absent collision representation")
        for index, relevance in enumerate(obj.get("relevance", [])):
            ref(relevance["context"], contexts, f"{path}.relevance[{index}].context", "context")

    for identity, phenomenon in phenomena.items():
        path = f"$.phenomena.{identity}"
        for index, participant in enumerate(phenomenon.get("participants", [])):
            ref(participant, objects, f"{path}.participants[{index}]", "object")
        if phenomenon.get("solver"):
            ref(phenomenon["solver"], solvers, path + ".solver", "solver")
        if phenomenon.get("coupling"):
            ref(phenomenon["coupling"], couplings, path + ".coupling", "coupling")
        for index, relevance in enumerate(phenomenon.get("relevance", [])):
            ref(relevance["context"], contexts, f"{path}.relevance[{index}].context", "context")

    owners = {}
    for identity, coupling in couplings.items():
        path = f"$.couplings.{identity}"
        ref(coupling["schedule"]["clock"], clocks, path + ".schedule.clock", "clock")
        entries = coupling.get("entries", {})
        for entry, data in entries.items():
            location = f"{path}.entries.{entry}"
            ref(data["solver"], solvers, location + ".solver", "solver")
            for index, owner in enumerate(data.get("owns", [])):
                ref(owner, {**objects, **phenomena}, f"{location}.owns[{index}]", "physical owner")
                declared_solver = objects.get(owner, {}).get("physics", {}).get("solver") or phenomena.get(owner, {}).get("solver")
                if declared_solver is not None and declared_solver != data["solver"]:
                    fail(location + ".owns", f"Owner {owner} declares a different solver: {declared_solver}")
                if owner in owners:
                    fail(location + ".owns", f"Duplicate physical ownership of {owner}; already owned by {owners[owner]}")
                owners[owner] = identity + "." + entry
        pairs = set()
        state_pairs, reaction_pairs = set(), set()
        for index, exchange in enumerate(coupling.get("exchanges", [])):
            location = f"{path}.exchanges[{index}]"
            for key in ("from", "to"):
                ref(exchange[key], entries, location + "." + key, "coupling entry")
            if exchange["from"] == exchange["to"]:
                fail(location, "Coupling exchange requires distinct entries")
            pairs.add((exchange["from"], exchange["to"]))
            if exchange["kind"] == "state":
                state_pairs.add((exchange["from"], exchange["to"]))
            elif exchange["kind"] == "reaction":
                reaction_pairs.add((exchange["from"], exchange["to"]))
        if coupling.get("mode") == "two_way":
            if (not state_pairs or not reaction_pairs or any((b, a) not in reaction_pairs for a, b in state_pairs)
                    or any((b, a) not in state_pairs for a, b in reaction_pairs)):
                fail(path + ".exchanges", "Two-way coupling requires matching state and mechanical reaction exchanges in both directions")

    def matrix(claim, path, intrinsic=False):
        value = claim.get("value") if isinstance(claim, dict) else None
        if value is None:
            return
        expected = [0, 0, 1] if intrinsic else [0, 0, 0, 1]
        if not all(abs(a - b) <= 1e-6 for a, b in zip(value[-1], expected)):
            fail(path, "Matrix has an invalid homogeneous last row")
        if intrinsic:
            if value[0][0] <= 0 or value[1][1] <= 0:
                fail(path, "Camera focal lengths must be positive")
        else:
            r = [row[:3] for row in value[:3]]
            dots = [sum(r[k][i] * r[k][j] for k in range(3)) for i in range(3) for j in range(3)]
            target = [1 if i == j else 0 for i in range(3) for j in range(3)]
            determinant = sum(r[0][i] * (r[1][(i + 1) % 3] * r[2][(i + 2) % 3] - r[1][(i + 2) % 3] * r[2][(i + 1) % 3]) for i in range(3))
            if max(abs(a - b) for a, b in zip(dots, target)) > 1e-4 or abs(determinant - 1) > 1e-4:
                fail(path, "Camera transform rotation must be right-handed and orthonormal within 1e-4")

    for identity, camera in cameras.items():
        path = f"$.cameras.{identity}"
        if camera.get("source"):
            ref(camera["source"], sources, path + ".source", "source")
        attachment = camera.get("attachment")
        if attachment:
            ref(attachment["object"], objects, path + ".attachment.object", "object")
            matrix(attachment.get("camera_to_parent"), path + ".attachment.camera_to_parent")
        for index, relevance in enumerate(camera.get("relevance", [])):
            ref(relevance["context"], contexts, f"{path}.relevance[{index}].context", "context")
        pose = camera.get("pose", {})
        if pose.get("clock"):
            ref(pose["clock"], clocks, path + ".pose.clock", "clock")
        matrix(pose.get("transform"), path + ".pose.transform")
        intrinsic = camera.get("intrinsics", {})
        matrix(intrinsic.get("value"), path + ".intrinsics.value", intrinsic=True)
        for key, specification, field, is_intrinsic in (("pose", pose, "transform", False), ("intrinsics", intrinsic, "value", True)):
            previous = None
            for index, sample in enumerate(specification.get("samples", [])):
                location = f"{path}.{key}.samples[{index}]"
                if previous is not None and sample["time_s"] <= previous:
                    fail(location + ".time_s", "Camera samples must be strictly increasing")
                previous = sample["time_s"]
                matrix(sample[field], location + "." + field, intrinsic=is_intrinsic)

    for index, relation in enumerate(world.get("relations", [])):
        for key in ("subject", "object"):
            ref(relation[key], semantic, f"$.relations[{index}].{key}", "semantic ID")
    targets = {"object": objects, "phenomenon": phenomena, "camera": cameras}
    edit_ids = set()
    for index, editable in enumerate(world.get("editables", [])):
        path = f"$.editables[{index}]"
        if editable["id"] in edit_ids:
            fail(path + ".id", "Editable IDs must be unique")
        edit_ids.add(editable["id"])
        target = editable["target"]
        mapping = targets[target["kind"]]
        if ref(target["id"], mapping, path + ".target.id", "edit target"):
            try:
                current = _pointer(mapping[target["id"]], target["path"])
                parent_path = target["path"].rsplit("/", 1)[0]
                parent = _pointer(mapping[target["id"]], parent_path)
                claim = current if isinstance(current, dict) and "value" in current and "status" in current else parent
                claim = claim if isinstance(claim, dict) and "value" in claim and "status" in claim else {}
                actual = current.get("value") if current is claim else current
                if "unit" in editable and editable["unit"] != claim.get("unit"):
                    fail(path + ".unit", "Edit unit must match the target claim unit")
                if "bounds" in editable:
                    numeric = isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    unknown_quantity = claim.get("status") == "unknown" and actual in (None, "unknown") and "unit" in claim
                    if not numeric and not unknown_quantity:
                        fail(path + ".bounds", "Numeric edit bounds require a number or an explicitly unknown quantity")
            except ValueError as exc:
                fail(path + ".target.path", f"{exc}: {target['path']!r}. Resolve the pointer relative to "
                     f"{target['kind']} {target['id']!r}, without a world-level /objects, /phenomena or /cameras prefix.")
        bounds = editable.get("bounds", {})
        if "minimum" in bounds and "maximum" in bounds and bounds["minimum"] > bounds["maximum"]:
            fail(path, "Edit minimum exceeds maximum")
    for identity, bindings in world.get("runtime_bindings", {}).items():
        path = f"$.runtime_bindings.{identity}"
        ref(identity, semantic, path, "semantic ID")
        for index, binding in enumerate(bindings):
            if binding.get("solver"):
                ref(binding["solver"], solvers, f"{path}[{index}].solver", "solver")
            if binding["status"] == "observed":
                observed = [sources.get(evidence["source"], {}) for evidence in binding.get("evidence", [])]
                if not any(source.get("kind") == "observation" and source.get("availability") == "available" for source in observed):
                    fail(f"{path}[{index}].evidence", "Runtime bindings require an available native observation source")
    if world["document_role"] == "design_fixture":
        for group in ("solvers", "couplings"):
            for identity, component in world.get(group, {}).items():
                if component["validation"]["status"] != "not_run" or component["validation"]["receipts"]:
                    fail(f"$.{group}.{identity}.validation", "Design fixtures cannot claim executed validation receipts")
        for identity, bindings in world.get("runtime_bindings", {}).items():
            if any(binding["status"] != "unresolved" for binding in bindings):
                fail(f"$.runtime_bindings.{identity}", "Design fixture runtime bindings must remain unresolved")
    return issues


def validate_world_file(path, schema=None):
    try:
        return validate_world(load_world(Path(path)), schema)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [Issue("$", str(exc))]
