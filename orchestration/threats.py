"""Scene archive handoff and editing evidence for safeRSI.

Threat tasks use the ordinary task lifecycle and shared skill versions. This
module owns the immutable reconstruction input; it does not run a robot or
assign a safety score.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil

from orchestration.config import normalize_threat_config
from orchestration.storage.repository import inside, load, save, sha, uid, now
from harness.core.contracts import ToolError
from harness.evidence.artifacts import archived_render
from harness.tools.world_state import validate_world, validate_code
from harness.tools.newton.workers.solution import checked_file, relative_path
from harness.tools.newton.workers.protocol import edit_inspection


class SourceRevisionChanged(ValueError):
    """The selected archive revision is no longer the current immutable input."""


def _changes(before, after, path="", limit=200):
    """Bounded structural deltas, not a language-alignment or risk evaluator."""
    result = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            pointer = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                result.append({"path": pointer, "before": before.get(key), "after": after.get(key)})
            else:
                result.extend(_changes(before[key], after[key], pointer, limit - len(result)))
            if len(result) >= limit:
                break
    elif before != after:
        result.append({"path": path or "/", "before": before, "after": after})
    return result[:max(0, limit)]


def _physical_description(world):
    # Source text, evidence, labels and receipt bookkeeping can change without
    # changing a scene. Keep them in the full delta, not the no-op decision.
    ignored = {"description", "note", "evidence", "validation", "relevance", "status"}
    def compact(value):
        if isinstance(value, dict):
            if "real" in value and "simulation" in value:
                return {"simulation": compact(value["simulation"])}
            return {key: compact(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value
    return compact({key: world.get(key, {}) for key in
                    ("objects", "phenomena", "solvers", "couplings", "cameras")})


def _native_values(state):
    return {key: state[key] for key in ("entities", "shapes", "particle_count") if key in state}


def _small(value):
    text = json.dumps(value, ensure_ascii=False)
    return value if len(text) <= 300 else {"preview": text[:300], "truncated": True}


class ThreatRunner:
    def __init__(self, app):
        self.app, self.store = app, app.store

    def _archive(self, task_id, run_id, revision=None, *, verify=True):
        source = self.store.task(task_id)
        if source.get("kind", "reconstruction") != "reconstruction":
            raise ValueError("Choose an output archive from worldRSI")
        run = next((row for row in source["runs"] if row["id"] == run_id), None)
        if run is None or run.get("status") != "completed":
            raise ValueError("The selected reconstruction run has no completed scene")
        directory = self.store.task_path(task_id)
        archive = archived_render(directory, run, for_replay=True)
        if not archive or not archive["is_current"]:
            raise ValueError("The selected run has no accepted current scene archive")
        if revision is not None and revision != archive["render_revision"]:
            raise SourceRevisionChanged("Source archive changed; refresh the scene selection")
        artifact = inside(directory, archive["render_dir"])
        manifest = load(artifact / "render_manifest.json", {})
        if manifest.get("engine") != "newton":
            raise ValueError("This archive is not a supported Newton scene")
        delivery = artifact / "delivery"
        if delivery.is_symlink() or not delivery.is_dir():
            raise ValueError("The archive delivery must be an ordinary directory")
        index = load(delivery / "dependency_manifest.json", {})
        files = index.get("files")
        if not isinstance(files, list) or not 2 <= len(files) <= 514:
            raise ValueError("The archive has no complete ordinary delivery")
        names = [relative_path(item["path"]) for item in files]
        if len(set(names)) != len(names) or not {"world.yaml", "scene.py"} <= set(names):
            raise ValueError("The archive dependency manifest is invalid")
        if manifest.get("dependency_files") != files:
            raise ValueError("Source render and delivery dependency identities disagree")
        for item in files:
            path = checked_file(delivery, item["path"])
            if path.stat().st_size != item["bytes"] or (verify and sha(path) != item["sha256"]):
                raise ValueError("Source archive dependency changed: " + item["path"])
        identities = {item["path"]: item["sha256"] for item in files}
        if identities["world.yaml"] != manifest.get("world_sha256") or identities["scene.py"] != manifest.get("code_sha256"):
            raise ValueError("Source world or code does not match its render")
        world = validate_world((delivery / "world.yaml").read_text())
        validate_code((delivery / "scene.py").read_text(), world)
        # Check these two small entry files even when building the thumbnail catalog.
        if any(sha(delivery / name) != identities[name] for name in ("world.yaml", "scene.py")):
            raise ValueError("Source world or code changed")
        return source, run, archive, artifact, manifest, files

    def list_archives(self):
        rows = []
        for path in self.store.task_records(kind="reconstruction"):
            source = load(path)
            for run in reversed(source.get("runs", [])):
                try:
                    _, _, archive, _, manifest, _ = self._archive(source["id"], run["id"], verify=False)
                except (OSError, ValueError, KeyError, TypeError, ToolError):
                    continue
                rows.append({"task_id": source["id"], "run_id": run["id"],
                             "source_task_id": source["id"], "source_run_id": run["id"],
                             "source_revision": archive["render_revision"],
                             "render_revision": archive["render_revision"], "render_dir": archive["render_dir"],
                             "title": source["title"], "created_at": run.get("finished_at", run.get("created_at")),
                             "source_split": source.get("split", "manual"),
                             "preview_path": archive["preview_path"], "frame_count": archive["frame_count"],
                             "engine": "newton", "engine_version": manifest.get("engine_version"),
                             "timestamps_s": [frame["timestamp_s"] for frame in manifest["frames"]],
                             "timestamp_range_s": [manifest["frames"][0]["timestamp_s"], manifest["frames"][-1]["timestamp_s"]],
                             "source_task_status": source["status"]})
        return sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True)

    def create(self, title, instruction, source_task_id, source_run_id, source_revision, cfg=None):
        self.app.verify_runtime()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Provide an instruction for editing the selected scene")
        if not isinstance(source_revision, str) or not source_revision:
            raise ValueError("Select a specific source archive revision")
        source, _, archive, artifact, manifest, files = self._archive(
            source_task_id, source_run_id, source_revision)
        cfg = normalize_threat_config(cfg, defaults=self.store.config())
        cfg["trial_enabled"] = False
        source_split = source.get("split", "manual")
        excluded = source_split in {"test", "validation", "heldout", "held-out", "held_out", "trial", "smoke"}
        if excluded:
            cfg["outer_rsi"] = False
        if cfg.get("seed") is None:
            source_seed = manifest.get("effective_seed")
            cfg["seed"] = 7 if source_seed is None else source_seed
        identity = uid("safe_task")
        directory = self.store.task_path(identity)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            baseline, work = directory / "source/scene", directory / "work"
            baseline.mkdir(parents=True)
            work.mkdir()
            for item in files:
                original = checked_file(artifact / "delivery", item["path"])
                target = baseline / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                if sha(target) != item["sha256"]:
                    raise ValueError("Source changed while copying: " + item["path"])
                target.chmod(0o444)
                destination = work / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, destination)
            observations = []
            for entry in source.get("inputs", []):
                original = inside(self.store.task_path(source_task_id), entry["local_path"])
                target = inside(directory, entry["local_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                if sha(target) != entry["sha256"]:
                    raise ValueError("Archived public source input changed")
                target.chmod(0o444)
                observations.append({key: value for key, value in entry.items() if key != "benchmark"})
            (directory / "inputs").mkdir(exist_ok=True)
            preview = inside(self.store.task_path(source_task_id), archive["preview_path"])
            shutil.copyfile(preview, directory / "source/preview.png")
            (directory / "source/preview.png").chmod(0o444)
            save(directory / "source/render_frames.json", [frame.get("pixel_sha256") for frame in manifest["frames"]])
            (directory / "source/render_frames.json").chmod(0o444)
            native = artifact / "runtime_state.json"
            if native.is_file() and not native.is_symlink():
                shutil.copyfile(native, directory / "source/runtime_state.json")
                (directory / "source/runtime_state.json").chmod(0o444)
            source_contract = load(baseline / "output_contract.json", {})
            times = [frame["timestamp_s"] for frame in manifest["frames"]]
            camera = manifest["frames"][0].get("camera", {})
            view = {'time_s': 0.}
            if all(key in camera for key in ('position','target','up')):
                view['camera'] = {key:camera[key] for key in ('position','target','up','fov_deg') if key in camera}
            contract = edit_inspection([view])
            source_scene = {"task_id": source_task_id, "run_id": source_run_id,
                "render_dir": archive["render_dir"], "render_revision": archive["render_revision"],
                "world_sha256": manifest["world_sha256"], "code_sha256": manifest["code_sha256"],
                "render_manifest_sha256": sha(artifact / "render_manifest.json"),
                "baseline_path": "source/scene", "preview_path": "source/preview.png",
                "preview_local_path": "source/preview.png", "preview_timestamp_s": times[0],
                "runtime_state_path": "source/runtime_state.json" if native.is_file() else None,
                "files": files, "output_contract": source_contract, "skill_version": source["version"],
                "source_split": source_split,
                "effective_seed": manifest.get("effective_seed"),
                "scope": "Immutable worldRSI implementation input; media retain reconstruction provenance, not similarity targets"}
            save(directory / "source/manifest.json", source_scene)
            (directory / "source/manifest.json").chmod(0o444)
            pinned = self.app.versions.active()
            snapshot = self.app.versions.snapshot(pinned)
            save(directory / "snapshot.json", snapshot)
            self.app.versions.materialize(pinned, directory / "skills")
            record = {"schema_version": 3, "kind": "threat", "id": identity,
                "title": title.strip() or "Scene edit", "instruction": instruction.strip(),
                "split": source_split if excluded else "manual", "status": "ready", "version": pinned,
                "created_at": now(), "updated_at": now(), "runs": [], "inputs": observations,
                "reference": None, "source_scene": source_scene, "output_contract": contract,
                "config": cfg, "artifacts": [], "metrics": None, "critique": None, "trials": [],
                "harness": self.app.harness_snapshot(self.app.bind(cfg)),
                "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()}
            save(directory / "task.json", record)
            return record
        except BaseException:
            shutil.rmtree(directory)
            raise

    def evaluate(self, directory, task, calls, *, artifact_dir=None, artifact_source=None):
        """Freeze local editing evidence; intentionally no rollout or safety score."""
        from evaluation.evaluator import task_usage
        directory = self.store.task_path(task["id"])
        source = task["source_scene"]
        baseline = inside(directory, source["baseline_path"])
        for item in source["files"]:
            if sha(checked_file(baseline, item["path"])) != item["sha256"]:
                raise ValueError("Immutable source scene changed")
        usage = task_usage(calls, task["id"])
        result = {"evaluation_profile": "scene_edit", "full_profile": False,
            "quality_status": "unavailable", "edit_status": "unavailable", "usage": usage,
            "metrics": {"tokens": {"status": "ok" if usage["complete"] else "missing",
                                    "value": usage["total_tokens"] if usage["complete"] else None}},
            "source_scene": source, "artifact_source": artifact_source,
            "instruction_alignment": {"status": "review_required", "reason": "The run critic assesses the requested scene edits"},
            "rollout": {"status": "deferred", "reason": "This snapshot is frozen at scene delivery. See the independent rollout archives for execution."},
            "safety_evaluation": {"status": "deferred", "reason": "Episode metrics and task-critic feedback belong to the post-task experiment archive"}}
        if artifact_dir is None:
            result["reason"] = "No accepted native edited scene is available"
            return result
        artifact = inside(directory, artifact_dir)
        manifest = load(artifact / "render_manifest.json")
        delivery = artifact / "delivery"
        world = validate_world((delivery / "world.yaml").read_text())
        validate_code((delivery / "scene.py").read_text(), world)
        after_files = manifest["dependency_files"]
        for item in after_files:
            if sha(checked_file(delivery, item["path"])) != item["sha256"]:
                raise ValueError("Edited archive dependency changed")
        before_world = validate_world((baseline / "world.yaml").read_text())
        before = {item["path"]: item["sha256"] for item in source["files"]}
        after = {item["path"]: item["sha256"] for item in after_files}
        changed_files = [{"path": name, "status": "added" if name not in before else "removed" if name not in after else "modified"}
                         for name in sorted(before.keys() | after.keys())
                         if name != "output_contract.json" and before.get(name) != after.get(name)]
        previous_state = load(directory / "source/runtime_state.json", {})
        state = load(artifact / "runtime_state.json", {})
        if state.get("engine") != "newton":
            raise ValueError("Edited archive has no actual native state")
        previous_frames = load(directory / "source/render_frames.json", [])
        # The copied source manifest retains frame hashes independently of the
        # original mutable task. New scene and metadata identities are excluded.
        comparison = load(artifact / 'comparison.json', {})
        if manifest.get('inspection'):
            paired = (comparison.get('source_revision') == source['render_revision'] and
                      comparison.get('edited_manifest_sha256') == sha(artifact/'render_manifest.json'))
            render_changed = paired and comparison['pixels_changed']
            native_changed = paired and comparison['native_changed']
        else:
            render_changed = bool(previous_frames) and previous_frames != [frame.get("pixel_sha256") for frame in manifest["frames"]]
            native_changed = bool(previous_state) and _native_values(previous_state) != _native_values(state)
        reported_metadata_changed = previous_state.get("metadata") != state.get("metadata")
        physical_changed = _physical_description(before_world) != _physical_description(world)
        status = "changed" if physical_changed or native_changed or render_changed else "unchanged"
        if manifest.get('inspection') and not (native_changed or render_changed):
            status = "unverified" if physical_changed or not paired else "unchanged"
        changes = _changes(before_world, world)
        save(directory / "evaluation/edit_report.json", {"world_changes": changes, "changed_files": changed_files})
        run = next((item for item in task.get('runs', []) if item.get('render_dir') ==
                    str(artifact.relative_to(directory)) and item.get('critic_status') == 'completed'), {})
        review = run.get('critique') or {}
        if review.get('edit_verdict'):
            result['instruction_alignment'] = {'status':review['edit_verdict'], 'run_id':run['id'],
                                               'reason':review.get('edit_reason',review['summary'])}
        result.update(quality_status="ok", edit_status=status, edit_report={
            "status": status, "changed_files": changed_files[:100], "changed_file_count": len(changed_files),
            "world_changes": [{"path": row["path"], "before": _small(row["before"]), "after": _small(row["after"])}
                              for row in changes[:30]], "world_change_count": len(changes),
            "full_report_path": "evaluation/edit_report.json", "render_changed": render_changed,
            "comparison_path": str((artifact/'comparison.json').relative_to(directory)) if comparison else None,
            "native_state_changed": native_changed, "physical_description_changed": physical_changed,
            "reported_metadata_changed": reported_metadata_changed,
            "runtime_effect": "observed_difference" if native_changed or render_changed else "unverified",
            "native_state_path": str((artifact / "runtime_state.json").relative_to(directory)),
            "comparison_scope": "Declared scene differences and actual native readback; no embodiment risk claim"},
            edit_assessment={"status": status, "scope": "scene editing engineering evidence only",
                             "instruction_alignment": review.get('edit_verdict', 'critic_review_required'), "safety_evaluation": "deferred"})
        return result
