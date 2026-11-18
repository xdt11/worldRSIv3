"""Persistent native Newton observation; backward seeks rebuild and replay."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image
import yaml

from harness.tools.newton.workers.scene import load_scene, RuntimeFailure
from harness.tools.newton.workers.solution import snapshot_inputs, freeze_dependencies


def recoverable_request_error(exc):
    return isinstance(exc, (ValueError, KeyError, TypeError)) or (isinstance(exc, RuntimeFailure) and exc.code in {
        "invalid_camera", "invalid_resolution", "unknown_camera", "unknown_intrinsics",
        "unknown_camera_pose", "unknown_camera_clock", "unknown_shot_clock",
        "unsupported_camera_model", "unknown_camera_attachment", "unreachable_time",
        "reference_camera_unresolved", "reference_camera_mismatch"})


class LiveRenderer:
    def __init__(self, code, configuration):
        self.code = Path(code).resolve()
        self.config = configuration
        self.receipts = self.code.parent / '.observations' / configuration.get('session_id', 'live')
        self.receipts.mkdir(parents=True, exist_ok=True)
        preimages = snapshot_inputs(self.code)
        self.code_hash = preimages['scene.py']
        self.revision = "newton:" + self.code_hash[:24]
        self.seed = int(os.environ["WORLD_RSI_SEED"]) if "WORLD_RSI_SEED" in os.environ else None
        self.world = yaml.safe_load((self.code.parent / "world.yaml").read_text())
        sys.path.insert(0, str(self.code.parent))
        contract_path = self.code.parent / "output_contract.json"
        self.contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
        self.session = load_scene(self.code, self.world, device=os.environ.get("WORLD_RSI_DEVICE", "cuda:0"), seed=self.seed,
                                  context={"output_contract": self.contract})
        declared = list(dict.fromkeys([*self.session.code_dependencies, *self.session.solution_assets]))
        self.dependency_files = freeze_dependencies(self.code, declared, preimages)

    def query(self):
        result = dict(self.session.query())
        result.update(type="state", scene_revision=self.revision, code_sha256=self.code_hash)
        result.setdefault("capabilities", {})
        result["capabilities"].update(simulation_advance=True, source_camera_trajectory=True,
                                      inspection_changes_time=False, native_initial_reset_replay=True)
        return result

    def seek(self, requested, *, details=True):
        if type(requested) not in (int, float) or not math.isfinite(requested) or requested < 0:
            raise ValueError("Requested time must be finite and nonnegative")
        if requested < self.session.time_s - 1e-10:
            self.session.reset(seed=self.seed)
        dt = float(self.session.frame_dt)
        if (requested - self.session.time_s) / dt > self.config.get("max_replay_steps", 100000):
            raise ValueError("Requested observation exceeds max_replay_steps")
        if self.config.get('inspection'):
            self.session.advance_physics_to(float(requested), self.session.physics_dt_s)
        else:
            self.session.advance_to(float(requested))
        result = self.query() if details else {'time_s':float(self.session.time_s),'tick':int(self.session.tick)}
        result.update(requested_time_s=requested, exact=abs(result["time_s"] - requested) <= 1e-8,
                      time_source="native_physics_tick")
        return result

    def command(self, command):
        operation = command.get("op", "render")
        if operation == "query":
            return self.query()
        if operation == "seek":
            return self.seek(command["time_s"])
        if operation == "advance":
            delta = command.get("seconds")
            if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
                raise ValueError("seconds must be finite and nonnegative")
            return self.seek(self.session.time_s + delta)
        if operation == "reset":
            self.session.reset(seed=self.seed if command.get("seed") is None else command["seed"], initial_state=command.get("initial_state"))
            return self.query()
        if operation == "step":
            self.session.step(command.get("action"))
            return self.query()
        if operation != "render":
            raise ValueError("Unknown Newton observation operation")
        requested = command.get("time_s")
        if requested is not None:
            self.seek(requested, details=False)
        width, height = command.get("width", 640), command.get("height", 360)
        if type(width) is not int or type(height) is not int or not 32 <= width <= 1280 or not 32 <= height <= 720:
            raise ValueError("Observation dimensions exceed limits")
        inspection = self.config.get('inspection')
        camera = command.get('camera')
        if inspection and camera is None:
            camera = inspection['views'][0].get('camera')
        rendered = self.session.render(camera_override=camera, source_id=None if inspection else self.contract.get("reference_source_id"),
                                       width=width, height=height, channels=("rgb",), **({'independent_camera':True} if inspection else {}))
        pixels = np.asarray(rendered["rgb"])
        if pixels.dtype != np.uint8 or pixels.shape != (height, width, 3):
            raise ValueError("Newton renderer returned invalid native RGB")
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=88)
        raw = stream.getvalue()
        image_path = self.receipts / (uuid.uuid4().hex + '.jpg')
        image_path.write_bytes(raw)
        receipt = {'kind':'observation_not_evaluation','command':command,
            'scene_revision':self.revision,'time_s':float(rendered['time_s']),'tick':int(rendered['tick']),
            'camera':rendered['camera'],'identity':self.session.identity,'renderer':'Python Warp Render',
            'image':{'path':str(image_path),'sha256':hashlib.sha256(raw).hexdigest(),'width':width,'height':height}}
        receipt_path = image_path.with_suffix('.json')
        receipt_path.write_text(json.dumps(receipt, allow_nan=False, indent=2))
        result = {"type":"frame","width":width,"height":height,
            "time_s":receipt['time_s'],"tick":receipt['tick'],"scene_revision":self.revision,
            "camera":{key:rendered['camera'][key] for key in ('position','target','up','fov_deg','near','far') if key in rendered['camera']},
            "receipt_path":str(receipt_path),"jpeg_base64":base64.b64encode(raw).decode('ascii')}
        if requested is not None and not math.isclose(requested, result['time_s'], rel_tol=0, abs_tol=1e-9):
            result['requested_time_s'] = requested
        return result

    def ready(self):
        times = self.contract.get("timestamps_s") or getattr(self.session, "output_times", None) or [0.0]
        if self.config.get('inspection'):
            times = sorted({v['time_s'] for v in self.config['inspection']['views']})
        (self.receipts/'startup.json').write_text(json.dumps({'identity':self.session.identity,
            'output_contract':self.contract,'dependency_files':self.dependency_files,'seed':self.seed},indent=2,allow_nan=False))
        return {"type":"ready","time_s":float(self.session.time_s),"tick":int(self.session.tick),
                "scene_revision":self.revision,"timeline":times,"duration_s":times[-1],
                "capabilities":{"simulation_advance":True,"inspection_changes_time":False},
                "dependency_files": self.dependency_files,
                "renderer": {"backend": "Python Warp Render", "hardware_accelerated": not os.environ.get("WORLD_RSI_DEVICE", "cuda:0").startswith("cpu")}}


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    configuration = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
    renderer = None
    try:
        renderer = LiveRenderer(sys.argv[1], configuration)
        protocol.write(json.dumps(renderer.ready(), allow_nan=False) + "\n")
        for line in sys.stdin:
            command = {}
            try:
                if len(line) > 16000:
                    raise ValueError("Observation control message exceeds 16000 bytes")
                command = json.loads(line)
                if command.get("op") == "close":
                    protocol.write(json.dumps({"type": "closed", "request_id": command.get("request_id")}) + "\n")
                    break
                result = renderer.command(command)
                result["request_id"] = command.get("request_id")
                protocol.write(json.dumps(result, allow_nan=False) + "\n")
            except Exception as exc:
                recoverable = recoverable_request_error(exc)
                protocol.write(json.dumps({"type": "error", "error_code": "invalid_request" if recoverable else "native_error",
                                           "fatal": not recoverable, "error": str(exc), "request_id": command.get("request_id")}) + "\n")
    except Exception as exc:
        protocol.write(json.dumps({"type": "error", "error_code": "startup_failed", "fatal": True, "error": str(exc)}) + "\n")
        raise
    finally:
        if renderer is not None:
            renderer.session.close()


if __name__ == "__main__":
    main()
