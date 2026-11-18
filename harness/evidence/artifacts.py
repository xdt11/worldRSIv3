"""Resolve the latest native output of each run independently of acceptance."""
from datetime import datetime
from pathlib import Path
from harness.evidence.archive import inside, load, sha

def _read(path, default=None):
    try:
        value = load(path, default)
        return value if isinstance(value, dict) else default
    except (OSError, ValueError, TypeError):
        return default



def archived_render(task_dir, run, *, for_replay=False, for_scoring=False):
    task_dir = Path(task_dir).resolve()
    if not run:
        return None
    run_dir = inside(task_dir, 'runs/' + run['id'])
    accepted = _read(run_dir / 'render.json', {})
    candidates = []
    for path in run_dir.glob('renders/*/render_manifest.json'):
        try:
            manifest = _read(path, {})
            frames = manifest.get('frames', [])
            # Workers write provenance only after verifying every saved RGB.
            if not frames or not manifest.get('code_sha256') or manifest.get('engine') not in {'genesis','newton'}:
                continue
            if not all(inside(path.parent, frame['path']).is_file() for frame in frames):
                continue
            receipt = _read(path.parent / 'render_receipt.json', {})
            modified = path.stat().st_mtime_ns
            try: generated = datetime.fromisoformat(receipt['timestamp']).timestamp()
            except (KeyError, TypeError, ValueError): generated = modified / 1e9
            candidates.append((generated, modified, path, manifest, receipt))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        return None
    # Older archives predate per-submission receipts; original manifest mtime
    # preserves generation order. Random directory names never establish order.
    ranked = sorted(candidates, key=lambda row: row[:2], reverse=True)
    skipped = 0
    from harness.tools.newton.workers.protocol import validate_output
    contract = _read(run_dir/'output_contract.json') if for_scoring else None
    for skipped, (_, _, path, manifest, receipt) in enumerate(ranked):
        if not for_scoring:
            break
        if not contract or _read(path.parent/'world_export.json', {}).get('status') not in {'ok','render_only'} or not (path.parent/'world/render.mp4').is_file() and not (path.parent/'world/camera.json').is_file():
            continue
        try:
            validate_output(manifest, contract)
            if contract.get('width') and contract.get('height'):
                from PIL import Image
                expected_size = (contract['width'], contract['height'])
                for frame in manifest['frames']:
                    with Image.open(inside(path.parent,frame['path'])) as image:
                        if image.size != expected_size: raise ValueError('Wrong formal output resolution')
        except (OSError, ValueError, KeyError, TypeError): continue
        break
    else:
        return None
    directory = path.parent
    relative = str(directory.relative_to(task_dir))
    if not receipt and accepted.get('path') == relative:
        receipt = accepted
    expected = manifest['code_sha256']
    code = next((p for p in [directory/'solution/scene.py', *run_dir.glob('scene_code_*.py'), run_dir/'scene.py']
                 if p.is_file() and sha(p) == expected), None)
    if code is not None and not receipt.get('world_sha256'):
        # Alignment receipts bind even submissions rejected by the output contract.
        alignment = next((_read(p) for p in run_dir.glob('scene_code_*.alignment.json')
                          if _read(p, {}).get('code_sha256') == expected), {})
        receipt = {**alignment, **receipt}
    world_hash = receipt.get('world_sha256')
    world = next((p for p in [*run_dir.glob('world_state_*.yaml'), run_dir/'world.yaml']
                  if p.is_file() and world_hash and sha(p) == world_hash), None)
    result = {'render_dir': relative, 'render_revision': directory.name + '-' + sha(path)[:16],
        'manifest_sha256': sha(path), 'newer_native_outputs_skipped': skipped,
        'fallback': bool(skipped), 'fallback_reason': 'Newer native outputs lack a valid formal export/output contract' if skipped else None, 'output_contract_validated': bool(for_scoring),
        'selection_reason': 'latest_scorable_native_output' if for_scoring else 'latest_native_output',
        'is_current': run.get('render_dir') == relative,
        'preview_path': str(inside(directory, manifest['frames'][0]['path']).relative_to(task_dir)),
        'frame_count': len(manifest['frames']), 'timestamp_s': manifest['frames'][0].get('timestamp_s'),
        'code_sha256': expected, 'world_sha256': world_hash, 'engine':manifest.get('engine'),
        'code_path': str(code.relative_to(task_dir)) if code else None,
        'scene_path': str(code.relative_to(task_dir)) if code else None,
        'world_path': str(world.relative_to(task_dir)) if world else None}
    if manifest.get('inspection'):
        result['inspection'] = True
        if (directory/'comparison.json').is_file():
            result['comparison_path'] = relative + '/comparison.json'
    if for_replay and code is None:
        raise ValueError('Code matching the latest native render hash is missing; replay refused')
    if for_replay and world_hash and world is None:
        raise ValueError('World matching the latest native render hash is missing; replay refused')
    return result
