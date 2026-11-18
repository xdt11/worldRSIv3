"""Matched source and candidate observations for a safeRSI scene submission."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from harness.evidence.archive import inside, load, save, sha
from harness.evidence.context import image_item
from harness.tools.newton.workers.protocol import validate_output
from harness.tools.newton.workers.solution import checked_file


def _baseline(ctx, box, inspection):
    source = ctx.task.record['source_scene']
    identity = [source['render_revision'], ctx.task.config.get('seed'), inspection]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = ctx.directory / 'inspections'
    root.mkdir(exist_ok=True)
    destination = root / key
    if destination.is_dir():
        return destination
    with TemporaryDirectory(dir=root) as temporary:
        staging = Path(temporary)
        work = staging / 'work'
        baseline = inside(ctx.task.directory, source['baseline_path'])
        for item in source['files']:
            original = checked_file(baseline, item['path'])
            target = inside(work, item['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if sha(target) != item['sha256']:
                raise ValueError('Frozen source dependency changed: ' + item['path'])
        config = {**ctx.task.config, 'sandbox':box.config}
        with ctx.services.sandbox_factory(staging, config, purpose='viewer', cache_dir=box.cache_dir or root/'.cache') as replay:
            run = replay.execute_newton((work/'scene.py').read_text(), inspection=inspection)
            if run['exit_code']:
                raise ValueError('Source inspection failed: ' + run['stderr'][-2000:])
        rendered = inside(work, Path(run['output_dir']).relative_to('/work'))
        manifest = load(rendered/'render_manifest.json')
        validate_output(manifest, inspection)
        expected = {item['path']:item['sha256'] for item in source['files']}
        actual = {item['path']:item['sha256'] for item in manifest['dependency_files']}
        if expected != actual:
            raise ValueError('Source inspection executed a different dependency closure')
        frozen = staging / 'baseline'
        frozen.mkdir()
        for name in ['render_manifest.json', 'runtime_state.json', *(f['path'] for f in manifest['frames'])]:
            shutil.copyfile(inside(rendered, name), inside(frozen, name))
        frozen.rename(destination)
    return destination


def compare_edit(ctx, box, artifact, manifest):
    baseline = _baseline(ctx, box, manifest['inspection'])
    before = load(baseline/'render_manifest.json')
    if before['effective_seed'] != manifest['effective_seed']:
        raise ValueError('Before and after inspections need the same seed')
    def relative(path):
        return str(path.relative_to(ctx.task.directory))
    def implementation(files):
        return {item['path']:item['sha256'] for item in files if item['path'] != 'output_contract.json'}
    pairs = []
    for old, new in zip(before['frames'], manifest['frames'], strict=True):
        if (not math.isclose(old['timestamp_s'], new['timestamp_s'], rel_tol=0, abs_tol=1e-6) or
                any(old['camera'][k] != new['camera'][k] for k in ('intrinsics','camera_to_world','width','height'))):
            raise ValueError('Before and after inspections need identical cameras and simulation times')
        pairs.append({'time_s':new['timestamp_s'], 'before':relative(inside(baseline, old['path'])),
                      'after':relative(inside(artifact, new['path'])),
                      'pixels_changed':old['pixel_sha256'] != new['pixel_sha256'],
                      'native_changed':old['physics_sha256'] != new['physics_sha256']})
    comparison = {'source_revision':ctx.task.record['source_scene']['render_revision'],
                  'seed':manifest['effective_seed'], 'edited_manifest_sha256':sha(artifact/'render_manifest.json'), 'views':pairs,
                  'source_native_state':relative(baseline/'runtime_state.json'),
                  'edited_native_state':relative(artifact/'runtime_state.json'),
                  'native_changed':any(pair['native_changed'] for pair in pairs),
                  'pixels_changed':any(pair['pixels_changed'] for pair in pairs),
                  'implementation_changed':implementation(before['dependency_files']) != implementation(manifest['dependency_files'])}
    save(artifact/'comparison.json', comparison)
    return comparison


def comparison_images(directory, comparison, indices):
    content = []
    for index in indices:
        view = comparison['views'][index]
        for side in ('before', 'after'):
            content.append({'type':'input_text','text':f"View {index + 1}, {view['time_s']} s, {side}: {view[side]}"})
            content.append(image_item(inside(directory, view[side]).read_bytes()))
    return content
