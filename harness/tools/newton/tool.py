"""Scene submission and native observation tools with immutable evidence."""
import hashlib
import json
import math
import shutil
from harness.evidence.archive import inside, load, save, sha, now, uid
from harness.evidence.context import image_item
from harness.tools.newton.workers.protocol import edit_inspection, validate_output, OutputContractError


def _dependency_status(box, expected):
    script = ("import json,pathlib,sys;"
              "from harness.tools.newton.workers.solution import dependency_status;"
              "print(json.dumps(dependency_status(pathlib.Path('/work'),json.loads(sys.argv[1]))))")
    checked = box._exec(['python', '-c', script, json.dumps(expected)], timeout_s=30)
    return json.loads(checked['stdout']) if checked['exit_code'] == 0 else {
        'current': False, 'reason': 'Dependency inspection failed; inspect archived logs.'}


def ensure_current(ctx, box):
    """Check accepted inputs, without treating unrelated workspace files as inputs."""
    state = ctx.state
    expected = state.get('dependency_files')
    if not state.get('rendered') or not expected:
        return None
    status = _dependency_status(box, expected)
    if status.get('current'):
        return None
    previous = state.get('render_dir')
    state.update(rendered=False, render_dir=None, render_hash=None)
    if hasattr(box, 'close_scene'):
        box.close_scene()
    diagnostic = {'code': 'scene_dependencies_changed', 'render_current': False,
        'message': 'An input used by the accepted scene changed or is missing. Submit the scene again before finishing.',
        'previous_render_dir': previous, **{key: value for key, value in status.items() if key != 'current'}}
    state['render_invalidation'] = diagnostic
    return diagnostic

def _source(box, inline, path, name):
    if (inline is None) == (path is None): raise ValueError(f'Provide {name} or path, exactly one')
    return box.read_file(path) if path is not None else inline


def _same_output_contract(candidate, frozen):
    """Only timestamp round-off may differ, under the controller's own tolerance."""
    if not isinstance(candidate,dict) or not isinstance(frozen,dict) or candidate.keys()!=frozen.keys():
        return False
    try:
        other=lambda value:json.dumps({key:item for key,item in value.items() if key!='timestamps_s'},
                                     sort_keys=True,allow_nan=False)
        if other(candidate)!=other(frozen): return False
    except (TypeError,ValueError): return False
    if 'timestamps_s' not in frozen: return True
    actual,expected=candidate['timestamps_s'],frozen['timestamps_s']
    if actual is None or expected is None: return actual is expected
    if not isinstance(actual,list) or not isinstance(expected,list) or len(actual)!=len(expected):
        return False
    tolerance=frozen.get('timestamp_tolerance_s',0)
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or tolerance<0:
        return False
    for values in (actual,expected):
        if (any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in values)
                or any(a>=b for a,b in zip(values,values[1:]))):
            return False
    return all(abs(a-b)<=tolerance for a,b in zip(actual,expected))


def submit_scene(ctx, box, code=None, path=None, inspection_views=None):
    code = _source(box, code, path, 'code')
    if hasattr(box, 'close_scene'): box.close_scene()
    state = ctx.state
    state.update(rendered=False, render_dir=None, render_hash=None)
    state.pop('edit_comparison', None)
    contract = state.get('output_contract', {'schema_version':1,'kind':'unconstrained'})
    editing = contract['kind'] == 'scene_edit'
    if inspection_views is not None and not editing:
        raise ValueError('Custom inspection views are available for scene edits')
    if editing and inspection_views is not None:
        contract = edit_inspection(inspection_views)
    if not state.get('world'):
        raise ValueError('Submit and validate World State Description before scene code')
    if hashlib.sha256(box.read_file('world.yaml').encode()).hexdigest() != state['world_hash']:
        raise ValueError('Working world.yaml changed; submit it through set_world_state before scene code')
    contract_sha = None
    if 'output_contract' in state and not editing:
        contract_text = box.read_file('output_contract.json')
        candidate=json.loads(contract_text)
        if not _same_output_contract(candidate,state['output_contract']):
            raise ValueError('Working output_contract.json differs from the controller contract; restore the provided contract before submitting')
        if candidate!=state['output_contract']:
            # Restore the published controller timeline before rendering and
            # hashing. Never modify the historical read-only source contract.
            box.write_file('output_contract.json',json.dumps(state['output_contract']))
            contract_text=box.read_file('output_contract.json')
            if json.loads(contract_text)!=state['output_contract']:
                raise ValueError('Could not restore the controller output contract')
        contract_sha = hashlib.sha256(contract_text.encode()).hexdigest()
    checks = ctx.services.world.align(code, state['world'])
    code_sha = hashlib.sha256(code.encode()).hexdigest()
    state['code_hash'] = code_sha
    code_path = ctx.directory / f'scene_{uid("code")}.py'
    code_path.write_text(code)
    (ctx.directory / 'scene.py').write_text(code)
    options = {'inspection':contract} if editing else {}
    result = ctx.services.renderer.execute(box, code, **options)
    visual = None
    checks.update(world_sha256=state['world_hash'], code_sha256=code_sha, timestamp=now())
    save(code_path.with_suffix('.alignment.json'), checks)
    if result['exit_code'] == 0:
        manifest = result.get('manifest', {})
        if (manifest.get('code_sha256') != code_sha or manifest.get('world_sha256') != state['world_hash']
                or contract_sha is not None and (manifest.get('output_contract_sha256') != contract_sha
                or manifest.get('reference_source_id') != state['output_contract'].get('reference_source_id'))):
            return {**result, 'error': {'code': 'native_identity_mismatch',
                    'message': 'Executed world/code/output contract identity differs from the accepted submission'}, 'alignment': checks}, None
        source = inside(ctx.task.directory, result['output_dir'].replace('/work/', 'work/', 1))
        if any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Render archives cannot contain symlinks')
        destination = ctx.directory / 'renders' / source.name
        shutil.copytree(source, destination)
        render_dir = str(destination.relative_to(ctx.task.directory))
        render_hash = sha(destination/'render_manifest.json')
        result['archive_dir'] = render_dir
        scene_revision = 'newton:' + source.name
        result['scene_revision'] = scene_revision
        export_path = destination / 'world_export.json'
        export = load(export_path)
        if export is not None:
            result['world_export'] = {**export, 'receipt_sha256':sha(export_path),
                'world_revision':state['world_hash'], 'scene_revision':scene_revision,
                'artifact_sha256':{str(p.relative_to(destination)):sha(p)
                    for folder in ('world','solution') for p in (destination/folder).rglob('*') if p.is_file()}}
            state['dependency_files'] = export.get('solution', {}).get('files', [])
        receipt = {'path':render_dir, **checks, 'world_path':state['world_path'],
            'code_path':str(code_path.relative_to(ctx.task.directory)),
            'scene_revision':scene_revision, 'world_revision':state['world_hash']}
        save(destination / 'render_receipt.json', receipt)
        try:
            result['output_contract'] = validate_output(manifest or load(destination/'render_manifest.json'), contract)
        except OutputContractError as exc:
            return {**result,'error':exc.diagnostic,'alignment':checks}, None
        if editing:
            from .inspection import compare_edit, comparison_images
            from harness.evidence.review import uniform_indices
            comparison = compare_edit(ctx, box, destination, manifest)
            state['edit_comparison'] = comparison
            state['output_contract'] = contract
            save(ctx.directory/'output_contract.json', contract)
            result['edit_comparison'] = comparison
            visual = comparison_images(ctx.task.directory, comparison, uniform_indices(len(comparison['views']), 3))
        save(ctx.directory / 'render.json', {'path':render_dir, **checks,
             'world_path':state['world_path'], 'code_path':str(code_path.relative_to(ctx.task.directory)),
             'scene_revision':scene_revision, 'world_revision':state['world_hash'],
             'world_export':result.get('world_export')})
        state.update(render_dir=render_dir, render_hash=render_hash, rendered=True)
    return {**result, 'alignment':checks}, visual


def _scene(ctx, box):
    from harness.tools.newton.session import AgentSession
    if not ctx.state.get('world'):
        raise ValueError('Accept world.yaml through set_world_state before previewing')
    world_hash = hashlib.sha256(box.read_file('world.yaml').encode()).hexdigest()
    if world_hash != ctx.state['world_hash']:
        raise ValueError('Working world.yaml changed; accept it through set_world_state before previewing')
    code = box.read_file('scene.py')
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    session = getattr(box, 'scene_session', None)
    if (session is None or session.closed or session.code_hash != code_hash or session.world_hash != world_hash
            or not _dependency_status(box, session.dependency_files).get('current')):
        ctx.services.world.align(code, ctx.state['world'])
        if session is not None: session.close()
        session = box.scene_session = AgentSession(ctx, box, code_hash=code_hash, world_hash=world_hash)
        session.initialize()
    return session


def scene_state(ctx, box, operation='query', time_s=None, seconds=None, action=None, seed=None, initial_state=None):
    if operation not in ('query','seek','advance','reset','step','close'): raise ValueError('Unknown scene operation')
    if operation == 'close':
        if hasattr(box, 'close_scene'): box.close_scene()
        return {'closed':True}, None
    session = _scene(ctx, box)
    result = session.request({'op':operation, 'time_s':time_s, 'seconds':seconds, 'action':action, 'seed':seed, 'initial_state':initial_state})
    return {**result, 'world_sha256':session.world_hash, 'code_sha256':session.code_hash,
            'scene_revision':session.revision, 'kind':'observation_not_evaluation'}, None


def observe_scene(ctx, box, time_s=None, camera=None, width=640, height=360):
    from harness.tools.newton.session import render_command
    session = _scene(ctx, box)
    command = render_command({'type':'render','time_s':time_s if time_s is not None else 0,'camera':camera,
                              'width':width,'height':height}, {'width':640})
    if time_s is None:command.pop('time_s')  # Rendering the current state never needs a full query.
    result, raw = session.render(command)
    archive = ctx.directory / 'observations' / (uid('view')+'.jpg')
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(raw)
    observation_path = '/work/.observations/' + archive.name
    box.write_bytes(observation_path, raw)
    receipt = json.loads(box.read_file(result['receipt_path']))
    receipt.update(archive_path=str(archive.relative_to(ctx.task.directory)),path=observation_path,
                   sha256=hashlib.sha256(raw).hexdigest(),world_sha256=session.world_hash,
                   code_sha256=session.code_hash,scene_revision=session.revision)
    if 'requested_time_s' in result:receipt['requested_time_s']=result['requested_time_s']
    receipt['image']['archive_path']=receipt['archive_path']
    save(archive.with_suffix('.json'),receipt)
    result = {'path':observation_path,'archive_path':receipt['archive_path'],
              'time_s':result['time_s'],'scene_revision':session.revision}
    if 'requested_time_s' in receipt:result['requested_time_s']=receipt['requested_time_s']
    from harness.tools.evidence import visual_groups
    return result, visual_groups([result], [image_item(raw)])
