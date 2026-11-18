"""Explicit project tools, bound to native runtime Tool objects by role builders."""
from harness.evidence.context import tool
from harness.tools.bash import bash
from harness.tools.evidence import view_image
from harness.tools import probe_media,extract_frames,crop_image,estimate_depth,segment_image
from copy import deepcopy
from harness.tools.world_state import set_world_state
from harness.tools.newton.tool import submit_scene,scene_state,observe_scene,ensure_current
from harness.core.contracts import ToolError


def validate_finish(context, sandbox, arguments):
    diagnostic = ensure_current(context, sandbox)
    if diagnostic:
        raise ToolError(diagnostic['code'], diagnostic['message'],
                        changed=diagnostic.get('changed'), missing=diagnostic.get('missing'))
    if context.state.get('output_contract', {}).get('kind') == 'scene_edit':
        comparison = context.state.get('edit_comparison') or {}
        if not context.state.get('rendered') or not comparison.get('implementation_changed') or not (
                comparison.get('pixels_changed') or comparison.get('native_changed')):
            raise ToolError('edit_effect_unverified',
                            'Submit a current scene with a measured native change or a visible change in matched views. '
                            'Choose inspection times and views that demonstrate the requested effect.')


def _definitions():
    entries=[
        ('bash','Run a focused Bash command inside the task container, normally from /work. '
         'Use python for the pinned Newton/Warp environment and targeted API/file queries. '
         'Full stdout/stderr are archived. Use max_output_chars when reading relevant documentation; '
         'if truncated, read a focused section from stdout_read_path instead of repeating the full command. '
         'For finite search output use sed -n to print a range without closing the pipe early.',
         {'command':{'type':'string'},'timeout_s':{'type':'integer','minimum':1},
          'max_output_chars':{'type':'integer','minimum':1024,'maximum':65536,
            'description':'Character allowance for stdout/stderr feedback; original logs are always archived.'}},['command'],bash),
        ('view_image','Read one image path or a nonempty batch of paths, exactly one of path/paths. '
         'Returns actual pixels; use it for saved renders, depth previews or masks. Paths must be under /input or /work.',
         {'path':{'type':'string'},'paths':{'type':'array','items':{'type':'string'},'minItems':1}},[],view_image),
        ('set_world_state','Validate world.yaml from inline yaml or a /work file path, exactly one. '
         'Use schema_version: 3, document_role: world_hypothesis and stable semantic/source IDs. '
         'Evidence references declared source IDs; keep real parameters and simulation choices distinct. '
         'Changing the accepted world invalidates its current formal render.',
         {'yaml':{'type':'string'},'path':{'type':'string'}},[],set_world_state),
        ('submit_scene','Run a native Newton Example(viewer, args) from inline code or a /work source path, exactly one, '
         'against the accepted world. Define the ordinary Example class with model, current native state, '
         'sim_time and step(); write geometry and solver composition directly with Newton/Warp. '
         'The worker samples actual state and saves Warp frames from the frozen /work/output_contract.json. '
         'A returned error leaves no valid current render. '
         'Inspect the returned image paths with view_image before concluding.',
         {'code':{'type':'string'},'path':{'type':'string'}},[],submit_scene),
        ('scene_state','Preview the current /work/scene.py against accepted world.yaml; formal submit is not required. '
         'Use its persistent Newton state: query, seek to time_s, '
         'advance by seconds, reset from the specified seed/initial state, step one control interval, or close. '
         'Read actual native time/tick; unsupported capabilities remain explicit. Query/close do not advance time.',
         {'operation':{'type':'string','enum':['query','seek','advance','reset','step','close'],'default':'query'},
          'time_s':{'type':'number'},'seconds':{'type':'number'},
          'action':{'type':['object','array','null']},'seed':{'type':['integer','null'],'minimum':0},
          'initial_state':{'type':['object','null']}},[],scene_state),
        ('observe_scene','Preview actual pixels from current /work/scene.py and accepted world.yaml, without formal submit. '
         'Omit time_s to preserve current state; camera-only changes do not advance time. '
         'Returns pixels, actual time, revision and image paths. Request physical measurements separately with scene_state. '
         'Camera position/target/up are world-space 3-vectors. Observations do not replace formal output.',
         {'time_s':{'type':'number'},'camera':{'type':'object','properties':{
             name:{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3}
             for name in ('position','target','up')},'required':['position','target','up'],'additionalProperties':False},
          'width':{'type':'integer','minimum':1},'height':{'type':'integer','minimum':1}},[],observe_scene)]
    result={}
    for name,description,properties,required,handler in entries:
        schema=tool(name,description,properties,required)
        if name in {'view_image','set_world_state','submit_scene'}:
            alternatives={'view_image':('path','paths'),'set_world_state':('yaml','path'),
                          'submit_scene':('code','path')}[name]
            schema['parameters']['oneOf']=[{'required':[field]} for field in alternatives]
        result[name]=(schema,handler)
    # Keep observation tools adjacent in the wire declaration for easy discovery.
    media=(probe_media,extract_frames,crop_image,estimate_depth,segment_image)
    return {**{'bash':result.pop('bash')},
            **{module.NAME:(deepcopy(module.SCHEMA),module.execute) for module in media},**result}


def declarations():
    return [schema for schema,_ in _definitions().values()]


def execute(context,sandbox,name,arguments):
    definitions=_definitions()
    if name not in definitions:raise ValueError('Tool is not available: '+name)
    _,handler=definitions[name]
    return handler(context,sandbox,**arguments)
