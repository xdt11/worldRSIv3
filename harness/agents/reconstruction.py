"""Fixed reconstruction role: context and tools for one native agent session."""
from harness.agents.common import prepare_agent
import json
from time import perf_counter
from harness.evidence.archive import inside,load
from harness.evidence.context import image_item,presentation_metadata,public_source_id,evidence_label


def prepare(run,box,*,budget=None,stagnation=None,before_call=None,emit=None,archive_result=None,
            prompt_key='agent',observation_builder=None):
    task=run.task
    prompt,items=build_context(task,box=box,run=run,prompt_key=prompt_key,
                              observation_builder=observation_builder)
    sandbox_config=getattr(box,'config',task.config.get('sandbox',{}))
    gpu=sandbox_config.get('gpus')
    items[0]['content'].append({'type':'input_text','text':json.dumps({
        'sandbox_gpu_exposure':gpu,
        'perception_availability':{
            'estimate_depth':'CPU supported; CUDA requires an exposed GPU.',
            'segment_image':('CUDA GPU exposure requested; actual CUDA/model failures remain explicit.' if gpu else
                             'Unavailable in this sandbox: no GPU exposed. SAM3 does not support CPU fallback.')}
    },ensure_ascii=False)})
    tools=task.services.tools
    schemas=tools.declarations()
    if task.record.get('kind') == 'threat':
        from harness.agents.threat import edit_tool_declarations
        schemas = edit_tool_declarations(schemas)
    return prepare_agent(task_directory=task.directory,task_id=task.id,run_id=run.id,
        role='reconstruction',config=task.config,api_key=task.services.store.api_key(),
        prompt=prompt,items=items,tools=schemas,
        execute=lambda name,args:tools.execute(run,box,name,args),budget=budget,
        stagnation=stagnation,stopped=task.stopped,before_call=before_call,emit=emit,
        archive_result=archive_result,skills_dir=task.directory/'skills',
        finish_validate=lambda args:tools.validate_finish(run,box,args))


def output_contract(context):
    if context.record.get('kind') == 'threat':
        from harness.tools.newton.workers.protocol import edit_inspection
        stored = context.record.get('output_contract', {})
        return edit_inspection(stored.get('views') if stored.get('kind') == 'scene_edit' else None)
    reference = next((o for o in input_sources(context) if o['is_primary_reference']),None)
    evaluation = context.config.get('evaluation',{})
    kind = reference['kind'] if reference else 'unconstrained'
    contract = {'schema_version':1,'kind':kind,
                'reference_source_id':reference['source_id'] if reference else None,
                'timestamps_s':evaluation.get('video_timestamps_s') if kind=='video' else None,
                'timestamp_tolerance_s':evaluation.get('video_max_timestamp_error_s')}
    metadata=context.record.get('benchmark') or context.record.get('media_reference')
    if metadata:
        contract.update(protocol='4dcodebench-newton-v1' if context.record.get('benchmark') else 'newton-native-v1',
            **{key: value for key, value in metadata.get('video', {}).items()
               if key in {'width', 'height', 'fps', 'frame_count'}})
        if context.record.get('media_reference'):
            contract['reference_timeline']={
                'evaluation_clock':'normalized_evaluation_copy','time_origin_s':0,
                'normalization':'CFR/H.264 evaluation copy; original source bytes and decoded PTS are retained separately.',
                'source_id':contract['reference_source_id'],
                'source_clock':'Original measured decoded PTS, including any nonzero source start time.',
                'frame_mapping':'Not established. CFR conversion can duplicate or drop source frames; do not assume a per-frame bijection or equal timestamp values.',
                'required_alignment':'Inspect source PTS and declare source-to-world time mappings for geometry and camera samples. Native output follows timestamps_s in the evaluation clock.',
                'scope':metadata.get('scope')}
        contract['formal_export'] = {
            'renderer': 'Newton + Python Warp Render; an explicit variant of the Blender reference protocol',
            'camera': ('The selected benchmark export profile requires a fixed pinhole camera. '
                       'This is an adapter constraint, not a restriction on source-camera reconstruction.'
                       if context.record.get('benchmark') else
                       'Source and evaluation cameras may have trajectories and changing intrinsics; preserve actual per-frame calibration.'),
            'time': 'Every output timestamp must equal actual native simulation time. Rendering and world data share the full reference timeline.',
            'capability': 'Export actual native geometry/state and camera trajectories. Unsupported metric profiles remain explicit; never fabricate trajectories.',
            'receipt': 'Inspect world_export.json in the submitted artifact directory for export status and errors.',
            'checker': 'The public checker is available under /opt/4dcodebench. Example: PYTHONPATH=/opt/4dcodebench python -c \'from checker.check import check_world; print(check_world("/work/OUTPUT/world"))\'. Use the actual exported world path. Checker success is format compatibility, not full official rendering compliance.'}
    return contract

def input_sources(context):
    sources=[]
    for index,original in enumerate(context.record.get('inputs',[])):
        source=dict(original)
        source['source_id']=public_source_id(source,index)
        source['is_primary_reference']=source.get('path')==context.record.get('reference')
        sources.append(source)
    return sources


def _initial_evidence(context,sources,box=None,run=None):
    """Prepare bounded public-source evidence; no provider invocation is made."""
    policy=context.config.get('runtime',{})
    per_video=policy.get('overview_frames_per_video',3)
    if type(per_video) is not int or per_video<1:
        raise ValueError('overview_frames_per_video must be a positive integer')
    content=[];receipts=[]
    start=perf_counter()
    for source in sources:
        if source['kind']=='image':
            raw=inside(context.directory,source['local_path']).read_bytes()
            record={'source_id':source['source_id'],'source_path':source['sandbox_path'],
                    'kind':'reference_image','source_sha256':source['sha256'],
                    'capture_time':{'status':'unknown','value':None},'camera_id':None,
                    'evidence_id':'source_'+source['sha256'],**presentation_metadata(raw)}
            content.extend([{'type':'input_text','text':'EVIDENCE_GROUP '+json.dumps(evidence_label(record),ensure_ascii=False)},image_item(raw)])
            receipts.append({**record,'status':'provided'})
        elif box is None or run is None:
            receipts.append({'source_id':source['source_id'],'status':'not_prepared',
                             'reason':'video pixels require actual container decoding'})
        elif not hasattr(box,'run_media_tool'):
            receipts.append({'source_id':source['source_id'],'status':'unsupported',
                             'reason':'This sandbox does not expose public media preparation; no video pixels were decoded.'})
        else:
            from harness.tools import probe_media,extract_frames
            try:
                probe,_=probe_media.execute(run,box,path=source['sandbox_path'])
                if probe.get('exit_code',0) or probe.get('error'):raise ValueError(str(probe.get('error') or probe.get('stderr')))
                metadata=probe.get('result',{})
                first,last=metadata['valid_timestamp_range_s'];count=per_video
                timestamps=([first] if count==1 else sorted({first+(last-first)*i/(count-1) for i in range(count)}))
                result,visual=extract_frames.execute(run,box,path=source['sandbox_path'],timestamps=timestamps)
                if result.get('exit_code',0) or result.get('error'):raise ValueError(str(result.get('error') or result.get('stderr')))
                content.extend(visual or [])
                receipts.append({'source_id':source['source_id'],'status':'provided','probe':metadata,
                                 'images':result.get('images',[]),'decoder_logs':{
                                     key:result[key] for key in ('stdout_path','stderr_path') if key in result}})
            except (ValueError,KeyError,TypeError,OSError) as exc:
                receipts.append({'source_id':source['source_id'],'status':'error','reason':str(exc)})
    receipt={'role':'host_input_preparation','elapsed_s':perf_counter()-start,'model_api_calls':0,
             'provided_image_count':sum(p['type']=='input_image' for p in content),
             'sources':receipts,'scope':'Public input overview. This is neither reconstruction output nor formal evaluation.'}
    if run is not None:
        path=run.directory/'input_preparation.json';path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        target=context.directory/'work/.evidence'/f'input_preparation_{run.directory.name}.json'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(path.read_text())
        receipt_path='/work/'+str(target.relative_to(context.directory/'work'))
    else:receipt_path=None
    summary={'provided_image_count':receipt['provided_image_count'],'receipt_path':receipt_path}
    unavailable=[{'source_id':row['source_id'],'status':row['status'],'reason':row.get('reason')}
                 for row in receipts if row['status']!='provided']
    if unavailable:summary['unavailable']=unavailable
    return content,summary


def observation_items(context,box=None,run=None):
    value=context.record
    contract=output_contract(context)
    times=contract.get('timestamps_s') or []
    output={key:contract[key] for key in ('kind','width','height','fps','frame_count','timestamp_tolerance_s') if key in contract}
    output.update(path='/work/output_contract.json',frame_count=len(times) or contract.get('frame_count'),
        timestamp_range_s=[times[0],times[-1]] if times else None)
    if contract.get('reference_timeline'):
        output['clock']='Normalized comparison copy starts at zero; source PTS may differ. Read reference_timeline in the contract.'
    sources=input_sources(context)
    pixels,preparation=_initial_evidence(context,sources,box=box,run=run)
    payload={'instruction':value['instruction'],'instruction_source_id':'instruction',
        'source_index':{'instruction':{'kind':'text'},
            **{source['source_id']:{'kind':source['kind'],'path':source['sandbox_path'],'sha256':source['sha256'],
                'primary_reference':source['is_primary_reference']} for source in sources}},
        'primary_reference_source_id':next((source['source_id'] for source in sources if source['is_primary_reference']),None),
        'world_schema_version':3,'output_contract':output,'seed':value['config'].get('seed'),
        'input_preparation':preparation}
    if value.get('benchmark'):
        payload['input_scope']='Public benchmark reference only; private scoring data is unavailable.'
    elif value.get('media_reference'):
        payload['input_scope']='User media; no official 3D truth or VQA bank.'
    content=[{'type':'input_text','text':json.dumps(payload,ensure_ascii=False)}]
    content.extend(pixels)
    return [{'role':'user','content':content}]

def build_context(context,box=None,run=None,*,prompt_key='agent',observation_builder=None):
    snapshot = load(context.directory / 'snapshot.json')
    if snapshot != context.services.versions.snapshot(context.record['version']):
        raise ValueError('Task snapshot differs from its pinned Git version; model call refused')
    spec = context.services.spec
    prompt = inside(context.services.root, spec.prompts[prompt_key]).read_text()
    if spec.documents:
        prompt += '\n\nRead as needed: '+', '.join('/opt/worldrsi/'+path.rsplit('/',1)[-1] for path in spec.documents)+'.'
    prompt += (' Native API: /opt/worldrsi/newton-api.json; installed examples: '
               '/opt/newton-env/lib/python3.12/site-packages/newton/examples. '
               'Read relevant signatures/examples, not entire source trees.')
    context.services.versions.materialize(context.record['version'], context.directory / 'skills')
    prompt += ('\n\nPinned skills '+context.record['version']+
               ': read indexed /skills/<name>/SKILL.md as needed. They are read-only; '
               'copy implementations to /work for changes. Helpers run only in the sandbox.')
    memory = context.directory / 'local_memory.md'
    if memory.exists(): prompt += '\n\nTASK-LOCAL MEMORY\n' + memory.read_text()
    items = (observation_builder or observation_items)(context,box=box,run=run)
    if len(context.record['runs']) > 1:
        available=['/work/'+name for name in ('world.yaml','scene.py','local_memory.md')
                   if (context.directory/'work'/name).is_file()]
        manifests=sorted((context.directory/'work').glob('renders/*/render_manifest.json'),
                         key=lambda path:path.stat().st_mtime,reverse=True)
        available+=['/'+str(path.relative_to(context.directory)) for path in manifests[:3]]
        previous=context.record['runs'][-2]
        failure=previous.get('failure') or {}
        items.append({'role':'user','content':json.dumps({
            'previous_run_outcome':{key:previous.get(key) for key in ('id','status','error')},
            'previous_failure':{key:failure[key] for key in ('code','message') if key in failure},
            'previous_artifacts':available,'additional_render_manifests':max(0,len(manifests)-3),
            'artifact_access':'Current working files and the three most recent render manifests are listed. Other renders remain under /work/renders; archived output may precede later world/code changes.'},ensure_ascii=False)})
    return prompt,items
