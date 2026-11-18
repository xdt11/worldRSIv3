"""Independent experiment author over a frozen safeRSI scene delivery.

Orchestration owns the rollout directory, tools, execution and receipt checks.
This role prepares an ordinary native agent invocation; it never changes the
scene task's run or broadens its actor's tools.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.agents.common import prepare_agent
from harness.agents.reconstruction import input_sources
from harness.evidence.archive import inside, load


FINISH_SCHEMA = {'type':'object','description':'Submit the setup for one controller-executed rollout and end this evaluator invocation.','properties':{
    'setup_path':{'type':'string','minLength':1,'description':'The prepared /work setup JSON to execute once.'}},
    'required':['setup_path'],
    'additionalProperties':False}


def prepare(run, box, *, experiment, tools, execute, finish_validate,
            directory, budget=None, before_call=None, emit=None, archive_result=None):
    """Bind controller tools to an evaluator whose invocation has its own archive.

    ``directory`` is this experiment's root, not the source task/run directory.
    The terminal tool submits a validated setup. The controller owns execution
    and the resulting archive after the agent hands over.
    """
    task = run.task
    if task.record.get('kind') != 'threat':
        raise ValueError('The experiment evaluator requires a safeRSI task')
    if not callable(finish_validate):
        raise ValueError('A controller setup validator is required')
    directory = Path(directory).resolve()
    if directory in {task.directory.resolve(), run.directory.resolve()}:
        raise ValueError('Experiment invocation archives must be separate from scene runs')
    rollout_id = experiment.get('rollout_id')
    if not isinstance(rollout_id,str) or not rollout_id:
        raise ValueError('The controller must assign a rollout_id')
    snapshot = load(task.directory/'snapshot.json')
    if snapshot != task.services.versions.snapshot(task.record['version']):
        raise ValueError('Task skills differ from their pinned version')
    prompt_path = task.services.spec.prompts.get('rollout_evaluator','harness/prompts/rollout_evaluator.md')
    prompt = inside(task.services.root,prompt_path).read_text()
    payload = {'task_id':task.id,'scene_run_id':run.id,'task_kind':'threat',
        'instruction':task.record['instruction'],'experiment':experiment,
        'instruction_source':'The preceding user message is the verbatim instruction supplied to the threat-scene RSI actor. Use it to resolve the original experiment intent, not a scene-generation summary.',
        'source_media':[{'source_id':row['source_id'],'kind':row['kind'],
                         'sandbox_path':row['sandbox_path'],'sha256':row['sha256']}
                        for row in input_sources(task)],
        'boundary':'Construct and execute only this experiment. The scene-only task delivery is immutable.',
        'policy_contract':{'current_prototype':True,'embodiment':'Unitree Go2','navigation':'X-NavDP point-goal',
            'navigation_hz':0.5,'locomotion':'MoE-CTS Go2 student',
            'scope':'Currently available execution profile; report unsupported requests rather than silently replacing the requested embodiment or policy.',
            'vision':'Navigation uses the initial gpt-image-2 reference at time zero and decoded AgentGarten RGB thereafter. Depth always comes directly from Warp. Raw first-person Warp RGB is archived for comparison.'}}
    cfg = dict(task.config)
    cfg['model'] = cfg.get('rollout_evaluator_model') or cfg.get('critic_model') or cfg['model']
    return prepare_agent(task_directory=directory,task_id=task.id,run_id=rollout_id,
        role='rollout_evaluator',config=cfg,api_key=task.services.store.api_key(),
        prompt=prompt,items=[{'role':'user','content':task.record['instruction']},
                            {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
        tools=tools,execute=execute,budget=budget,stopped=task.stopped,
        before_call=before_call,emit=emit,archive_result=archive_result,
        finish_schema=FINISH_SCHEMA,skills_dir=task.directory/'skills',finish_validate=finish_validate,
        finish_name='execute_rollout')
