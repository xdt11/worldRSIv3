"""Adapt the pinned official VQA functions, questions and score command."""
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import threading

SCORE_LOCK=threading.RLock()

from harness.evidence.archive import save, sha, load, inside

def current_recheck(directory):
    directory=Path(directory)
    pointer=load(directory/'evaluation/vqa-rechecks/latest.json')
    if not pointer:return None
    path=inside(directory,pointer['result_path'])
    if sha(path)!=pointer['sha256']:raise ValueError('VQA recheck evidence changed')
    result=load(path)
    if result.get('status')!='ok':raise ValueError('Only successful VQA rechecks may replace the displayed measurement')
    return result,path.parent

def effective_task(task,directory):
    selected=current_recheck(directory)
    if not selected:return task
    result,folder=selected
    metrics={**task.get('metrics',{}),**result['metrics']}
    from evaluation.groups import metric_groups
    evaluation={**task.get('evaluation',{}),'vqa':result,'metrics':metrics,'metric_groups':metric_groups(metrics),
                'vqa_recheck_path':str(folder.relative_to(directory)),
                'vqa_recheck_note':'Supplemental VQA measurement; original pre-RSI evaluation remains archived unchanged.'}
    return {**task,'metrics':metrics,'evaluation':evaluation}

@lru_cache(maxsize=4)
def official(source):
    path=Path(source)/'vlm_judge/judge.py'
    spec=importlib.util.spec_from_file_location('worldrsi_official_vqa',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def score(source, scenes, rows, directory, system):
    """Native denominator and presence gates, restricted to the frozen sample set."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    save(directory/'questions.json',{'scenes':scenes})
    # A sentinel establishes a system even when every render is absent/gated.
    normalized=[{**row,'system':system} for row in rows] or [
        {'system':system,'scene':'','id':'','category':'sentinel','correct':False}]
    (directory/'answers.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in normalized))
    output=StringIO()
    with SCORE_LOCK, redirect_stdout(output):
        official(str(source)).cmd_score(SimpleNamespace(answers=str(directory/'answers.jsonl'),
            questions=str(directory/'questions.json'),no_presence=False))
    text=output.getvalue();(directory/'score.txt').write_text(text)
    values=next(line.split() for line in reversed(text.splitlines()) if line.startswith(system+' '))
    return {'accuracy':float(values[1]),'yes_accuracy':float(values[2]),'no_accuracy':float(values[3]),
        'balanced_accuracy':float(values[4]),'missing':int(values[5]),'asked':int(values[6]),
        'question_count':sum(q['category']!='presence' for scene in scenes for q in scene['questions']),
        'precision':'official cmd_score output (3 decimals)'}

def evaluate(source, case, video, directory, judge, *, checkpoint=lambda:None):
    from evaluation.evaluator import source_identity
    source_identity(source)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    module=official(str(source));bank=Path(source)/'vlm_judge/vqa_questions_and_ontology_final.json'
    kind,identity=case.split('/',1)
    scenes=[s for s in module.load_scenes(bank) if s['id']==identity and s['kind']==kind]
    if len(scenes)!=1:return {'status':'not_applicable','reason':'No native question set for this case','metrics':{}}
    scene=scenes[0];questions=scene['questions'];rows=[];gate=None;state='ok';error=None
    try:
        checkpoint()
        if video is None or not Path(video).is_file():state='no_render'
        else:
            gate=module.layer1(video);save(directory/'layer1.json',gate)
            if gate.get('edge') is None:raise ValueError('Official layer1 could not decode the video')
            if gate['gated']:state='gated'
            else:
                frames=module.even_frames(video,16,768,directory/'frames');checkpoint()
                if not frames:raise ValueError('Official frame sampler returned no frames')
                # Gold answers, reference videos and category metadata never enter this request.
                text=judge(frames,module.vqa_prompt(questions,len(frames)))
                predictions=module.parse_vqa(text,len(questions))
                for question,pred in zip(questions,predictions):
                    rows.append({'scene':identity,'kind':kind,'id':question['id'],'category':question['category'],
                        'answer':question['answer'],'pred':pred,'correct':pred==question['answer'],'missing':pred=='Missing'})
                if any(p is None for p in predictions):state='invalid_answer'
    except InterruptedError:raise
    except Exception as exc:
        state='error';error=str(exc)
        rows=[{'scene':identity,'kind':kind,'id':q['id'],'category':q['category'],
            'answer':q['answer'],'pred':None,'correct':False,'missing':False} for q in questions]
    values=score(source,scenes,rows,directory,'task')
    result={'status':state,'judge':'kimi-k3','temperature':0,'model_variant':'Requested judge differs from the released Gemini judge',
        'question_bank_sha256':sha(bank),'judge_code_sha256':sha(Path(source)/'vlm_judge/judge.py'),
        'case':case,'gate':gate,'scores':values,'error':error,
        'scope':'Native questions and score rules on this task sample; judge tokens are evaluation overhead',
        'metrics':{'4dcodebench/vqa_'+name:{'status':'ok','measurement_status':state,
            'value':values[name],
            'reason':error or state,'source':'official_vqa_cmd_score'} for name in ('accuracy','balanced_accuracy')}}
    save(directory/'result.json',result)
    return result

def aggregate(source, cases, task_directories):
    """Pool native questions, not per-case balanced accuracies."""
    from tempfile import TemporaryDirectory
    module=official(str(source))
    wanted={tuple(case.split('/',1)) for case in cases}
    scenes=[s for s in module.load_scenes(module.DATA) if (s['kind'],s['id']) in wanted]
    if not scenes:return None
    rows=[]
    for directory in task_directories:
        selected=current_recheck(directory)
        path=(selected[1] if selected else Path(directory)/'evaluation/vqa')/'answers.jsonl'
        if path.is_file():rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    with TemporaryDirectory(prefix='worldrsi-vqa-score-') as directory:
        return score(source,scenes,rows,directory,'cohort')
