"""Role preparation: bind domain tools to native Stirrup types; no execution loop."""
import base64
import copy
import json
from dataclasses import dataclass
from jsonschema import Draft202012Validator, ValidationError
from pydantic import RootModel, model_validator
from harness.core.agent import Agent
from harness.core.models import Tool, ToolResult, UserMessage, ImageContentBlock
from harness.core.budget import Budget, Stagnation, LoopFailure, LoopResult
from harness.evidence.archive import InvocationArchive, now, uid, save
from harness.evidence.usage import ToolEvent
from harness.evidence.context import tool_feedback, execution_failure
from harness.llm import Responses


def parameters(schema):
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    class Parameters(RootModel[dict]):
        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            return copy.deepcopy(schema)
        @model_validator(mode='before')
        @classmethod
        def validate_schema(cls, value):
            try:
                validator.validate(value)
            except ValidationError as exc:
                raise ValueError(f'{exc.json_path}: {exc.message}') from exc
            return value
    return Parameters


def content(parts):
    if isinstance(parts,str):
        return parts
    result = []
    for part in parts:
        if part['type'] in ('input_text','output_text'):
            result.append(part['text'])
        elif part['type'] == 'input_image':
            header,payload = part['image_url'].split(',',1)
            if not header.startswith('data:image/') or ';base64' not in header:
                raise ValueError('Only archived inline image data is accepted')
            result.append(ImageContentBlock(data=base64.b64decode(payload,validate=True)))
        else:
            raise ValueError('Unsupported input content: '+part['type'])
    return result


@dataclass
class PreparedInvocation:
    agent: Agent
    messages: list
    archive: InvocationArchive
    finish_schema: dict | None = None
    transform: object = None
    finish_validate: object = None

    def parse_result(self,result):
        if result is None:
            text = self.agent.final_text
            if text is None:
                raise LoopFailure('turn_limit_reached','Invocation reached its turn limit without completion')
            value = json.loads(text) if self.finish_schema else None
        else:
            value = result.model_dump(mode='json')
            text = json.dumps(value,ensure_ascii=False) if self.finish_schema else value['reason']
        if self.finish_schema:
            Draft202012Validator(self.finish_schema).validate(value)
            if result is None and self.finish_validate is not None:
                self.finish_validate(value)
            outcome = self.transform(value) if self.transform else value
        else:
            if result is None and self.finish_validate is not None:
                self.finish_validate({'reason': text, 'paths': []})
            outcome = LoopResult('completed',text)
        save(self.archive.directory / 'result.json', value if self.finish_schema else {'text':text})
        self.archive.update(status='completed',usage=self.archive.usage(),finished_at=now())
        return outcome


class Audit:
    depth = 0
    on_step = None
    def __init__(self,client,schemas,*,stagnation,emit,archive_result):
        self.client,self.schemas,self.stagnation=client,schemas,stagnation
        self.emit,self.archive_result=emit,archive_result
    def __getattr__(self,name):
        return lambda *args,**kwargs:None
    def record(self,call,result,args,success,event=None):
        client=self.client
        event=event or ToolEvent(uid('tool'),call['call_id'],call['name'],call['arguments'],
            now(), 'agent' if client.archive.role=='reconstruction' else client.archive.role,
            client.calls[-1]['id'] if client.calls else None)
        event.result,event.status,event.finished_at=result,'completed' if success else 'failed',now()
        limit=args.get('max_output_chars',client.policy.get('tool_output_chars',4096)) if call['name']=='bash' and isinstance(args,dict) else client.policy.get('tool_output_chars',4096)
        event.feedback=tool_feedback(result,limit)
        target=client.archive.directory/'tools'/f'{event.id}.json'
        save(target,event.result)
        if self.archive_result:
            event.feedback['full_result_path']=self.archive_result(event)
        client.archive.event(event.to_dict())
        if self.emit:
            self.emit(event.to_dict())
        client.archived.add(call['call_id'])
        self.stagnation.observe(event,args)
        return event.feedback
    def tool_result(self,message):
        if message.tool_call_id in self.client.archived:
            return
        call=self.client.pending[message.tool_call_id]
        args={}
        result={'error':{'code':'invalid_tool_call','message':str(message.content)}}
        if call['name'] in self.schemas:
            try:
                args=json.loads(call['arguments'])
                errors=list(Draft202012Validator(self.schemas[call['name']]['parameters']).iter_errors(args))
                fields=[{'path':e.json_path,'message':e.message} for e in errors[:10]]
            except (ValueError,TypeError) as exc:
                fields=[{'path':'$','message':str(exc)}]
            result={'error':{'code':'invalid_tool_arguments','message':'Tool arguments failed validation','fields':fields}}
        message.content=json.dumps(self.record(call,result,args,False),ensure_ascii=False)


def prepare_agent(*, task_directory, task_id, run_id, role, config, api_key,
                  prompt,items,tools,execute,budget=None,stagnation=None,stopped=lambda:False,
                  before_call=None,emit=None,archive_result=None,finish_schema=None,skills_dir=None,
                  finish_validate=None,finish_name='finish'):
    archive=InvocationArchive(task_directory,role=role,task_id=task_id,
                              run_id=None if role=='task_critic' else run_id)
    budget=budget or Budget()
    client=Responses(archive,config,api_key,budget=budget,stopped=stopped,before_call=before_call)
    declarations={tool['name']:tool for tool in tools}
    finish_schema_default={'type':'object','properties':{'reason':{'type':'string'},
        'paths':{'type':'array','items':{'type':'string'}}},'required':['reason','paths'],'additionalProperties':False}
    finish={'type':'function','name':finish_name,'description':(finish_schema or {}).get('description','Complete this invocation with its required structured result.'),
            'parameters':finish_schema or finish_schema_default}
    declarations[finish['name']]=finish
    audit=Audit(client,declarations,stagnation=stagnation or Stagnation(),emit=emit,archive_result=archive_result)
    def executor(name,is_finish=False):
        async def run(params):
            client.checkpoint()
            call=next(v for k,v in client.pending.items() if k not in client.archived and v['name']==name)
            event=ToolEvent(uid('tool'),call['call_id'],name,call['arguments'],now(),
                'agent' if role=='reconstruction' else role,client.calls[-1]['id'] if client.calls else None)
            args=params.model_dump(mode='json',exclude_unset=True)
            try:
                if is_finish and finish_validate is not None:
                    finish_validate(args)
                result,visual=(args,None) if is_finish else execute(name,args)
                success=result.get('exit_code',0)==0 and not result.get('error')
                if not success and not result.get('error'):
                    result={**result,'error':execution_failure(result)}
            except InterruptedError:
                raise
            except Exception as exc:
                result,visual,success={'error':getattr(exc,'diagnostic',str(exc))},None,False
            feedback=audit.record(call,result,args,success,event)
            payload=json.dumps(feedback,ensure_ascii=False)
            if visual:
                payload=[payload,*content(visual if isinstance(visual,list) else [visual])]
            return ToolResult(content=payload,success=success)
        return run
    native_tools=[Tool(name=t['name'],description=t['description'],parameters=parameters(t['parameters']),
        executor=executor(t['name'])) for t in tools]
    finish_tool=Tool(name=finish_name,description=finish['description'],parameters=parameters(finish['parameters']),
        executor=executor(finish_name,True))
    if skills_dir:
        from harness.skills.loader import index
        skill_index=index(skills_dir)
        if skill_index:
            prompt+='\n\nFROZEN SKILLS\n'+skill_index
    save(archive.directory/'context.json',{'instructions':prompt,'input':items,
        'tool_schemas':list(declarations.values())})
    policy=config.get('runtime',{})
    agent=Agent(client=client,name=role,system_prompt=prompt,tools=native_tools,finish_tool=finish_tool,
        max_turns=(budget.calls+1) if budget.calls is not None else 2**31-1,
        context_summarization_cutoff=policy.get('context_summarization_cutoff',.75),logger=audit)
    return PreparedInvocation(agent,[UserMessage(content=content(item['content'])) for item in items],archive,finish_schema,
                              finish_validate=finish_validate)
