"""Request budgets and tool-outcome limits shared by invocations."""
from dataclasses import dataclass
import hashlib
import json
import re



class LoopFailure(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.reason = {'code':code, 'message':message, **details}


def output_text(response):
    return '\n'.join(c.get('text', '') for item in response.get('output', [])
                     if item.get('type') == 'message' for c in item.get('content', [])
                     if c.get('type') == 'output_text')


@dataclass(frozen=True)
class Budget:
    calls: int | None = None
    tokens: int | None = None
    meter: object = None

    def check(self, transport, *, after_response=False):
        if self.calls is None and self.tokens is None: return
        usage = self.meter() if self.meter else transport.usage() if self.tokens is not None else {}
        calls = usage.get('api_calls',len(transport.calls))
        if (self.tokens is not None and calls and not usage.get('complete')
                and usage.get('unmeasured_calls',calls)>usage.get('context_rejected_calls',0)):
            raise LoopFailure('usage_unavailable','Run failed: provider usage is incomplete; no further calls are allowed.',
                              usage=usage)
        inputs, outputs = usage.get('input_tokens',0), usage.get('output_tokens',0)
        consumed = inputs + outputs
        details = {'api_calls':calls,'input_tokens':inputs,'output_tokens':outputs,'total_tokens':consumed,
                   'call_limit':self.calls,'token_limit':self.tokens}
        if self.tokens is not None and (consumed > self.tokens or (not after_response and consumed >= self.tokens)):
            raise LoopFailure('token_limit_exceeded' if consumed > self.tokens else 'token_limit_reached',
                f'Run failed: input + output tokens = {consumed:,}; limit = {self.tokens:,}.',**details)
        if self.calls is not None and (calls > self.calls or (not after_response and calls >= self.calls)):
            raise LoopFailure('call_limit_exceeded' if calls > self.calls else 'call_limit_reached',
                f'Run failed: {calls} API calls used; limit = {self.calls}. No next call is allowed.',**details)
        return details


class Stagnation:
    """Count repeated causes/outcomes since the last accepted artifact change."""
    def __init__(self, limit=3, progress=lambda:None):
        self.limit, self.progress = limit, progress
        self.previous, self.seen = object(), {}

    def observe(self, event, arguments):
        if self.limit is None: return
        # Reopening evidence is not a failed attempt to change the world. Budgets
        # still bound these reads; actual tool errors remain eligible below.
        if event.status != 'failed' and (event.name in {'view_image','observe_scene','inspect_evidence',
                'probe_media','extract_frames','crop_image','estimate_depth','segment_image'}
                or (event.name == 'scene_state' and arguments.get('operation', 'query') == 'query')):
            return
        current = self.progress()
        if current != self.previous:
            self.previous, self.seen = current, {}
        result = event.result
        error = result.get('error')
        if event.status == 'failed':
            outcome = error.get('signature',error.get('message',error)) if isinstance(error,dict) else error
            signature = {'tool':event.name,'failure':outcome or
                         {k:result[k] for k in ('exit_code','stdout','stderr') if k in result}}
        else:
            signature = {'tool':event.name,'arguments':arguments,
                         'outcome':{k:v for k,v in result.items() if k not in
                             {'id','command','elapsed_s','stdout_path','stderr_path','stdout_bytes','stderr_bytes'}}}
        def stable(value):
            ignored = {'id','command','elapsed_s','archive_path','stdout_path','stderr_path','output_dir',
                       'manifest_path','timestamp','stdout_bytes','stderr_bytes','full_result_path'}
            if isinstance(value,dict): return {k:stable(v) for k,v in value.items() if k not in ignored}
            if isinstance(value,list): return [stable(v) for v in value]
            if isinstance(value,str):
                value = re.sub(r'(?<=/)[0-9a-f]{32}(?=/)', '<generated>', value)
                return re.sub(r'(world_state_|scene_code_)[0-9a-f]{12}(?=\.)',r'\1<generated>',value)
            return value
        if 'outcome' in signature:
            signature['outcome'] = stable(signature['outcome'])
        key = hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        occurrences = self.seen.setdefault(key,[])
        occurrences.append(event.id)
        if len(occurrences) >= self.limit:
            cause = error.get('message') if isinstance(error,dict) else error
            raise LoopFailure('stagnation',f'Run failed: the same {event.name} outcome repeated {len(occurrences)} times '
                'without a new accepted world, validated code or render.' +
                (f' Last error: {str(cause)[:800]}' if cause else ''),
                tool=event.name,tool_ids=occurrences,repeat_limit=self.limit,signature=signature)


@dataclass
class LoopResult:
    status: str
    text: str = ''
