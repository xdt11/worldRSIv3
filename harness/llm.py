"""The single Responses gateway implementation used directly by the local runtime.

No SDK, retries, provider router or task-state mutations. Full provider output
items are replayed verbatim, including opaque encrypted reasoning items.
"""
import json
import re
import hashlib
from time import perf_counter
from typing import Any
import httpx
from harness.core.agent import TextFinal
from harness.core.exceptions import ContextOverflowError, CompactionRequiredError
from harness.core.models import (AssistantMessage, AssistantBlock, ImageContentBlock,
    ToolMessage, SystemMessage, UserMessage, TextBlock, ToolCall, EncryptedReasoningBlock,
    ReasoningBlock, TokenUsage)
from harness.core.budget import LoopFailure, Budget, output_text
from harness.evidence.archive import now, uid, save
from harness.prompts import MESSAGE_SUMMARIZER

STIRRUP_REVISION = 'c092cee80ce09cd1dfb93aef053296a9ecd9dbb6'

def _get_attr(obj: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    """Get attribute from object or dict, with fallback default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_response_output(
    output: list[Any],
    *,
    allow_reference_reasoning: bool = True,
) -> list[AssistantBlock]:
    """Parse response output items into ordered assistant blocks.

    One exhaustive pass in item order: each ``message`` item becomes its own
    TextBlock (refusal content surfaces as answer text), each ``function_call``
    a ToolCall block, and each ``reasoning`` item an EncryptedReasoningBlock or
    readable ReasoningBlock. Stored continuation state lives on the assistant
    message as ``provider_response_id``; reasoning item IDs are not duplicated
    into blocks. Unknown item and content types raise until their semantics and
    passback behavior are explicitly implemented.
    """
    blocks: list[AssistantBlock] = []

    for item in output:
        item_type = _get_attr(item, "type")

        match item_type:
            case "message":
                text_parts: list[str] = []
                for content_item in _get_attr(item, "content", []):
                    content_type = _get_attr(content_item, "type")
                    if content_type == "refusal":
                        # A refusal is a normal API response, not malformed output: surface
                        # its text as the assistant's answer rather than crashing the turn.
                        refusal = _get_attr(content_item, "refusal")
                        if not isinstance(refusal, str):
                            raise ValueError(f"OpenAI Responses refusal is not a string: {refusal!r}")
                        text_parts.append(refusal)
                        continue
                    if content_type != "output_text":
                        raise NotImplementedError(
                            f"Unsupported OpenAI Responses message content type: {content_type!r}"
                        )
                    text = _get_attr(content_item, "text")
                    if not isinstance(text, str):
                        raise ValueError(f"OpenAI Responses output_text is not a string: {text!r}")
                    text_parts.append(text)
                if text_parts:
                    blocks.append(TextBlock(text="".join(text_parts)))

            case "function_call":
                blocks.append(
                    ToolCall.from_provider(
                        provider_id=_get_attr(item, "call_id"),
                        name=_get_attr(item, "name"),
                        arguments=_get_attr(item, "arguments", ""),
                    )
                )

            case "reasoning":
                content = _get_attr(item, "content")
                content_parts: list[str] = []
                if isinstance(content, list):
                    for part in content:
                        text = _get_attr(part, "text")
                        if not isinstance(text, str):
                            raise ValueError(f"OpenAI Responses reasoning content entry has no text: {part!r}")
                        if text:
                            content_parts.append(text)
                elif content:
                    content_parts = [str(content)]

                # summary can be a list of Summary objects with .text attribute
                summary = _get_attr(item, "summary")
                if isinstance(summary, list):
                    summary_parts = []
                    for s in summary:
                        text = _get_attr(s, "text")
                        if not isinstance(text, str):
                            raise ValueError(f"OpenAI Responses reasoning summary entry has no text: {s!r}")
                        # Empty parts carry no data; dropping them keeps re-emission faithful.
                        if text:
                            summary_parts.append(text)
                elif summary:
                    summary_parts = [str(summary)]
                else:
                    thinking = _get_attr(item, "thinking")
                    summary_parts = [thinking] if thinking else []

                item_id = _get_attr(item, "id")
                encrypted_content = _get_attr(item, "encrypted_content")
                if encrypted_content and not item_id:
                    # The id is the passback handle; without it the encrypted payload
                    # cannot be re-emitted and ZDR reasoning state would silently vanish.
                    raise ValueError(f"OpenAI Responses reasoning item has encrypted_content but no id: {item!r}")
                if item_id and encrypted_content:
                    # Stateless (store=false / ZDR) item: carried whole for verbatim re-emission.
                    blocks.append(
                        EncryptedReasoningBlock(
                            id=item_id,
                            encrypted_content=encrypted_content,
                            content="".join(content_parts) or None,
                            summary=summary_parts,
                        )
                    )
                elif item_id:
                    if not allow_reference_reasoning:
                        raise NotImplementedError(
                            "A stateless OpenAI Responses call returned reference-only reasoning; "
                            "enable encrypted_reasoning so it can be passed back"
                        )
                    readable_parts = [*content_parts, *summary_parts]
                    if readable_parts:
                        blocks.append(ReasoningBlock(content="\n".join(readable_parts)))
                elif content_parts or summary_parts:
                    blocks.append(ReasoningBlock(content="\n".join([*content_parts, *summary_parts])))

            case _:
                raise NotImplementedError(f"Unsupported OpenAI Responses output item type: {item_type!r}")

    return blocks


class Responses:
    def __init__(self, archive, config, api_key, *, budget=None, stopped=lambda:False,
                 before_call=None, on_call=None):
        self.archive, self.config, self.api_key = archive, config, api_key
        self.calls = archive.calls
        self.budget = budget or Budget()
        self.stopped, self.before_call, self.on_call = stopped, before_call, on_call
        self.policy = config.get('runtime', {})
        self.model_slug = (config.get('critic_model') or config.get('model')) if archive.role != 'reconstruction' else config.get('model')
        # This is an operator-selected working budget, not inferred model capacity.
        self.context_window_tokens = self.policy.get('context_window_tokens') or 65536
        self.max_output_tokens = config.get('max_output_tokens')
        self.summary_output_tokens = min(self.max_output_tokens or 8192,2048)
        if self.context_window_tokens<=(self.max_output_tokens or 8192)+self.summary_output_tokens:
            raise ValueError('Working context budget must exceed max_output_tokens plus summary output reserve')
        self.original_outputs, self.pending, self.archived = {}, {}, set()
        self.repairs, self.repair_prompt = 0, None
        self.schemas = {}

    def usage(self):
        return self.archive.usage()

    def checkpoint(self):
        if self.stopped():
            raise InterruptedError('Invocation stop requested')


    def call(self, instructions, items, *, tools=None, request_kind='generation', temperature=None, tool_choice=None):
        """One actual HTTP request. Every attempted call is archived and budgeted."""
        self.checkpoint()
        if self.before_call:
            self.before_call()
        remaining = self.budget.check(self)
        if request_kind != 'summary' and remaining is not None:
            notice = {'kind':'invocation_budget',
                'calls_remaining':None if self.budget.calls is None else self.budget.calls - remaining['api_calls'],
                'tokens_remaining':None if self.budget.tokens is None else self.budget.tokens - remaining['total_tokens'],
                'scope':'Includes this request and cached input tokens.'}
            items = [*items, {'role':'user','content':json.dumps(notice)}]
        call_id = uid('call')
        folder = self.archive.directory / 'calls' / call_id
        folder.mkdir(parents=True)
        body = {'model': self.model_slug, 'instructions': instructions, 'input': items,
                'store': False, 'include': ['reasoning.encrypted_content']}
        if temperature is not None:body['temperature']=temperature
        if self.max_output_tokens is not None:
            body['max_output_tokens'] = self.summary_output_tokens if request_kind=='summary' else self.max_output_tokens
        if tools:
            body.update(tools=tools, parallel_tool_calls=self.policy.get('parallel_tool_calls', True))
            if tool_choice is not None: body['tool_choice']=tool_choice
        request = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        (folder / 'request.json').write_bytes(request)
        stage = 'agent' if self.archive.role == 'reconstruction' else self.archive.role
        meta = {'id': call_id, 'invocation_id': self.archive.id, 'task_id': self.archive.task_id,
                'run_id': self.archive.run_id, 'role': self.archive.role, 'stage': stage,
                'model': self.model_slug, 'status': 'running', 'created_at': now(), 'usage': None,
                'instructions_sha256': hashlib.sha256(instructions.encode()).hexdigest(),
                'request_path': self.archive.relative(folder / 'request.json'),
                'response_path': self.archive.relative(folder / 'response.json'),
                'meta_path': self.archive.relative(folder / 'meta.json'),
                'runtime': {'framework':'stirrup', 'revision':STIRRUP_REVISION, 'source':'harness/core',
                    'request_kind':request_kind, 'sdk_retries':0, 'resume':False,
                    'working_context_budget_tokens':self.context_window_tokens},
                'context_image_count':sum(part.get('type') == 'input_image' for item in items
                    if isinstance(item.get('content'),list) for part in item['content'])}
        self.calls.append(meta)
        save(folder / 'meta.json', meta)
        self.archive.update(status='running')
        started = perf_counter()
        try:
            with httpx.Client(timeout=self.config.get('request_timeout_s',240),
                              follow_redirects=False, trust_env=False) as client:
                response = client.post(self.config['base_url'].rstrip('/') + '/responses', content=request,
                    headers={'Authorization': 'Bearer ' + self.api_key, 'Content-Type':'application/json'})
            (folder / 'response.raw').write_bytes(response.content)
            try:
                parsed = response.json()
            except ValueError:
                parsed = {'non_json_body': response.text}
            save(folder / 'response.json', parsed)
            if not isinstance(parsed, dict):
                raise RuntimeError('Provider response must be an object: ' + call_id)
            meta.update(http_status=response.status_code, usage=parsed.get('usage'),
                llmgate_usage=parsed.get('llmgate_usage'),
                charged_amount=(parsed.get('llmgate_usage') or {}).get('charged_amount',parsed.get('charged_amount')),
                currency=parsed.get('currency'), response_id=parsed.get('id'), provider_status=parsed.get('status'),
                headers={key:val for key,val in response.headers.items()
                    if key.lower() in {'content-type','x-request-id','request-id'}})
            if response.status_code >= 400:
                error=parsed.get('error') or {}
                code=str(error.get('code','')).lower() if isinstance(error,dict) else ''
                message=str(error.get('message','')).lower() if isinstance(error,dict) else ''
                if response.status_code==400 and (code in {'context_length_exceeded','max_context_length_exceeded','prompt_too_long'}
                        or 'maximum context length' in message or 'context window exceeded' in message):
                    meta['context_rejected_before_generation']=True
                    raise ContextOverflowError(f'Provider rejected context; inspect archived response {call_id}')
                if response.status_code==400 and code=='too_many_images':
                    meta['context_rejected_before_generation']=True
                    raise CompactionRequiredError(f'Provider rejected image capacity; inspect archived response {call_id}',reason='image_count')
                if response.status_code==413:
                    meta['context_rejected_before_generation']=True
                    raise CompactionRequiredError(f'Provider rejected request size; inspect archived response {call_id}',reason='request_size')
                raise RuntimeError(f'Provider HTTP {response.status_code}; inspect archived response {call_id}')
            if parsed.get('status') != 'completed' or not isinstance(parsed.get('output'), list):
                raise RuntimeError('Provider returned incomplete or invalid Responses payload: ' + call_id)
            meta['status'] = 'completed'
            return parsed
        except Exception as exc:
            meta.update(status='failed', error=str(exc).replace(self.api_key,'[REDACTED]') if self.api_key else str(exc))
            raise
        finally:
            meta.update(elapsed_s=perf_counter()-started, finished_at=now())
            save(folder / 'meta.json', meta)
            self.archive.update(usage=self.usage(), call_count=len(self.calls))
            self.budget.check(self, after_response=True)

    async def generate(self, messages, tools):
        instructions, items = '', []
        for message in messages:
            if isinstance(message, SystemMessage):
                instructions = str(message.content)
            elif isinstance(message, AssistantMessage):
                items.extend(self.original_outputs[message.id])
            elif isinstance(message, ToolMessage):
                content = message.content if isinstance(message.content,str) else '\n'.join(message.content)
                items.append({'type':'function_call_output','call_id':message.tool_call_id,'output':content})
            else:
                content = message.content
                parts = [{'type':'input_text','text':content}] if isinstance(content,str) else [
                    {'type':'input_text','text':p} if isinstance(p,str) else
                    {'type':'input_image','image_url':p.to_base64_url()} for p in content]
                items.append({'role':'user','content':parts})
        if self.repair_prompt:
            items.append({'role':'user','content':self.repair_prompt})
            self.repair_prompt = None
        summarized = isinstance(messages[-1],UserMessage) and isinstance(messages[-1].content,str) and messages[-1].content.startswith(MESSAGE_SUMMARIZER)
        declarations = [{'type':'function', 'name':t.name, 'description':t.description,
                         'parameters':t.parameters.model_json_schema(), 'strict':False} for t in tools.values()]
        if summarized:
            # Keep the same tool-definition prefix during compaction. Tools
            # remain disabled for the summary; changing the schema to [] would
            # invalidate the otherwise reusable conversation prefix.
            declarations=list(self.schemas.values())
        else:
            self.schemas={entry['name']:entry for entry in declarations}
        started = perf_counter()
        response = self.call(instructions, items, tools=declarations,
            request_kind='summary' if summarized else 'generation',
            tool_choice='none' if summarized else None)
        if self.on_call:
            self.on_call()
        self.checkpoint()
        calls = [x for x in response['output'] if x.get('type') == 'function_call']
        if summarized and calls:
            raise LoopFailure('tool_protocol_error','Context summary returned a tool call despite tool_choice=none')
        self.pending = {x['call_id']:x for x in calls}
        self.archived.clear()
        if len(self.pending) != len(calls):
            raise LoopFailure('tool_protocol_error','Duplicate function call IDs in Responses output')
        if not calls and not summarized:
            text = output_text(response)
            if re.match(r'^\s*<(?:function=|tool_call>)',text):
                if self.repairs >= self.policy.get('protocol_repair_attempts',1):
                    raise LoopFailure('tool_protocol_error','Provider returned tool text instead of native function_call')
                self.repairs += 1
                self.repair_prompt = 'Use native Responses function_call items to execute tools, or provide a final answer.'
            else:
                raise TextFinal(text)
        usage = response.get('usage') or {}
        result = AssistantMessage(blocks=_parse_response_output(response['output'],allow_reference_reasoning=False),
            token_usage=TokenUsage(input=usage.get('input_tokens',0),answer=usage.get('output_tokens',0)),
            request_start_time=started,request_end_time=perf_counter())
        self.original_outputs[result.id] = response['output']
        return result
