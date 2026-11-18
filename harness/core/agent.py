"""Stirrup's session/tool runtime, locally maintained from c092cee80ce09cd1dfb93aef053296a9ecd9dbb6.

Upstream run_tool/step/run/summarization algorithms are retained. Removed unused
provider/subagent execution, automatic system prompts, disk cache and resumption.
Task-owned sandboxes, prompts, skills and tool declarations live outside core.
"""
import inspect
import logging
from itertools import chain
from time import perf_counter
from typing import Any
import anyio
from pydantic import BaseModel, ValidationError
from .models import (AssistantMessage, ChatMessage, ImageContentBlock, SummaryMessage,
    SummaryTrigger, SystemMessage, Tool, ToolCall, ToolMessage, ToolResult, UserMessage,
    TurnWarningMessage, TokenUsage, joined_text, tool_call_blocks)
from .exceptions import CompactionRequiredError, ContextOverflowError
from harness.prompts import MESSAGE_SUMMARIZER, MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE

LOGGER = logging.getLogger(__name__)
logger = LOGGER

class TextFinal(Exception):
    """Provider returned a plain final after its request was archived and checked."""

def _tool_arguments_json(tool_call: ToolCall) -> str:
    """Return the call's arguments as JSON, treating an empty string as no arguments.

    Clients coerce a missing arguments field to "", so every site that validates a call
    must normalize it the same way.
    """
    return tool_call.arguments if tool_call.arguments.strip() else "{}"


def _handle_text_only_tool_responses(tool_messages: list[ToolMessage]) -> tuple[list[ToolMessage], list[UserMessage]]:
    """Extract image blocks from tool messages and convert them to user messages for text-only models."""
    user_messages: list[UserMessage] = []
    for tm in tool_messages:
        if isinstance(tm.content, list):
            for idx, block in enumerate(tm.content):
                if isinstance(block, ImageContentBlock):
                    label=(tm.content[idx-1] if idx and isinstance(tm.content[idx-1],str)
                           and tm.content[idx-1].startswith('EVIDENCE_GROUP ') else None)
                    user_messages.append(
                        UserMessage(content=[f"Here is the image for tool call {tm.tool_call_id}",
                                             *([label] if label else []), block]),
                    )
                    tm.content[idx] = f"Done! The User will provide the image for tool call {tm.tool_call_id}"
                elif isinstance(block, str):
                    continue
                else:
                    raise NotImplementedError(f"Unsupported content block: {type(block)}")

    return tool_messages, user_messages


def _get_total_token_usage(messages: list[list[ChatMessage]]) -> list[TokenUsage]:
    """
    Returns a list of TokenUsage objects aggregated from all AssistantMessage
    instances across the provided grouped message history.

    Args:
        messages: A list where each item is a list of ChatMessage objects representing a segment
                  or turn group of the conversation history.

    Returns:
        List of TokenUsage corresponding to each AssistantMessage in the flattened conversation history.
    """
    return [msg.token_usage for msg in chain.from_iterable(messages) if isinstance(msg, AssistantMessage)]


def _get_tool_durations(messages: list[list[ChatMessage]]) -> dict[str, list[float]]:
    """Collect tool execution durations grouped by tool name from message history."""
    durations: dict[str, list[float]] = {}
    for msg in chain.from_iterable(messages):
        if isinstance(msg, ToolMessage) and msg.name and msg.tool_duration is not None:
            durations.setdefault(msg.name, []).append(msg.tool_duration)
    return durations


def _get_turn_count(full_msg_history: list[list[ChatMessage]], messages: list[ChatMessage]) -> int:
    """Count accepted assistant turns still present in history."""
    all_messages = chain(chain.from_iterable(full_msg_history), messages)
    return sum(1 for msg in all_messages if isinstance(msg, AssistantMessage))


def _merge_run_metadata(run_metadata_by_turn: dict[str, dict[str, list[Any]]]) -> dict[str, list[Any]]:
    """Merge per-turn metadata into the public run_metadata shape."""
    run_metadata: dict[str, list[Any]] = {}
    for turn_metadata in run_metadata_by_turn.values():
        for name, metadata_list in turn_metadata.items():
            run_metadata.setdefault(name, []).extend(metadata_list)
    return run_metadata


def _get_model_speed_stats(messages: list[list[ChatMessage]], model_slug: str) -> dict[str, float | int | str]:
    """Compute speed stats for this agent's model from AssistantMessages.

    Returns a flat dict with model_slug, num_calls, output_tokens, duration, e2e_otps.
    Returns empty dict if no timed messages found.
    """
    num_calls = 0
    output_tokens = 0
    duration = 0.0
    for msg in chain.from_iterable(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        if msg.request_start_time is None or msg.request_end_time is None:
            continue
        msg_duration = msg.request_end_time - msg.request_start_time
        if msg_duration <= 0:
            continue
        num_calls += 1
        output_tokens += msg.token_usage.output
        duration += msg_duration
    if num_calls == 0:
        return {}
    return {
        "model_slug": model_slug,
        "num_calls": num_calls,
        "output_tokens": output_tokens,
        "duration": duration,
        "e2e_otps": output_tokens / duration if duration > 0 else 0.0,
    }


def _num_turns_remaining_msg(number_of_turns_remaining: int) -> TurnWarningMessage:
    """Create a user message warning the agent about remaining turns before max_turns is reached."""
    if number_of_turns_remaining == 1:
        return TurnWarningMessage(content="This is the last turn. Please finish the task by calling a finish tool.")
    return TurnWarningMessage(
        content=f"You have {number_of_turns_remaining} turns remaining to complete the task. Please continue. Remember you will need a separate turn to call a finish tool.",
    )


class Agent:
    def __init__(self, client, name, *, system_prompt, tools, finish_tool, max_turns=50,
                 context_summarization_cutoff=.75, logger,
                 run_sync_in_thread=False, text_only_tool_responses=True):
        if client.context_window_tokens <= 0:
            raise ValueError('context_window_tokens must be positive')
        self._client, self._name, self._system_prompt = client, name, system_prompt
        self._tools, self._finish_tools = list(tools), {finish_tool.name: finish_tool}
        self._active_tools = {tool.name: tool for tool in tools}
        if len(self._active_tools) != len(tools) or finish_tool.name in self._active_tools:
            raise ValueError('Tool names must be unique')
        self._active_tools.update(self._finish_tools)
        self._max_turns, self._logger = max_turns, logger
        self._context_window_tokens = client.context_window_tokens
        self._context_summarization_cutoff = context_summarization_cutoff
        self._turns_remaining_warning_threshold = 0
        self._run_sync_in_thread = run_sync_in_thread
        self._text_only_tool_responses = text_only_tool_responses
        self._block_successive_assistant_messages = True
        self._entered = False
        self.final_text = None

    def session(self):
        if self._entered:
            raise RuntimeError('Agent session is already active')
        return self

    async def __aenter__(self):
        self._entered = True
        self.final_text = None
        return self

    async def __aexit__(self, *args):
        self._entered = False

    async def run_tool(self, tool_call: ToolCall, run_metadata: dict[str, list[Any]]) -> ToolMessage:
        """Execute a single tool call with error handling for invalid JSON/arguments.

        Returns a ToolMessage containing either the tool output or an error description.
        Metadata from the tool result is stored in the provided run_metadata dict.
        """
        tool = self._active_tools.get(tool_call.name)
        result: ToolResult
        args_valid = True

        # Ensure tool is tracked in metadata dict (even if no metadata returned)
        if tool_call.name not in run_metadata:
            run_metadata[tool_call.name] = []

        tool_start_time = perf_counter()

        if tool:
            try:
                params = tool.parameters.model_validate_json(_tool_arguments_json(tool_call))

                if inspect.iscoroutinefunction(tool.executor):
                    result = await tool.executor(params)
                elif self._run_sync_in_thread:
                    result = await anyio.to_thread.run_sync(tool.executor, params)
                else:
                    result = tool.executor(params)

                # Store metadata if present
                if result.metadata is not None:
                    run_metadata[tool_call.name].append(result.metadata)
            except ValidationError:
                LOGGER.debug(
                    "LLMClient tried to use the tool %s but the tool arguments are not valid: %r",
                    tool_call.name,
                    tool_call.arguments,
                )
                result = ToolResult(content="Tool arguments are not valid", success=False)
                args_valid = False
        else:
            LOGGER.debug(f"LLMClient tried to use the tool {tool_call.name} which is not in the tools list")
            result = ToolResult(content=f"{tool_call.name} is not a valid tool", success=False)

        tool_end_time = perf_counter()

        return ToolMessage(
            content=result.content,
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            args_was_valid=args_valid,
            success=result.success,
            tool_start_time=tool_start_time,
            tool_end_time=tool_end_time,
        )


    @staticmethod
    def _task_context(messages):
        """Retain original instructions/source text; images compact with all history."""
        context=[]
        for message in messages:
            if isinstance(message,(AssistantMessage,SummaryMessage)):
                break
            if isinstance(message.content,list):
                text=[part for part in message.content if isinstance(part,str)]
                if text:context.append(message.model_copy(update={'content':text}))
            else:context.append(message)
        return context

    def _context_boundary_error(self):
        return ContextOverflowError(
            f'A single complete evidence/tool round exceeds the configured working context budget '
            f'({self._context_window_tokens}); request a smaller evidence result. Original receipts are retained.')

    async def step(
        self,
        messages: list[ChatMessage],
        run_metadata: dict[str, list[Any]],
        turn: int = 0,
        max_turns: int = 0,
    ) -> tuple[AssistantMessage, list[ToolMessage], BaseModel | None]:
        """Execute one agent step: generate assistant message and run any requested tool calls.

        Args:
            messages: Current conversation messages
            run_metadata: Metadata storage for tool results
            turn: Current turn number (1-indexed) for logging
            max_turns: Maximum turns for logging

        Returns the assistant message, tool execution results, and finish tool call (if present).
        There is one tool result per requested call, ordered as the calls were executed: every
        ordinary call first, then any finish call.

        """
        assistant_message = await self._client.generate(messages, self._active_tools)

        # Log assistant message immediately
        if turn > 0:
            self._logger.assistant_message(turn, max_turns, assistant_message)

        finish_params: BaseModel | None = None
        tool_messages: list[ToolMessage] = []
        tool_calls = tool_call_blocks(assistant_message.blocks)
        if tool_calls:
            # If multiple finish tools were called in one turn, refuse all of them without
            # executing — silently picking one would discard the model's other intent and
            # may swap in contradictory params. Force the model to retry with a single call.
            finish_call_names = [tc.name for tc in tool_calls if tc.name in self._finish_tools]
            reject_all_finish_calls = len(finish_call_names) > 1

            # Every ordinary call runs before any finish call, so a successful finish is always the
            # last thing that happens in the turn and no side effect can follow the run ending.
            # `sorted` is stable, so calls keep their relative order within each group. The returned
            # messages are in this execution order; providers match them to calls by tool_call_id.
            for tool_call in sorted(tool_calls, key=lambda tc: tc.name in self._finish_tools):
                if reject_all_finish_calls and tool_call.name in self._finish_tools:
                    tool_message = ToolMessage(
                        content=(
                            f"Cannot call finish tool '{tool_call.name}': multiple finish tools "
                            f"({sorted(set(finish_call_names))}) were called in the same turn. "
                            "Only one finish tool may be called per turn — retry with a single "
                            "finish tool call."
                        ),
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        success=False,
                    )
                else:
                    tool_message = await self.run_tool(tool_call, run_metadata)
                    if tool_message.success and tool_message.name in self._finish_tools:
                        finish_tool = self._finish_tools[tool_message.name]
                        finish_params = finish_tool.parameters.model_validate_json(_tool_arguments_json(tool_call))

                tool_messages.append(tool_message)
                self._logger.tool_result(tool_message)

        return assistant_message, tool_messages, finish_params


    async def summarize_messages(
        self,
        messages: list[ChatMessage],
        run_metadata_by_turn: dict[str, dict[str, list[Any]]],
        *,
        trigger: SummaryTrigger = "token_threshold",
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Summarize at complete tool-round boundaries, never discard a tool result.

        If the full summary request is rejected, first compact its older prefix,
        then summarize the remaining whole round with that prefix summary.
        """
        original_messages=messages
        current_messages=messages
        while True:
            try:
                summary=await self._client.generate(
                    [*current_messages,UserMessage(content=MESSAGE_SUMMARIZER)],{})
                summary_content=joined_text(summary.blocks)
                if summary_content is None:
                    raise RuntimeError('Summarizer returned no text; original history remains archived')
                bridge=MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary_content)
                replaced=[]
                for message in original_messages:
                    if isinstance(message,SummaryMessage):replaced.extend(message.replaced_ids)
                    elif isinstance(message,AssistantMessage):replaced.append(message.id)
                summary_message=SummaryMessage(content=bridge,replaced_ids=replaced,trigger=trigger)
                self._logger.context_summarization_complete(summary_content,bridge)
                return original_messages,[*self._task_context(original_messages),summary_message]
            except (ContextOverflowError,CompactionRequiredError):
                # Bisect at a complete round boundary, avoiding one failed paid
                # request for every historical turn. No tool call/result is dropped.
                boundaries=[i for i,message in enumerate(current_messages) if isinstance(message,AssistantMessage)]
                split=boundaries[len(boundaries)//2] if boundaries else None
                if split is None:
                    raise self._context_boundary_error() from None
                prefix=current_messages[:split]
                if (not any(isinstance(message,AssistantMessage) for message in prefix)
                        and any(isinstance(message,SummaryMessage) for message in prefix)):
                    raise self._context_boundary_error() from None
                _,compacted_prefix=await self.summarize_messages(prefix,run_metadata_by_turn,trigger=trigger)
                current_messages=[*compacted_prefix,*current_messages[split:]]


    async def run(
        self,
        init_msgs: str | list[ChatMessage],
        *,
        depth: int | None = None,
    ) -> tuple[BaseModel | None, list[list[ChatMessage]], dict[str, Any]]:
        """Execute a fresh invocation through Stirrup's native turn/tool loop."""
        if not self._entered:
            raise RuntimeError('Use async with agent.session() before run')
        msgs = [SystemMessage(content=self._system_prompt)]
        msgs.extend([UserMessage(content=init_msgs)] if isinstance(init_msgs, str) else init_msgs)
        full_msg_history = []
        step_callback = self._logger.on_step
        run_metadata_by_turn = {}

        # Cumulative stats for spinner
        total_tool_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        finish_params: BaseModel | None = None

        while _get_turn_count(full_msg_history, msgs) < self._max_turns:
            turn = _get_turn_count(full_msg_history, msgs)
            if self._max_turns - turn <= self._turns_remaining_warning_threshold and turn != 0:
                num_turns_remaining_msg = _num_turns_remaining_msg(self._max_turns - turn)
                msgs.append(num_turns_remaining_msg)
                self._logger.user_message(num_turns_remaining_msg)

            compacted = False
            while True:
                turn_metadata: dict[str, list[Any]] = {}
                try:
                    # Pass turn info to step() for real-time logging
                    assistant_message, tool_messages, finish_params = await self.step(
                        msgs,
                        turn_metadata,
                        turn=_get_turn_count(full_msg_history, msgs) + 1,
                        max_turns=self._max_turns,
                    )
                    break
                except TextFinal as final:
                    self.final_text = str(final)
                    full_msg_history.append(msgs)
                    return None, full_msg_history, _merge_run_metadata(run_metadata_by_turn)
                except (ContextOverflowError, CompactionRequiredError) as error:
                    # Compact complete rounds, including the newest evidence; tools must not rerun.
                    if compacted:
                        raise
                    if not any(isinstance(message,(AssistantMessage,SummaryMessage)) for message in msgs):
                        raise self._context_boundary_error() from error
                    trigger = error.reason if isinstance(error, CompactionRequiredError) else "token_overflow"
                    LOGGER.warning(f"Compacting the conversation after a rejected request: {error}")
                    messages_to_summarize, msgs = await self.summarize_messages(
                        msgs, run_metadata_by_turn, trigger=trigger
                    )
                    full_msg_history.append(messages_to_summarize)
                    compacted = True

            # Update cumulative stats
            run_metadata_by_turn[assistant_message.id] = turn_metadata
            accepted_turn = _get_turn_count(full_msg_history, msgs) + 1
            total_tool_calls += len(tool_messages)
            total_input_tokens += assistant_message.token_usage.input
            total_output_tokens += assistant_message.token_usage.output

            # Call progress callback after step completes
            if step_callback:
                step_callback(accepted_turn, total_tool_calls, total_input_tokens, total_output_tokens)

            user_messages: list[UserMessage] = []
            if self._text_only_tool_responses:
                tool_messages, user_messages = _handle_text_only_tool_responses(tool_messages)

            # Log user messages (e.g., image content extracted from tool responses)
            for user_msg in user_messages:
                self._logger.user_message(user_msg)

            msgs.extend([assistant_message, *tool_messages, *user_messages])

            if finish_params:
                break

            # Provider input includes cached tokens and image cost. This is one
            # request's actual input+output, never cumulative run expenditure.
            context_used=assistant_message.token_usage.total
            output_reserve=getattr(self._client,'max_output_tokens',None) or 8192
            summary_reserve=getattr(self._client,'summary_output_tokens',2048)
            threshold=max(1,min(self._context_window_tokens*self._context_summarization_cutoff,
                                self._context_window_tokens-output_reserve-summary_reserve))
            pct_context_used=context_used/self._context_window_tokens
            if context_used>=threshold and accepted_turn != self._max_turns:
                self._logger.context_summarization_start(pct_context_used, self._context_summarization_cutoff)
                messages_to_summarize, msgs = await self.summarize_messages(msgs, run_metadata_by_turn)
                full_msg_history.append(messages_to_summarize)

            # Avoid successive assistant messages (only if next turn won't show turns remaining)
            next_turn_will_show_warning = self._max_turns - accepted_turn <= self._turns_remaining_warning_threshold
            if (
                self._block_successive_assistant_messages
                and not tool_messages
                and not user_messages
                and not next_turn_will_show_warning
            ):
                continue_msg = UserMessage(content="Please continue the task")
                msgs.append(continue_msg)

        if finish_params is None:
            LOGGER.error(
                f"Maximum number of turns reached: {self._max_turns}. The agent was not able to finish the task. Consider increasing the max_turns parameter.",
            )

        full_msg_history.append(msgs)

        # Add agent's own token usage, tool durations, and model speed to run_metadata
        run_metadata: dict[str, Any] = _merge_run_metadata(run_metadata_by_turn)
        run_metadata["token_usage"] = _get_total_token_usage(full_msg_history)
        run_metadata["_tool_durations"] = _get_tool_durations(full_msg_history)
        run_metadata["_model_speed"] = _get_model_speed_stats(full_msg_history, self._client.model_slug)
        return finish_params, full_msg_history, run_metadata
