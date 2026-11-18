"""Stirrup runtime types, source revision c092cee80ce09cd1dfb93aef053296a9ecd9dbb6.
Pruned: media decoders, provider lifecycle and subagent metadata.
"""
import base64
import mimetypes
import warnings
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO
from math import isinf, isnan, sqrt
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self, overload, runtime_checkable
from uuid import uuid4

import filetype
from PIL import Image
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PlainSerializer,
    PlainValidator,
    model_validator,
)

RESOLUTION_1MP = 1_000_000
RESOLUTION_480P = 854 * 480
from .exceptions import CompactionReason

__all__ = [
    "AnyReasoningBlock",
    "AssistantBlock",
    "AssistantMessage",
    "BinaryContentBlock",
    "ChatMessage",
    "Content",
    "ContentBlock",
    "EmptyParams",
    "EncryptedReasoningBlock",
    "ImageContentBlock",
    "LLMClient",
    "OpaqueBlock",
    "Reasoning",
    "ReasoningBlock",
    "ReasoningRefBlock",
    "RedactedReasoningBlock",
    "SignedReasoningBlock",
    "SummaryMessage",
    "SummaryTrigger",
    "SystemMessage",
    "TextBlock",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "ToolResult",
    "ToolUseCountMetadata",
    "TurnWarningMessage",
    "UserMessage",
    "final_text",
    "joined_text",
    "reasoning_blocks",
    "tool_call_blocks",
]


def _bytes_to_b64(v: bytes) -> str:
    return base64.b64encode(v).decode("ascii")


def _b64_to_bytes(v: bytes | str) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return base64.b64decode(v.encode("ascii"))
    raise TypeError("Invalid bytes value")


Base64Bytes = Annotated[
    bytes,
    PlainValidator(_b64_to_bytes),
    PlainSerializer(_bytes_to_b64, when_used="json"),
]


def downscale_image(w: int, h: int, max_pixels: int | None = 1_000_000) -> tuple[int, int]:
    """Downscale image dimensions to fit within max pixel count while maintaining aspect ratio.

    Returns even dimensions with minimum 2x2 size.
    """
    s = 1.0 if max_pixels is None or w * h <= max_pixels else sqrt(max_pixels / (w * h))
    nw, nh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    return max(nw, 2), max(nh, 2)


# Content
class BinaryContentBlock(BaseModel, ABC, frozen=True):
    """Base class for binary content (images, video, audio) with MIME type validation."""

    data: Base64Bytes
    allowed_mime_types: ClassVar[set[str]]

    @property
    def mime_type(self) -> str:
        """MIME type for data based on headers."""
        match: filetype.Type = filetype.guess(self.data)
        if match is None:
            raise ValueError(f"Unsupported file type {self.data!r}")
        return match.mime

    @property
    def extension(self) -> str:
        """File extension for the content (e.g., 'png', 'mp4', 'mp3') without leading dot."""
        _extension = mimetypes.guess_extension(self.mime_type)
        if _extension is None:
            raise ValueError(f"Unsupported mime_type {self.mime_type!r}")
        return _extension[1:]

    @model_validator(mode="after")
    def _check_mime(self) -> Self:
        """Validate MIME type against allowed list and verify content is readable."""
        if self.allowed_mime_types and self.mime_type not in self.allowed_mime_types:
            raise ValueError(f"Unsupported mime_type {self.mime_type!r}; allowed: {self.allowed_mime_types}")
        self._probe()  # light corruption check; no heavy work
        return self

    @abstractmethod
    def _probe(self) -> None:
        """Verify content can be opened and read; subclasses implement format-specific checks."""


class ImageContentBlock(BinaryContentBlock, frozen=True):
    """Image content supporting PNG, JPEG, WebP, PSD formats with automatic downscaling."""

    kind: Literal["image_content_block"] = "image_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "image/jpeg",  # JPEG
        "image/png",  # PNG
        "image/webp",  # WebP
        "image/gif",  # GIF
        "image/bmp",  # BMP
        "image/tiff",  # TIFF
        "image/vnd.adobe.photoshop",  # PSD
    }

    def _probe(self) -> None:
        """Verify image data is valid by attempting to open and verify it with PIL."""
        with Image.open(BytesIO(self.data)) as im:
            im.verify()

    def to_base64_url(self, max_pixels: int | None = RESOLUTION_1MP) -> str:
        """Convert image to base64 data URL, optionally resizing to max pixel count."""
        img: Image.Image = Image.open(BytesIO(self.data))
        if max_pixels is not None and img.width * img.height > max_pixels:
            tw, th = downscale_image(img.width, img.height, max_pixels)
            img.thumbnail((tw, th), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{b64encode(buf.getvalue()).decode()}"


type ContentBlock = ImageContentBlock | str
"""Union of all content block types (image, video, audio, or text)."""

type Content = list[ContentBlock] | str
"""Message content: either a plain string or list of mixed content blocks."""


# Metadata Protocol and Aggregation
class TokenUsage(BaseModel):
    """Token counts for LLM usage.

    Token terminology: output = reasoning + answer.
    """

    input: int = 0
    answer: int = 0
    reasoning: int = 0

    @model_validator(mode="before")
    @classmethod
    def _deprecate_output_field(cls, data: object) -> object:
        if isinstance(data, dict) and "output" in data:
            warnings.warn(
                "The 'output' field is deprecated. Use 'answer' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            values = dict(data)
            values.setdefault("answer", values.pop("output"))
            return values
        return data

    @property
    def output(self) -> int:
        """Total output tokens (reasoning + answer)."""
        return self.answer + self.reasoning

    @property
    def total(self) -> int:
        """Total token count across input, answer, and reasoning."""
        return self.input + self.answer + self.reasoning

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together, summing each field independently."""
        return TokenUsage(
            input=self.input + other.input,
            answer=self.answer + other.answer,
            reasoning=self.reasoning + other.reasoning,
        )


class ToolUseCountMetadata(BaseModel):
    """Generic metadata tracking tool usage count.

    Implements Addable protocol for aggregation. Use this for tools that only need
    to track how many times they were called.

    Subclasses can override __add__ with their own type thanks to Self typing.
    """

    num_uses: int = 1

    def __add__(self, other: Self) -> Self:
        return self.__class__(num_uses=self.num_uses + other.num_uses)


class ToolResult[M](BaseModel):
    """Result from a tool executor with optional metadata.

    Generic over metadata type M. M should implement Addable protocol for aggregation support,
    but this is not enforced at the class level due to Pydantic schema generation limitations.

    Attributes:
        content: The result content (string, list of content blocks, or images)
        success: Whether the tool call was successful. For finish tools, controls if agent terminates.
        metadata: Optional metadata (e.g., usage stats) that implements Addable for aggregation
    """

    content: Content
    success: bool = True
    metadata: M | None = None


class EmptyParams(BaseModel):
    """Empty parameter model for tools that don't require parameters."""


class Tool[P: BaseModel, M](BaseModel):
    """Tool definition with name, description, parameter schema, and executor function.

    Generic over:
        P: Parameter model type (Pydantic BaseModel subclass, or EmptyParams for parameterless tools)
        M: Metadata type (should implement Addable for aggregation; use None for tools without metadata)

    Tools are simple, stateless callables. For tools requiring lifecycle management
    (setup/teardown, resource pooling), use a ToolProvider instead.

    Example with parameters:
        ```python
        class CalcParams(BaseModel):
            expression: str

        calc_tool = Tool[CalcParams, None](
            name="calc",
            description="Evaluate math",
            parameters=CalcParams,
            executor=lambda p: ToolResult(content=str(eval(p.expression))),
        )
        ```

    Example without parameters (uses EmptyParams by default):
        ```python
        time_tool = Tool[EmptyParams, None](
            name="time",
            description="Get current time",
            executor=lambda _: ToolResult(content=datetime.now().isoformat()),
        )
        ```
    """

    name: str
    description: str
    parameters: type[P] = EmptyParams  # ty: ignore[invalid-assignment]
    executor: Callable[[P], ToolResult[M] | Awaitable[ToolResult[M]]]


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the interface for LLM client implementations.

    Any LLM client must implement this protocol to work with the Agent class.
    Provides text generation with tool support and model capability inspection.
    """

    @abstractmethod
    async def generate(self, messages: list["ChatMessage"], tools: dict[str, Tool]) -> "AssistantMessage": ...

    @property
    def model_slug(self) -> str: ...

    @property
    def context_window_tokens(self) -> int: ...


class ToolCall(BaseModel, frozen=True):
    """Represents a tool invocation request from the LLM.

    Also a member of the ``AssistantBlock`` union: the ``kind`` discriminator is
    defaulted, so legacy payloads without the key still validate anywhere ToolCall
    is used as a plain input, and new dumps always carry it.

    Attributes:
        name: Name of the tool to invoke
        arguments: JSON string containing tool parameters
        tool_call_id: Unique identifier for tracking this tool call and its result
    """

    kind: Literal["tool_call"] = "tool_call"
    signature: str | None = None
    """Opaque passback state attached to this exact block, e.g. a Google thought signature."""
    name: str
    arguments: str
    tool_call_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    has_provider_tool_call_id: bool = True
    """Whether ``tool_call_id`` was present on the provider's original block.

    A client may synthesize ``tool_call_id`` for internal call/result matching
    while retaining that it must be omitted from provider-attached passback.
    """

    @model_validator(mode="before")
    @classmethod
    def _normalize_internal_id(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("tool_call_id") in (None, ""):
            normalized["tool_call_id"] = uuid4().hex
            normalized["has_provider_tool_call_id"] = False
        return normalized

    @classmethod
    def from_provider(
        cls,
        *,
        provider_id: str | None,
        name: str,
        arguments: str,
        signature: str | None = None,
    ) -> Self:
        """Capture one provider call with a stable internal correlation ID."""
        native_id = provider_id or None
        return cls(
            tool_call_id=native_id or uuid4().hex,
            has_provider_tool_call_id=native_id is not None,
            name=name,
            arguments=arguments,
            signature=signature,
        )


class SystemMessage(BaseModel):
    """System-level instructions and context for the LLM."""

    role: Literal["system"] = "system"
    content: Content


class UserMessage(BaseModel):
    """User input message to the LLM."""

    role: Literal["user"] = "user"
    kind: Literal["user"] = "user"
    content: Content


type SummaryTrigger = Literal["token_threshold", "token_overflow"] | CompactionReason


class SummaryMessage(UserMessage):
    """Summary message bridging summarized-away conversation context.

    Attributes:
        replaced_ids: Ids of the AssistantMessages this summary replaced, so
            consumers of dumped histories can reconstruct lineage offline.
        trigger: Why the agent summarized: ``"token_threshold"`` when context usage
            crossed the summarization cutoff, ``"token_overflow"`` after a
            ``ContextOverflowError``, otherwise the ``reason`` of a ``CompactionRequiredError``.
    """

    kind: Literal["summary"] = "summary"
    replaced_ids: list[str] = Field(default_factory=list)
    trigger: SummaryTrigger = "token_threshold"


class TurnWarningMessage(UserMessage):
    """Warning message injected when the agent is close to max_turns."""

    kind: Literal["turn_warning"] = "turn_warning"


class Reasoning(BaseModel):
    """Channel-era reasoning shape accepted only while reading serialized v0.1 messages.

    Deprecated as a standalone type: match on the ``AnyReasoningBlock`` kinds
    (ReasoningBlock / SignedReasoningBlock / RedactedReasoningBlock /
    ReasoningRefBlock / EncryptedReasoningBlock) instead.
    """

    signature: str | None = None
    content: str


# Assistant blocks: one assistant turn is an ordered sequence of blocks, in the
# model's actual emission order (thinking → text → thinking → tool call, ...).
class TextBlock(BaseModel, frozen=True):
    """One contiguous run of answer text in an assistant turn.

    ``signature`` carries opaque passback state attached to this exact block,
    e.g. a Google thought signature emitted on a visible text part. A client
    that cannot re-emit the signature must reject passback rather than silently
    stripping it.
    """

    kind: Literal["text"] = "text"
    text: str
    signature: str | None = None


class ReasoningBlock(BaseModel, frozen=True):
    """In-band reasoning text with no passback token.

    E.g. ``reasoning_content`` on Chat Completions-compatible hosts, or
    <think>-tag extraction.
    """

    kind: Literal["reasoning"] = "reasoning"
    content: str


class SignedReasoningBlock(BaseModel, frozen=True):
    """Reasoning bound to an opaque provider signature re-emitted verbatim on passback.

    E.g. Anthropic signed thinking blocks.
    """

    kind: Literal["signed_reasoning"] = "signed_reasoning"
    signature: str
    content: str = ""


class RedactedReasoningBlock(BaseModel, frozen=True):
    """Reasoning the provider withheld, replaced by an opaque payload.

    E.g. Anthropic ``redacted_thinking``: ``data`` must be re-emitted verbatim as a
    ``redacted_thinking`` block on passback. Carries no readable content.
    """

    kind: Literal["redacted_reasoning"] = "redacted_reasoning"
    data: str


class ReasoningRefBlock(BaseModel, frozen=True):
    """Reasoning held provider-side and passed back by reference.

    This is retained for providers that require an item-level handle on replay.
    OpenAI Responses continuation uses ``AssistantMessage.provider_response_id``
    instead and does not create this block.
    """

    kind: Literal["reasoning_ref"] = "reasoning_ref"
    id: str
    content: str | None = None
    summary: list[str] = Field(default_factory=list)


class EncryptedReasoningBlock(BaseModel, frozen=True):
    """Reasoning returned as an opaque encrypted payload for stateless passback.

    E.g. OpenAI Responses reasoning items requested with
    ``include: ["reasoning.encrypted_content"]`` (``store=false`` /
    zero-data-retention): the item — id, summary parts, and encrypted payload —
    is re-emitted verbatim in position on passback. The payload is opaque and
    non-inspectable.
    """

    kind: Literal["encrypted_reasoning"] = "encrypted_reasoning"
    id: str
    encrypted_content: str
    content: str | None = None
    summary: list[str] = Field(default_factory=list)


class OpaqueBlock(BaseModel, frozen=True):
    """Provider-native block the framework carries uninterpreted.

    For provider-issued marker/control blocks that must round-trip untouched:
    ``data`` holds the block's raw JSON (self-describing — the provider's own
    ``type`` field travels inside it). The framework preserves it in position
    through history, projections, and serialization so a client that understands
    the payload can re-emit it verbatim on passback; other clients fail loudly.
    """

    kind: Literal["opaque"] = "opaque"
    data: str


type AnyReasoningBlock = (
    ReasoningBlock | SignedReasoningBlock | RedactedReasoningBlock | ReasoningRefBlock | EncryptedReasoningBlock
)
"""The reasoning family: one kind per passback mechanism."""

REASONING_BLOCK_TYPES = (
    ReasoningBlock,
    SignedReasoningBlock,
    RedactedReasoningBlock,
    ReasoningRefBlock,
    EncryptedReasoningBlock,
)
"""Runtime mirror of ``AnyReasoningBlock`` for isinstance checks — keep in lockstep."""

# No current provider signs *and* references the same reasoning payload; if one
# appears, it becomes a new kind rather than fields grafted onto an existing one.
type AssistantBlock = Annotated[
    TextBlock
    | ReasoningBlock
    | SignedReasoningBlock
    | RedactedReasoningBlock
    | ReasoningRefBlock
    | EncryptedReasoningBlock
    | OpaqueBlock
    | ToolCall
    | ImageContentBlock,
    Field(discriminator="kind"),
]
"""One block of an assistant turn, discriminated on ``kind``."""

type _MediaBlock = ImageContentBlock


def joined_text(blocks: Sequence[AssistantBlock]) -> str | None:
    """All answer text across text blocks, directly concatenated; None when absent."""
    texts = [block.text for block in blocks if isinstance(block, TextBlock)]
    if not texts:
        return None
    return "".join(texts)


def final_text(blocks: Sequence[AssistantBlock]) -> str | None:
    """Text of the last text block — the "answer" in thinking→text→thinking→text turns."""
    for block in reversed(blocks):
        if isinstance(block, TextBlock):
            return block.text
    return None


def tool_call_blocks(blocks: Sequence[AssistantBlock]) -> list[ToolCall]:
    """Tool calls in emission order."""
    return [block for block in blocks if isinstance(block, ToolCall)]


def reasoning_blocks(blocks: Sequence[AssistantBlock]) -> list[AnyReasoningBlock]:
    """Reasoning blocks (any kind) in emission order."""
    return [block for block in blocks if isinstance(block, REASONING_BLOCK_TYPES)]


def _reasoning_to_block(reasoning: object) -> object:
    """Upgrade a flat channel-era ``Reasoning`` value to its block equivalent.

    A signature means signed passback; bare content is in-band reasoning.
    """
    match reasoning:
        case Reasoning(signature=signature, content=content):
            pass
        case {**mapping}:
            signature = mapping.get("signature")
            content = mapping.get("content")
        case _:
            return reasoning  # let block validation fail loudly
    if signature is not None:
        if not isinstance(signature, str):
            return reasoning
        return SignedReasoningBlock(signature=signature, content=content if isinstance(content, str) else "")
    return ReasoningBlock(content=content if isinstance(content, str) else "")


_CHANNEL_PROJECTION_DEPRECATION = (
    "AssistantMessage.{channel} is deprecated; {replacement}. "
    "The compatibility projection will be removed in a future release."
)


def _warn_channel_projection(channel: str, replacement: str) -> None:
    warnings.warn(
        _CHANNEL_PROJECTION_DEPRECATION.format(channel=channel, replacement=replacement),
        DeprecationWarning,
        stacklevel=3,
    )


def _upgrade_legacy_assistant_message(data: object) -> object:
    """Read a serialized v0.1 assistant message into canonical blocks."""
    if not isinstance(data, dict):
        return data
    upgraded = dict(data)
    content = upgraded.pop("content", None)
    reasoning = upgraded.pop("reasoning", None)
    tool_calls = upgraded.pop("tool_calls", None)

    if "blocks" in upgraded:
        if content not in (None, "", []) or reasoning or tool_calls:
            raise ValueError("AssistantMessage cannot mix 'blocks' with legacy channels")
        return upgraded

    blocks: list[object] = []
    if reasoning:
        blocks.append(_reasoning_to_block(reasoning))
    if isinstance(content, str):
        if content:
            blocks.append(TextBlock(text=content))
    elif isinstance(content, list):
        blocks.extend(TextBlock(text=item) if isinstance(item, str) else item for item in content if item != "")
    elif content is not None:
        raise ValueError(f"AssistantMessage 'content' must be a string or list, got {type(content).__name__}")
    if tool_calls is not None and not isinstance(tool_calls, list | tuple):
        raise ValueError("AssistantMessage 'tool_calls' must be a sequence")
    blocks.extend({**call, "kind": "tool_call"} if isinstance(call, dict) else call for call in tool_calls or [])
    upgraded["blocks"] = blocks
    return upgraded


class AssistantMessage(BaseModel):
    """LLM response message: an ordered sequence of assistant blocks.

    ``blocks`` is the only stored content. The channel-era ``content`` and
    ``tool_calls`` attributes remain deprecated views; ``reasoning`` raises because
    an ordered reasoning block sequence has no faithful channel-shaped projection.
    Serialized v0.1 payloads upgrade to blocks during validation. Channel-shaped
    construction is not part of the v0.2 API; new code constructs blocks directly.
    Mixing ``blocks`` with non-empty legacy channel keys raises.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    provider_response_id: str | None = None
    """Provider-attached continuation state, e.g. an OpenAI Responses ``resp_...`` id.

    This is turn metadata rather than emitted assistant content, so it lives beside
    ``blocks`` instead of inside their emission order. It is distinct from ``id``
    (Stirrup's message identity) and ``ReasoningRefBlock.id`` (an emitted reasoning
    item handle).
    """
    role: Literal["assistant"] = "assistant"
    blocks: list[AssistantBlock] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_start_time: float | None = None
    request_end_time: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_channel_fields(cls, data: object) -> object:
        return _upgrade_legacy_assistant_message(data)

    # Channel-era compatibility accessors.
    @property
    def content(self) -> list[AssistantBlock] | str:
        """Bare text for one text block, empty text for no blocks, or the block list."""
        _warn_channel_projection("content", "inspect blocks or use joined_text(message.blocks)")
        match self.blocks:
            case []:
                return ""
            case [TextBlock(text=text)]:
                return text
            case blocks:
                return blocks

    @property
    def reasoning(self) -> Reasoning | None:
        """Deprecated channel accessor retained only to fail with migration guidance."""
        raise NotImplementedError(
            "AssistantMessage.reasoning is deprecated and no longer available; use reasoning_blocks(message.blocks)"
        )

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls in emission order."""
        _warn_channel_projection("tool_calls", "use tool_call_blocks(message.blocks)")
        return tool_call_blocks(self.blocks)

    @property
    def e2e_otps(self) -> float | None:
        """End-to-end output tokens per second."""
        if self.request_start_time is None or self.request_end_time is None:
            return None
        duration = self.request_end_time - self.request_start_time
        if duration <= 0 or self.token_usage.output <= 0:
            return None
        return self.token_usage.output / duration


class ToolMessage(BaseModel):
    """Tool execution result returned to the LLM.

    Attributes:
        role: Always "tool"
        content: The tool result content
        tool_call_id: ID linking this result to the corresponding tool call
        name: Name of the tool that was called
        args_was_valid: Whether the tool arguments were valid
        success: Whether the tool executed successfully (used by finish tool to control termination)
    """

    role: Literal["tool"] = "tool"
    content: Content
    tool_call_id: str = Field(min_length=1)
    name: str | None = None
    args_was_valid: bool = True
    success: bool = False
    tool_start_time: float | None = None
    tool_end_time: float | None = None

    @property
    def tool_duration(self) -> float | None:
        """Tool execution duration in seconds."""
        if self.tool_start_time is None or self.tool_end_time is None:
            return None
        duration = self.tool_end_time - self.tool_start_time
        if duration < 0:
            return None
        return duration


def _reject_untagged_user_message(data: object) -> object:
    if isinstance(data, dict) and "kind" not in data:
        raise ValueError(
            "Cannot load a cached user message without the required 'kind' discriminator. "
            "This cache predates typed user-message kinds and cannot be resumed safely; "
            "delete the old Stirrup cache and restart the run."
        )
    return data


type UserRoleMessage = Annotated[
    UserMessage | SummaryMessage | TurnWarningMessage,
    Field(discriminator="kind"),
    BeforeValidator(_reject_untagged_user_message),
]
"""User-role messages, discriminated on ``kind`` — the agent-injected ``UserMessage``
subclasses share ``role="user"``, so dumped histories need the nested discriminator
to rehydrate them as their own types (e.g. ``SummaryMessage.replaced_ids``)."""

type ChatMessage = Annotated[
    SystemMessage | UserRoleMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]
"""Discriminated union of all message types, automatically parsed based on role field."""
