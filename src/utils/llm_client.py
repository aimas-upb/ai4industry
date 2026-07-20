"""
Unified LLM client interface with provider-specific implementations.

`LLMClient` is an abstract base defining the interface used by the rest of the
codebase. Concrete subclasses handle the differences between providers:

- `OpenAIResponsesClient`: OpenAI Responses API (reasoning models, e.g. gpt-5.4-mini).
- `MistralChatClient`: Mistral via the OpenAI-compatible chat.completions API.

Both support reasoning models and tool-use loops. Callers construct a client via
`LLMClient.create()` (or `LLMClient()`, which dispatches to the correct subclass)
and interact only through the abstract interface.
"""

import logging
from abc import ABC, abstractmethod
from openai import OpenAI

from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_API_KEY,
    MISTRAL_API_KEY,
    MISTRAL_BASE_URL,
    get_llm_params,
)

logger = logging.getLogger(__name__)


class ToolCall:
    """
    Provider-agnostic representation of a single tool/function call.

    Exposes the fields the rest of the codebase relies on regardless of which
    provider produced it.
    """

    def __init__(self, name: str, arguments: str, call_id: str, raw=None):
        """
        Args:
            name: Tool/function name.
            arguments: JSON string of arguments.
            call_id: Provider tool-call id, used to link the tool result message
                back to this call in the conversation history.
            raw: The provider's original tool-call object (opaque). Used by
                providers that must echo the exact item back into history.
        """
        self.name = name
        self.arguments = arguments
        self.call_id = call_id
        self.raw = raw


class LLMClient(ABC):
    """
    Abstract interface for LLM interactions with tool-calling support.

    The rest of the codebase depends only on this interface:
      - `call_with_tools(system_prompt, messages, tools) -> (tool_calls, content)`
      - `append_tool_turn(messages, content, tool_calls, results)`

    Construct via `LLMClient.create()` or `LLMClient()`; both return the concrete
    client for the configured provider.
    """

    def __new__(cls, *args, **kwargs):
        """
        Allow `LLMClient()` to transparently build the correct subclass.

        When instantiated directly (not via a subclass), dispatch to the provider
        implementation selected by configuration so existing call sites keep working.
        """
        if cls is LLMClient:
            return LLMClient.create()
        return super().__new__(cls)

    @staticmethod
    def create() -> "LLMClient":
        """
        Factory that returns the concrete client for the configured provider.

        Returns:
            An `OpenAIResponsesClient` or `MistralChatClient` instance.

        Raises:
            ValueError: If `LLM_PROVIDER` is not a supported value.
        """
        if LLM_PROVIDER == "openai":
            return OpenAIResponsesClient()
        elif LLM_PROVIDER == "mistral":
            return MistralChatClient()
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    @abstractmethod
    def call_with_tools(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
    ) -> tuple[list, str]:
        """
        Call the LLM with tool/function calling support.

        Args:
            system_prompt: System context (instructions).
            messages: Conversation history in this client's native message format.
            tools: List of tool definitions (each with name/description/parameters).

        Returns:
            Tuple of (tool_calls, content):
              - tool_calls: List of `ToolCall` objects (.name, .arguments, .call_id).
              - content: Any assistant text (thinking excluded), as a string.
        """
        raise NotImplementedError

    @abstractmethod
    def append_tool_turn(
        self,
        messages: list,
        content: str,
        tool_calls: list,
        results: list,
    ) -> None:
        """
        Append one assistant tool-use turn and its results to the message history.

        A tool-use loop must record, in provider-correct form, both the assistant's
        tool calls and the results returned for them; otherwise the model cannot see
        what it already requested and keeps re-issuing the same calls without
        converging. Mutates `messages` in place.

        Args:
            messages: The running conversation history (mutated in place).
            content: The assistant's text content for this turn (may be empty).
            tool_calls: The `ToolCall` objects returned by `call_with_tools`.
            results: The tool result strings, index-aligned with `tool_calls`.

        Returns:
            None. `messages` is updated in place.
        """
        raise NotImplementedError


class OpenAIResponsesClient(LLMClient):
    """
    LLM client backed by the OpenAI Responses API.

    Supports reasoning models (e.g. gpt-5.4-mini). To keep the reasoning chain
    intact across tool calls, the raw output items (including reasoning items) are
    echoed back into `input` on the next turn, alongside function_call_output items.
    """

    def __init__(self):
        """Initialize the OpenAI client and record the configured model."""
        self.client = OpenAI(api_key=LLM_API_KEY, timeout=600)
        self.model = LLM_MODEL

    def call_with_tools(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
    ) -> tuple[list, str]:
        """Call the Responses API and extract tool calls + text (thinking excluded)."""
        llm_params = get_llm_params()

        # Responses API takes function tools in a flat shape.
        api_tools = [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in tools
        ]

        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=messages,
            tools=api_tools,
            **llm_params,
        )

        tool_calls = []
        content = ""

        # Extract tool calls and text from Responses API output.
        for item in response.output:
            if hasattr(item, "content") and item.content is not None:
                # ResponseOutputMessage: extract text, skip reasoning/thinking chunks.
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        content = content_item.text
            elif hasattr(item, "name") and hasattr(item, "arguments"):
                # ResponseFunctionToolCall: wrap it, keeping the raw item and call_id.
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                tool_calls.append(
                    ToolCall(
                        name=item.name,
                        arguments=item.arguments,
                        call_id=call_id,
                        raw=item,
                    )
                )

        # Preserve the full raw output (reasoning items included) so it can be echoed
        # back into history on the next turn, keeping the reasoning chain intact.
        self._last_output = list(response.output)

        return tool_calls, content

    def append_tool_turn(
        self,
        messages: list,
        content: str,
        tool_calls: list,
        results: list,
    ) -> None:
        """
        Append the assistant turn and tool outputs to Responses-API history.

        Echoes back the raw output items from the last response (reasoning +
        function_call items) so reasoning models keep their chain, then appends one
        function_call_output item per result keyed by call_id.
        """
        # Echo the raw output items (reasoning + tool calls + any message) verbatim.
        last_output = getattr(self, "_last_output", None)
        if last_output:
            messages.extend(last_output)
        else:
            # Fallback: reconstruct minimal function_call items if raw output is missing.
            for tool_call in tool_calls:
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                )

        # Append the tool results, each linked to its call by call_id.
        for tool_call, result in zip(tool_calls, results):
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                }
            )


class MistralChatClient(LLMClient):
    """
    LLM client backed by Mistral's OpenAI-compatible chat.completions API.

    Supports reasoning models (e.g. mistral-medium). Reasoning models return
    message.content as a list of chunks (thinking + text); thinking chunks are
    stripped from any re-sent content because Mistral rejects ThinkChunks in
    user/assistant messages.
    """

    def __init__(self):
        """Initialize the Mistral (OpenAI-compatible) client and record the model."""
        self.client = OpenAI(
            api_key=MISTRAL_API_KEY,
            base_url=MISTRAL_BASE_URL,
            timeout=600,
        )
        self.model = LLM_MODEL

    def call_with_tools(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
    ) -> tuple[list, str]:
        """Call chat.completions and extract tool calls + text (thinking excluded)."""
        llm_params = get_llm_params()

        # System prompt is a leading system message for chat.completions.
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages

        # chat.completions nests function tools under a "function" key.
        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages_with_system,
            tools=api_tools,
            tool_choice="auto",
            **llm_params,
        )

        tool_calls = []
        choice = response.choices[0]

        # With reasoning enabled, content may be a list of chunks (thinking + text);
        # keep only the text so ThinkChunks never re-enter the message history.
        content = self._extract_text_content(choice.message.content)

        # Wrap tool calls, preserving the id so results can be linked back to them.
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        call_id=tc.id,
                        raw=tc,
                    )
                )

        return tool_calls, content

    def append_tool_turn(
        self,
        messages: list,
        content: str,
        tool_calls: list,
        results: list,
    ) -> None:
        """
        Append the assistant turn and tool outputs to chat.completions history.

        Emits one assistant message carrying all tool_calls (with plain-text content
        only, thinking already stripped), followed by one tool message per result
        keyed by tool_call_id.
        """
        messages.append(
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )
        for tool_call, result in zip(tool_calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id,
                    "content": result,
                }
            )

    def _extract_text_content(self, message_content) -> str:
        """
        Extract plain text from a chat.completions message content field.

        With reasoning enabled, Mistral returns `content` as a list of chunks
        (thinking + text) instead of a plain string. Thinking chunks must not be
        re-sent to the API, so this collects only the text chunks and logs any
        thinking separately at debug level.

        Args:
            message_content: The `choice.message.content` value — either a str or a
                list of chunk objects/dicts.

        Returns:
            The concatenated text content as a string (empty if none).
        """
        # Plain string (no reasoning): return as-is.
        if message_content is None:
            return ""
        if isinstance(message_content, str):
            return message_content

        # List of chunks (reasoning enabled): keep text, drop/log thinking.
        text_parts = []
        for chunk in message_content:
            # Support both dict-shaped chunks and SDK objects via getattr fallback.
            chunk_type = chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)

            if chunk_type == "thinking":
                # Log thinking separately (only surfaces at debug level to avoid spam).
                thinking = chunk.get("thinking") if isinstance(chunk, dict) else getattr(chunk, "thinking", None)
                logger.debug(f"Mistral thinking chunk: {thinking}")
            elif chunk_type == "text":
                text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", None)
                if text:
                    text_parts.append(text)

        return "".join(text_parts)
