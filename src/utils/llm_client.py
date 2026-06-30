"""
Unified LLM client wrapper for both Responses API (OpenAI) and chat.completions (Mistral).

Abstracts away provider differences so calling code remains clean.
"""

import json
import logging
from typing import Optional
from openai import OpenAI

from src.config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, MISTRAL_API_KEY, MISTRAL_BASE_URL, get_llm_params

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified interface for LLM interactions.

    Handles both OpenAI Responses API and Mistral chat.completions transparently.
    Abstracts response parsing so callers don't need to know which API is used.
    """

    def __init__(self):
        """Initialize OpenAI client with appropriate base_url and API key."""
        if LLM_PROVIDER == "mistral":
            self.client = OpenAI(
                api_key=MISTRAL_API_KEY,
                base_url=MISTRAL_BASE_URL,
                timeout=600,
            )
        else:
            self.client = OpenAI(api_key=LLM_API_KEY, timeout=600)

        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL

    def call_with_tools(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
    ) -> tuple[list, str]:
        """
        Call LLM with tool/function calling support.

        Args:
            system_prompt: System context (instructions)
            messages: Conversation messages (user/assistant/tool roles)
            tools: List of tool definitions

        Returns:
            Tuple of (tool_calls list, text_content string)
            - tool_calls: List of tool call objects with .name and .arguments
            - text_content: Any text response from the model
        """
        llm_params = get_llm_params()

        if self.provider == "openai":
            return self._call_responses_api(system_prompt, messages, tools, llm_params)
        else:
            return self._call_chat_completions(system_prompt, messages, tools, llm_params)

    def _call_responses_api(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
        llm_params: dict,
    ) -> tuple[list, str]:
        """Call OpenAI Responses API and extract tool calls + text."""
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=messages,
            tools=tools,
            **llm_params,
        )

        tool_calls = []
        content = ""

        # Extract tool calls and text from Responses API output
        for item in response.output:
            if hasattr(item, "content"):
                # ResponseOutputMessage: extract text from content list
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        content = content_item.text
            elif hasattr(item, "name") and hasattr(item, "arguments"):
                # ResponseFunctionToolCall: extract tool call
                tool_calls.append(item)

        return tool_calls, content

    def _call_chat_completions(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
        llm_params: dict,
    ) -> tuple[list, str]:
        """Call chat.completions API (Mistral) and extract tool calls + text."""
        # Prepend system prompt to messages for chat.completions
        messages_with_system = [
            {"role": "system", "content": system_prompt}
        ] + messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages_with_system,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ],
            tool_choice="auto",
            **llm_params,
        )

        tool_calls = []
        content = ""

        choice = response.choices[0]
        if choice.message.content:
            content = choice.message.content

        # Wrap chat.completions tool calls to match Responses API format
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                # Create a wrapper object that mimics ResponseFunctionToolCall
                tool_calls.append(
                    _ToolCallWrapper(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return tool_calls, content


class _ToolCallWrapper:
    """
    Wraps chat.completions tool calls to match Responses API interface.

    Allows calling code to treat both APIs uniformly.
    """

    def __init__(self, name: str, arguments: str):
        """
        Args:
            name: Tool/function name
            arguments: JSON string of arguments
        """
        self.name = name
        self.arguments = arguments
