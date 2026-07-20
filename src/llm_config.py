"""
LLM model configuration for OpenAI and Mistral.

Defines model-specific parameters based on model family.
For OpenAI, uses the Responses API (v1/responses endpoint).

Reasoning effort follows a two-tier scheme: each config declares a hardcoded
default effort, and the `LLM_REASONING_EFFORT` environment variable — when set —
overrides that default for any model that supports reasoning.
"""

from typing import Dict, Any

from src.config import LLM_REASONING_EFFORT


def resolve_reasoning_effort(default: str) -> str:
    """
    Resolve the reasoning effort to use, letting the environment override the default.

    Args:
        default: The hardcoded default effort declared by the model config.

    Returns:
        The `LLM_REASONING_EFFORT` env value if it is set (non-empty), otherwise
        the provided default.
    """
    # Env var overrides the hardcoded default only when explicitly set
    if LLM_REASONING_EFFORT:
        return LLM_REASONING_EFFORT
    return default


class OpenAIModelConfig:
    """Configuration for OpenAI models using the Responses API."""

    # Hardcoded default reasoning effort for GPT-5 reasoning models,
    # overridable via the LLM_REASONING_EFFORT environment variable.
    DEFAULT_REASONING_EFFORT = "medium"

    @staticmethod
    def get_params(model: str) -> Dict[str, Any]:
        """
        Get API parameters for an OpenAI model (Responses API).

        - GPT-5 series: reasoning models; effort defaults to DEFAULT_REASONING_EFFORT
          and can be overridden by the LLM_REASONING_EFFORT env var.
        - GPT-4 series: non-reasoning models.

        Args:
            model: The model name (e.g., "gpt-5.4-mini", "gpt-4o")

        Returns:
            Dictionary of parameters to pass to the Responses API
        """
        params: Dict[str, Any] = {}

        if model.startswith("gpt-5"):
            # GPT-5 series: reasoning models (Responses API accepts effort, not type)
            params["reasoning"] = {
                "effort": resolve_reasoning_effort(
                    OpenAIModelConfig.DEFAULT_REASONING_EFFORT
                ),
            }
        elif model.startswith("gpt-4"):
            # GPT-4 series: non-reasoning models
            params["text"] = {
                "format": "text",
            }

        return params


class MistralModelConfig:
    """Configuration for Mistral models."""

    # Hardcoded default reasoning effort for Mistral reasoning models,
    # overridable via the LLM_REASONING_EFFORT environment variable.
    DEFAULT_REASONING_EFFORT = "high"

    @staticmethod
    def get_params(model: str) -> Dict[str, Any]:
        """
        Get API parameters for a Mistral model.

        Reasoning models (mistral-medium*) get a reasoning_effort that defaults to
        DEFAULT_REASONING_EFFORT and can be overridden by the LLM_REASONING_EFFORT
        env var.

        Args:
            model: The model name

        Returns:
            Dictionary of parameters to pass to the Mistral API
        """
        params: Dict[str, Any] = {
            "temperature": 0,
            "max_tokens": 32000,
        }

        if model.startswith("mistral-medium"):
            params["reasoning_effort"] = resolve_reasoning_effort(
                MistralModelConfig.DEFAULT_REASONING_EFFORT
            )
            params["top_p"] = 1.0

        return params
