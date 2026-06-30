"""
LLM model configuration for OpenAI and Mistral.

Defines model-specific parameters based on model family.
For OpenAI, uses the Responses API (v1/responses endpoint).
"""

from typing import Dict, Any


class OpenAIModelConfig:
    """Configuration for OpenAI models using the Responses API."""

    @staticmethod
    def get_params(model: str) -> Dict[str, Any]:
        """
        Get API parameters for an OpenAI model (Responses API).

        - GPT-5 series: reasoning models with reasoning effort medium
        - GPT-4 series: non-reasoning models

        Args:
            model: The model name (e.g., "gpt-5-mini", "gpt-4o")

        Returns:
            Dictionary of parameters to pass to the Responses API
        """
        params = {}

        if model.startswith("gpt-5"):
            # GPT-5 series: reasoning models (Responses API only accepts effort, not type)
            params["reasoning"] = {
                "effort": "medium",
            }
        elif model.startswith("gpt-4"):
            # GPT-4 series: non-reasoning models
            params["text"] = {
                "format": "text",
            }

        return params


class MistralModelConfig:
    """Configuration for Mistral models."""

    @staticmethod
    def get_params(model: str) -> Dict[str, Any]:
        """
        Get API parameters for a Mistral model.

        Args:
            model: The model name

        Returns:
            Dictionary of parameters to pass to the Mistral API
        """
        return {
            "temperature": 0,
            "max_tokens": 16000,
        }
