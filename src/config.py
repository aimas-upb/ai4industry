import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "mistral"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")

# If using Mistral via OpenAI-compatible endpoint
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# RDF/examples fallback
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Simulator settings
SIMULATOR_GROUP = int(os.getenv("SIMULATOR_GROUP", "10"))  # Default group 10
SIMULATOR_USERNAME = f"simu{SIMULATOR_GROUP}"
SIMULATOR_PASSWORD = f"simu{SIMULATOR_GROUP}"

# Discovery settings
MAX_DISCOVERY_ITERATIONS = 10

# Plan caching settings
PLAN_CACHE_PATH = os.getenv("PLAN_CACHE_PATH")
if PLAN_CACHE_PATH and PLAN_CACHE_PATH.strip():
    PLAN_CACHE_PATH = Path(PLAN_CACHE_PATH)
else:
    PLAN_CACHE_PATH = None

# Flag to control whether cached plans are reused (set by run.py --with-plan-caching)
PLAN_CACHE_REUSE_ENABLED = os.getenv("PLAN_CACHE_ENABLED", "false").lower() == "true"


def get_llm_params() -> dict:
    """
    Get LLM API parameters based on configured provider and model.

    Returns dictionary with model-specific parameters (temperature, max_tokens, reasoning_effort, etc.)
    """
    from src.llm_config import OpenAIModelConfig, MistralModelConfig

    if LLM_PROVIDER == "openai":
        return OpenAIModelConfig.get_params(LLM_MODEL)
    elif LLM_PROVIDER == "mistral":
        return MistralModelConfig.get_params(LLM_MODEL)
    else:
        return {}
