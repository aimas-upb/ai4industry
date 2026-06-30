import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Artifact KG URIs
ARTIFACT_REGISTRY = {
    "APAS": "https://ci.mines-stetienne.fr/kg/itmfactory/bosch-apas#this",
    "DX10": "https://ci.mines-stetienne.fr/kg/itmfactory/dx10#this",
    "XY10": "https://ci.mines-stetienne.fr/kg/itmfactory/xy10#this",
    "VL10": "https://ci.mines-stetienne.fr/kg/itmfactory/vl10#this",
}

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
