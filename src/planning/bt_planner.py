import json
import logging
from typing import Optional

from openai import OpenAI

from src.config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, MISTRAL_API_KEY, MISTRAL_BASE_URL
from src.discovery.capability_model import CapabilityModel
from src.planning.prompts import create_system_prompt

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
RESET = "\033[0m"

# BT JSON-IR schema for the LLM
BT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["sequence", "selector", "parallel", "action", "condition"],
            "description": "Node type",
        },
        "name": {
            "type": "string",
            "description": "Human-readable node name",
        },
        "children": {
            "type": "array",
            "items": {"$ref": "#"},
            "description": "Child nodes (for composite nodes)",
        },
        "policy": {
            "type": "string",
            "enum": ["success_on_all", "success_on_one"],
            "description": "Policy for parallel nodes",
        },
        "action_url": {
            "type": "string",
            "description": "HTTP endpoint URL (for action nodes)",
        },
        "parameters": {
            "type": "object",
            "description": "Parameters for the action",
        },
        "property_url": {
            "type": "string",
            "description": "Property endpoint URL (for condition nodes)",
        },
        "expected_value": {
            "description": "Expected value to compare against (for condition nodes)",
        },
        "operator": {
            "type": "string",
            "enum": ["==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "matches"],
            "description": "Comparison operator (for condition nodes)",
        },
    },
    "required": ["type", "name"],
}


class BTPlanner:
    """
    Generate BehaviorTree plans using an LLM.

    Uses function calling to generate valid BT JSON-IR structures and validates
    the output before returning. Logs all planning steps at DEBUG level.
    """

    def __init__(self, capability_model: CapabilityModel, goal: str = ""):
        """
        Initialize the BT planner with a capability model and optional goal.

        Args:
            capability_model: The CapabilityModel containing discovered affordances
            goal: The goal predicate instance (e.g., '!carry("APAS", "DX10_output", "XY10_input")')
        """
        if LLM_PROVIDER == "mistral":
            self.client = OpenAI(
                api_key=MISTRAL_API_KEY,
                base_url=MISTRAL_BASE_URL,
            )
        else:
            self.client = OpenAI(api_key=LLM_API_KEY)

        self.capability_model = capability_model
        self.goal = goal
        self.system_prompt = create_system_prompt(
            capability_model.to_prompt_context(),
            goal_predicate=goal
        )
        logger.debug("BTPlanner initialized")

    def plan(self, goal: str) -> dict:
        """
        Generate a BehaviorTree plan for the given goal.

        Uses an LLM with function calling to generate a valid BT JSON-IR structure.
        Validates the structure and retries up to 3 times if validation fails.

        Args:
            goal: The achievement goal to plan for

        Returns:
            A valid BT JSON-IR dictionary structure

        Raises:
            ValueError: If unable to generate a valid BT after 3 attempts
        """
        logger.debug(f"Starting BT planning for goal: {goal}")
        messages = [
            {
                "role": "user",
                "content": f"Goal: {goal}\n\nGenerate a BehaviorTree JSON structure to solve this goal.",
            }
        ]

        for attempt in range(3):
            logger.debug(f"BT generation attempt {attempt + 1}/3")
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                system=self.system_prompt,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "generate_behavior_tree",
                            "description": "Generate a BehaviorTree to solve the goal",
                            "parameters": BT_TOOL_SCHEMA,
                        },
                    }
                ],
                tool_choice="auto",
            )

            # Check for tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "generate_behavior_tree":
                        bt_dict = json.loads(tool_call.function.arguments)
                        logger.debug("BT JSON generated, validating structure")

                        # Validate structure
                        if self._validate_tree(bt_dict):
                            logger.debug("BT structure is valid, returning plan")
                            return bt_dict
                        else:
                            # Invalid structure, retry
                            logger.debug("BT structure invalid, retrying with error feedback")
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": response.content,
                                }
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "The tree structure is invalid. Please ensure all action nodes have 'action_url' and all condition nodes have 'property_url'. Try again.",
                                }
                            )
                            continue

            # If no tool calls, try to extract JSON from text
            if response.content:
                try:
                    logger.debug("No tool call detected, attempting to parse JSON from response text")
                    # Try to parse JSON from the response
                    json_str = response.content
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    bt_dict = json.loads(json_str)
                    if self._validate_tree(bt_dict):
                        logger.debug("Extracted and validated BT from text")
                        return bt_dict
                except Exception as e:
                    logger.debug(f"Failed to parse JSON from text: {e}")
                    pass

        error_msg = f"{YELLOW}[ERROR]{RESET} Failed to generate valid BehaviorTree after 3 attempts"
        logger.error(error_msg)
        raise ValueError("Failed to generate valid BehaviorTree after 3 attempts")

    def _validate_tree(self, tree: dict) -> bool:
        """Validate BT structure."""
        if not isinstance(tree, dict):
            return False

        node_type = tree.get("type")
        if node_type not in ["sequence", "selector", "parallel", "action", "condition"]:
            return False

        if not tree.get("name"):
            return False

        # Composite nodes must have children
        if node_type in ["sequence", "selector", "parallel"]:
            if "children" not in tree or not isinstance(tree["children"], list):
                return False
            return all(self._validate_tree(child) for child in tree["children"])

        # Action nodes must have action_url
        if node_type == "action":
            return "action_url" in tree and isinstance(tree["action_url"], str)

        # Condition nodes must have property_url
        if node_type == "condition":
            return "property_url" in tree and isinstance(tree["property_url"], str)

        return True
