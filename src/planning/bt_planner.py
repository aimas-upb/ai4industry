import json
import logging

from src.utils.llm_client import LLMClient
from src.discovery.capability_model import CapabilityModel
from src.utils.models import GoalSpecification
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

    def __init__(self, capability_model: CapabilityModel, goal_spec: GoalSpecification = None):
        """
        Initialize the BT planner with a capability model and goal specification.

        Args:
            capability_model: The CapabilityModel containing discovered affordances
            goal_spec: GoalSpecification containing schema and instance
        """
        self.llm = LLMClient()
        self.capability_model = capability_model
        self.goal_spec = goal_spec or GoalSpecification("", "")
        # System prompt should not include goal_predicate; that goes in user message
        self.system_prompt = create_system_prompt(
            capability_model.to_prompt_context()
        )
        logger.debug("BTPlanner initialized")

    def plan(self, goal_spec: GoalSpecification) -> dict:
        """
        Generate a BehaviorTree plan for the given goal.

        Uses the Responses API with function calling to generate a valid BT JSON-IR structure.
        Validates the structure and retries up to 3 times if validation fails.

        Args:
            goal_spec: GoalSpecification containing schema and instance

        Returns:
            A valid BT JSON-IR dictionary structure

        Raises:
            ValueError: If unable to generate a valid BT after 3 attempts
        """
        logger.debug(f"Starting BT planning for goal: {goal_spec.instance}")

        # Build user prompt with goal schema and instance
        user_prompt = f"Goal schema: {goal_spec.schema}\nGoal instance: {goal_spec.instance}\n\nGenerate a BehaviorTree JSON structure to solve this goal."

        # For Responses API, messages contain only user and tool result messages (no system role)
        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        for attempt in range(3):
            logger.debug(f"BT generation attempt {attempt + 1}/3")

            # Call LLM with tools (abstracts away provider differences)
            try:
                tool_calls, content = self.llm.call_with_tools(
                    self.system_prompt,
                    messages,
                    [
                        {
                            "type": "function",
                            "name": "generate_behavior_tree",
                            "description": "Generate a BehaviorTree to solve the goal",
                            "parameters": BT_TOOL_SCHEMA,
                        }
                    ],
                )
            except KeyboardInterrupt:
                logger.warning(f"BT generation attempt {attempt + 1}/3 interrupted by user")
                raise
            except Exception as e:
                logger.error(f"BT generation attempt {attempt + 1}/3 failed: {e}", exc_info=True)
                continue

            # Try tool call first (preferred path)
            if tool_calls:
                result = self._process_tool_calls(tool_calls, content, messages)
                if result:
                    return result

            # Fallback: parse JSON from text response
            if content:
                bt_dict = self._extract_json_from_text(content)
                if bt_dict and self._validate_tree(bt_dict):
                    logger.debug("Extracted and validated BT from text response")
                    return bt_dict

        error_msg = f"{YELLOW}[ERROR]{RESET} Failed to generate valid BehaviorTree after 3 attempts"
        logger.error(error_msg)
        raise ValueError("Failed to generate valid BehaviorTree after 3 attempts")

    def _process_tool_calls(self, tool_calls: list, content: str, messages: list) -> dict | None:
        """
        Process tool calls from LLM response. Returns BT if valid, None if retry needed.

        Args:
            tool_calls: List of tool call objects
            content: Text content from response
            messages: Messages list to append retry feedback to

        Returns:
            Valid BT dict if successful, None to continue retry loop
        """
        for tool_call in tool_calls:
            if tool_call.name == "generate_behavior_tree":
                bt_dict = json.loads(tool_call.arguments)
                if self._validate_tree(bt_dict):
                    return bt_dict
                else:
                    # Append error feedback for retry
                    if content:
                        messages.append({"role": "user", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": "The tree structure is invalid. Please ensure all action nodes have 'action_url' and all condition nodes have 'property_url'. Try again.",
                        }
                    )
                    break
        return None

    def _extract_json_from_text(self, text: str) -> dict | None:
        """
        Extract JSON from text response, handling markdown code blocks.

        Args:
            text: Text response from LLM

        Returns:
            Parsed JSON dict, or None if extraction fails
        """
        try:
            json_str = text
            # Remove markdown code block markers
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return None

    def _validate_tree(self, tree: dict) -> bool:
        """Validate BT structure."""
        if not isinstance(tree, dict):
            logger.debug(f"Validation failed: not a dict, got {type(tree)}")
            return False

        node_type = tree.get("type")
        if node_type not in ["sequence", "selector", "parallel", "action", "condition"]:
            logger.debug(f"Validation failed: invalid type '{node_type}'")
            return False

        if not tree.get("name"):
            logger.debug("Validation failed: missing 'name'")
            return False

        # Composite nodes must have children
        if node_type in ["sequence", "selector", "parallel"]:
            if "children" not in tree or not isinstance(tree["children"], list):
                logger.debug(f"Validation failed: {node_type} missing or invalid 'children'")
                return False
            return all(self._validate_tree(child) for child in tree["children"])

        # Action nodes must have action_url
        if node_type == "action":
            if "action_url" not in tree:
                logger.debug("Validation failed: action node missing 'action_url'")
                return False
            if not isinstance(tree["action_url"], str):
                logger.debug(f"Validation failed: action_url is not string, got {type(tree['action_url'])}")
                return False
            return True

        # Condition nodes must have property_url
        if node_type == "condition":
            if "property_url" not in tree:
                logger.debug("Validation failed: condition node missing 'property_url'")
                return False
            if not isinstance(tree["property_url"], str):
                logger.debug(f"Validation failed: property_url is not string, got {type(tree['property_url'])}")
                return False
            return True

        return True
