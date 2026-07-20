import json
import re
import logging
from typing import Optional
from rdflib import Graph

from src.config import MAX_DISCOVERY_ITERATIONS
from src.utils.llm_client import LLMClient
from src.discovery.capability_model import (
    CapabilityModel,
    Artifact,
    Affordance,
)
from src.discovery.td_tools import (
    fetch_graph,
    get_thing_description,
    list_action_affordances,
    list_property_affordances,
    get_location_info,
)
from src.discovery.prompts import DISCOVERY_SYSTEM_PROMPT, create_discovery_user_prompt

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class AgenticDiscovery:
    def __init__(self):
        # Initialize OpenAI client for Responses API
        # Use Mistral client with base_url if Mistral provider, otherwise use standard OpenAI
        # Add timeout for Responses API calls (in seconds)
        self.llm = LLMClient()

        # Tool definitions (works with both Responses API and chat.completions)
        self.discovery_tools = [
            {
                "type": "function",
                "name": "fetch_artifact_graph",
                "description": "Fetch and load the RDF graph for an artifact",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_name": {
                            "type": "string",
                            "description": "Name of the artifact (e.g., APAS, DX10, XY10)",
                        }
                    },
                    "required": ["artifact_name"],
                },
            },
            {
                "type": "function",
                "name": "inspect_thing_description",
                "description": "Get basic metadata about a Thing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_name": {
                            "type": "string",
                            "description": "Name of the artifact",
                        }
                    },
                    "required": ["artifact_name"],
                },
            },
            {
                "type": "function",
                "name": "list_actions",
                "description": "List all action affordances for an artifact",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_name": {
                            "type": "string",
                            "description": "Name of the artifact",
                        }
                    },
                    "required": ["artifact_name"],
                },
            },
            {
                "type": "function",
                "name": "list_properties",
                "description": "List all property affordances for an artifact",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_name": {
                            "type": "string",
                            "description": "Name of the artifact",
                        }
                    },
                    "required": ["artifact_name"],
                },
            },
            {
                "type": "function",
                "name": "get_location",
                "description": "Get location/coordinate information for an artifact",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_name": {
                            "type": "string",
                            "description": "Name of the artifact",
                        }
                    },
                    "required": ["artifact_name"],
                },
            },
            {
                "type": "function",
                "name": "done_discovering",
                "description": "Signal that discovery is complete",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of discovered affordances",
                        }
                    },
                    "required": ["summary"],
                },
            },
        ]

        # Cache for fetched graphs
        self.graphs = {}

    def discover(self, goal: str) -> CapabilityModel:
        """
        Run agentic discovery loop to discover affordances from RDF graphs.

        The LLM navigates RDF graphs from the root entry point using provided tools
        to extract action and property affordances for artifacts involved in the goal.

        Uses the Responses API for all LLM calls. System prompt is passed via the
        instructions parameter, and messages contain only user and tool result roles.

        Args:
            goal: The achievement goal to discover affordances for

        Returns:
            CapabilityModel containing all discovered affordances and initial state
        """
        logger.debug(f"Starting agentic discovery for goal: {goal}")
        capability_model = CapabilityModel(goal=goal)

        # Get system and user prompts from prompts.py
        system_prompt = DISCOVERY_SYSTEM_PROMPT
        user_prompt = create_discovery_user_prompt(goal)

        # Initialize messages with user prompt only (system goes in instructions)
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        for iteration in range(MAX_DISCOVERY_ITERATIONS):
            logger.debug(f"Discovery iteration {iteration + 1}/{MAX_DISCOVERY_ITERATIONS}")

            try:
                # Call LLM with tools (abstracts away provider differences)
                tool_calls, content = self.llm.call_with_tools(
                    system_prompt,
                    messages,
                    self.discovery_tools,
                )
            except KeyboardInterrupt:
                logger.warning(f"Discovery iteration {iteration + 1}/{MAX_DISCOVERY_ITERATIONS} interrupted by user")
                raise

            # Check if we're done (no tool calls remaining)
            if not tool_calls:
                logger.debug("No tool calls in response, breaking")
                break

            # Execute each tool call and collect results (index-aligned with tool_calls)
            # so the whole assistant turn can be appended to history in provider-correct form.
            results = []
            done = False
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)

                # Log tool call with parameters
                args_str = json.dumps(tool_args, indent=2)
                logger.debug(f"{BLUE}[TOOL]{RESET} ({i+1}/{len(tool_calls)}) {tool_name}\n{args_str}")

                try:
                    result = self._execute_tool(
                        tool_name,
                        tool_args,
                        capability_model,
                    )
                    logger.debug(f"{BLUE}[TOOL]{RESET} {tool_name} → completed")
                except KeyboardInterrupt:
                    logger.warning(f"Tool execution interrupted by user during {tool_name}")
                    raise
                except Exception as e:
                    error_msg = f"{YELLOW}[ERROR]{RESET} Tool execution failed for {tool_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise

                results.append(result)

                # done_discovering terminates the loop once this turn is recorded
                if tool_name == "done_discovering":
                    done = True

            # Record the assistant tool-use turn and results in the correct format for
            # the active provider, so the model sees what it already fetched next turn.
            self.llm.append_tool_turn(messages, content, tool_calls, results)

            if done:
                logger.debug(f"Discovery phase complete. Found {len(capability_model.artifacts)} artifacts")
                return capability_model

        # Loop ended without an explicit done_discovering signal (e.g. iteration cap
        # reached). Build the model from whatever graphs were fetched so discovery
        # still yields artifacts instead of returning empty.
        if not capability_model.artifacts:
            logger.debug("Discovery loop ended without done_discovering; building model from fetched graphs")
            self._build_capability_model(capability_model)
        logger.debug(f"Discovery phase complete. Found {len(capability_model.artifacts)} artifacts")

        return capability_model

    def _get_thing_uri_from_graph(self, graph: Graph) -> Optional[str]:
        """Extract the Thing URI (with #this) from a graph."""
        for subject in graph.subjects():
            subject_str = str(subject)
            if "#this" in subject_str:
                return subject_str
        return None

    def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        capability_model: CapabilityModel,
    ) -> str:
        """Execute a discovery tool and return result."""
        if tool_name == "fetch_artifact_graph":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                graph = fetch_graph(artifact_name)
                if graph:
                    self.graphs[artifact_name] = graph
                    return f"Graph for {artifact_name} loaded successfully"
                else:
                    return f"Failed to load graph for {artifact_name}"
            return f"Graph for {artifact_name} already loaded"

        elif tool_name == "inspect_thing_description":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = self._get_thing_uri_from_graph(graph)
            if not thing_uri:
                return f"Could not find Thing URI in graph for {artifact_name}"
            metadata = get_thing_description(graph, thing_uri)
            return json.dumps(metadata)

        elif tool_name == "list_actions":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = self._get_thing_uri_from_graph(graph)
            if not thing_uri:
                return f"Could not find Thing URI in graph for {artifact_name}"
            actions = list_action_affordances(graph, thing_uri)
            return json.dumps(actions)

        elif tool_name == "list_properties":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = self._get_thing_uri_from_graph(graph)
            if not thing_uri:
                return f"Could not find Thing URI in graph for {artifact_name}"
            properties = list_property_affordances(graph, thing_uri)
            return json.dumps(properties)

        elif tool_name == "get_location":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = self._get_thing_uri_from_graph(graph)
            if not thing_uri:
                return f"Could not find Thing URI in graph for {artifact_name}"
            location = get_location_info(graph, thing_uri)
            return json.dumps(location)

        elif tool_name == "done_discovering":
            # Build the capability model from all graphs fetched during discovery
            self._build_capability_model(capability_model)
            return "Discovery complete. CapabilityModel built."

        return "Unknown tool"

    def _build_capability_model(self, capability_model: CapabilityModel) -> None:
        """
        Populate the capability model from every RDF graph fetched during discovery.

        For each cached artifact graph this extracts the Thing URI, its action and
        property affordances, and its location, and records them on the model. It is
        idempotent (re-adding an artifact simply overwrites it), so it can be called
        both when the model signals done_discovering and as a fallback when the
        discovery loop ends without that signal. Mutates `capability_model` in place.

        Args:
            capability_model: The model to populate with discovered artifacts.

        Returns:
            None. `capability_model.artifacts` is updated in place.
        """
        for artifact_name, graph in self.graphs.items():
            thing_uri = None
            # Try to find the Thing URI from the graph's subjects
            for subject in graph.subjects():
                subject_str = str(subject)
                if "#this" in subject_str:
                    thing_uri = subject_str
                    break

            if not thing_uri:
                continue

            # Create artifact
            artifact = Artifact(
                name=artifact_name,
                kg_uri=thing_uri,
            )

            # Add actions
            actions_data = list_action_affordances(graph, thing_uri)
            for action_data in actions_data:
                affordance = Affordance(
                    name=action_data.get("name", ""),
                    endpoint_url=action_data.get("endpoint_url", ""),
                    semantic_type=action_data.get("semantic_type", ""),
                    op_type="invokeaction",
                    schema=action_data.get("input_schema", {}),
                )
                artifact.actions.append(affordance)

            # Add properties
            properties_data = list_property_affordances(graph, thing_uri)
            for prop_data in properties_data:
                affordance = Affordance(
                    name=prop_data.get("name", ""),
                    endpoint_url=prop_data.get("endpoint_url", ""),
                    semantic_type=prop_data.get("semantic_type", ""),
                    op_type=prop_data.get("op_type", "readproperty"),
                    schema=prop_data.get("schema", {}),
                )
                artifact.properties.append(affordance)

            # Add location info
            location = get_location_info(graph, thing_uri)
            artifact.location = location

            capability_model.artifacts[artifact_name] = artifact


def parse_goal_artifacts(goal: str) -> list[str]:
    """
    Parse artifact names from goal string.

    Extracts the robot name (first argument) from predicates like:
        !carry("APAS", "DX10_output", "XY10_input")

    Also maps location aliases to their artifact names:
        - DX10_output → DX10
        - DX10_input → DX10
        - XY10_output → XY10
        - XY10_input → XY10

    Args:
        goal: The goal predicate instance (e.g., '!carry("APAS", "DX10_output", "XY10_input")')

    Returns:
        List of artifact names to discover affordances for
    """
    # Extract all quoted strings from the goal
    quoted_strings = re.findall(r"['\"]([^'\"]+)['\"]", goal)

    if not quoted_strings:
        return []

    artifacts = set()

    # First argument is always the artifact name (robot/executor)
    if quoted_strings:
        artifacts.add(quoted_strings[0])

    # Map location names to artifact names (e.g., DX10_output → DX10)
    location_to_artifact = {
        "DX10_output": "DX10",
        "DX10_input": "DX10",
        "XY10_output": "XY10",
        "XY10_input": "XY10",
        "VL10_output": "VL10",
        "VL10_input": "VL10",
    }

    for loc in quoted_strings[1:]:  # Remaining arguments are locations
        if loc in location_to_artifact:
            artifacts.add(location_to_artifact[loc])

    return sorted(list(artifacts))
