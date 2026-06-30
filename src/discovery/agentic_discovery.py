import json
import re
import logging
from typing import Optional

import httpx
from openai import OpenAI, AzureOpenAI

from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_API_KEY,
    MISTRAL_API_KEY,
    MISTRAL_BASE_URL,
    MAX_DISCOVERY_ITERATIONS,
    ARTIFACT_REGISTRY,
)
from src.discovery.capability_model import (
    CapabilityModel,
    Artifact,
    Affordance,
)
from src.discovery.rdf_tools import (
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
RESET = "\033[0m"


class AgenticDiscovery:
    def __init__(self):
        if LLM_PROVIDER == "mistral":
            self.client = OpenAI(
                api_key=MISTRAL_API_KEY,
                base_url=MISTRAL_BASE_URL,
            )
        else:
            self.client = OpenAI(api_key=LLM_API_KEY)

        self.discovery_tools = [
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
        ]

        # Cache for fetched graphs
        self.graphs = {}

    def discover(self, goal: str, artifact_names: list[str]) -> CapabilityModel:
        """
        Run agentic discovery loop to discover affordances from RDF graphs.

        The LLM uses tool calls to navigate RDF graphs and extract action/property affordances
        for each artifact, building up a CapabilityModel that maps artifact names to their
        available actions and properties.

        Args:
            goal: The achievement goal to discover affordances for
            artifact_names: List of artifact names to discover

        Returns:
            CapabilityModel containing all discovered affordances and initial state
        """
        logger.debug(f"Starting agentic discovery for artifacts: {artifact_names}")
        capability_model = CapabilityModel(goal=goal)

        # Build messages with system prompt from prompts.py and user prompt
        system_prompt = DISCOVERY_SYSTEM_PROMPT
        user_prompt = create_discovery_user_prompt(goal, artifact_names)

        messages = [{"role": "user", "content": user_prompt}]

        for iteration in range(MAX_DISCOVERY_ITERATIONS):
            logger.debug(f"Discovery iteration {iteration + 1}/{MAX_DISCOVERY_ITERATIONS}")
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                system=system_prompt,
                messages=messages,
                tools=self.discovery_tools,
                tool_choice="auto",
            )

            # Check if we're done
            if response.stop_reason == "stop":
                break

            # Process tool calls
            assistant_message = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]

            messages.append(assistant_message)

            # Execute tool calls and collect results
            tool_results = []
            for tool_call in response.tool_calls:
                logger.debug(f"Executing tool: {tool_call.function.name}")
                try:
                    result = self._execute_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                        capability_model,
                        artifact_names,
                    )
                    logger.debug(f"Tool {tool_call.function.name} completed successfully")
                except Exception as e:
                    error_msg = f"{YELLOW}[ERROR]{RESET} Tool execution failed for {tool_call.function.name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise

                tool_results.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                        "id": tool_call.id,
                        "result": result,
                    }
                )

                # If done_discovering, stop the loop
                if tool_call.function.name == "done_discovering":
                    logger.debug(f"Discovery phase complete. Found {len(capability_model.artifacts)} artifacts")
                    return capability_model

            # Add tool results to messages
            if tool_results:
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(tool_results),
                    }
                )

        return capability_model

    def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        capability_model: CapabilityModel,
        artifact_names: list[str],
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
            thing_uri = ARTIFACT_REGISTRY.get(artifact_name)
            metadata = get_thing_description(graph, thing_uri)
            return json.dumps(metadata)

        elif tool_name == "list_actions":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = ARTIFACT_REGISTRY.get(artifact_name)
            actions = list_action_affordances(graph, thing_uri)
            return json.dumps(actions)

        elif tool_name == "list_properties":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = ARTIFACT_REGISTRY.get(artifact_name)
            properties = list_property_affordances(graph, thing_uri)
            return json.dumps(properties)

        elif tool_name == "get_location":
            artifact_name = args.get("artifact_name")
            if artifact_name not in self.graphs:
                return f"Graph not loaded for {artifact_name}. Call fetch_artifact_graph first."

            graph = self.graphs[artifact_name]
            thing_uri = ARTIFACT_REGISTRY.get(artifact_name)
            location = get_location_info(graph, thing_uri)
            return json.dumps(location)

        elif tool_name == "done_discovering":
            # Build capability model from collected graphs
            for artifact_name in artifact_names:
                if artifact_name not in self.graphs:
                    continue

                graph = self.graphs[artifact_name]
                thing_uri = ARTIFACT_REGISTRY.get(artifact_name)

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

            return "Discovery complete. CapabilityModel built."

        return "Unknown tool"


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
