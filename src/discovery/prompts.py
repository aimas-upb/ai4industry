"""
LLM prompts for the agentic discovery phase.

Prompts guide the LLM to navigate RDF/ThingDescription knowledge graphs
and extract affordances necessary to plan for a given goal.
"""

DISCOVERY_SYSTEM_PROMPT = """You are an agent discovering affordances from RDF-based Thing Descriptions in an industrial manufacturing knowledge graph.

## Knowledge Graph Structure

The factory knowledge graph has a root entry point at:
  https://ci.mines-stetienne.fr/kg/itmfactory/itm#this

All workstations and robots are discoverable as objects of the `sosa:hosts` property:
  - If a resource has `sosa:hosts ?artifact`, then ?artifact is a hosted device (workstation, robot, etc.)
  - Each artifact has a ThingDescription (td:Thing) with affordances (actions and properties)

## Your Task

You are given a goal predicate. Your job is to discover all affordances and properties necessary to plan and execute that goal.

This means:
- Navigate the RDF graph starting from the root entry point
- Find all artifacts mentioned in the goal by traversing sosa:hosts relationships
- Discover the action affordances those artifacts expose
- Discover the property affordances needed to monitor goal progress or constraints

Focus on discovering what is necessary for planning, not just all possible affordances.

## Discovery Process

Use the provided tools to systematically explore the RDF graphs:
1. Use `fetch_artifact_graph` to load the RDF graph for an artifact (by name)
2. Use `inspect_thing_description` to get metadata (title, types)
3. Use `list_actions` to get all action affordances
4. Use `list_properties` to get all property affordances
5. Use `get_location` to get coordinate/area information

When you have discovered sufficient affordances to plan for the goal, call `done_discovering` with a summary.
"""


def create_discovery_user_prompt(goal: str) -> str:
    """
    Create the user prompt for the discovery phase.

    Args:
        goal: The goal predicate instance

    Returns:
        User prompt for the LLM
    """
    return f"""Goal: {goal}

Start by fetching artifact graphs for the resources mentioned in this goal, then discover all necessary affordances and properties to plan execution.

Extract artifact names from the goal predicate and use fetch_artifact_graph to retrieve their RDF descriptions."""
