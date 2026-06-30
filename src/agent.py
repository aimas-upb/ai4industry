import logging
import json
from src.discovery.agentic_discovery import AgenticDiscovery, parse_goal_artifacts
from src.planning.bt_planner import BTPlanner
from src.planning.ir_executor import IRExecutor
from src.models import GoalResponse, GoalSpecification
from src.config import LLM_PROVIDER, LLM_MODEL, get_llm_params

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
RESET = "\033[0m"


class AgentLifecycle:
    """
    Orchestrate the agent lifecycle: discovery -> planning -> optional execution.

    Logs all major steps at DEBUG level and errors with a yellow header.
    """

    def solve(self, goal_instance: str, execute: bool = False, goal_schema: str = "") -> GoalResponse:
        """
        Solve an achievement goal through the complete agent lifecycle.

        The goal follows a predicate schema (e.g., !carry(RobotName, FromLocation, ToLocation))
        with a concrete instance (e.g., !carry("APAS", "DX10_output", "XY10_input")).

        Args:
            goal_instance: The goal predicate instance (e.g., "!carry('APAS', 'DX10_output', 'XY10_input')")
            execute: If True, execute the generated BehaviorTree; otherwise just plan
            goal_schema: Optional goal predicate schema for context

        Returns:
            GoalResponse containing the capability summary, BT plan, and optional execution result

        Raises:
            ValueError: If goal cannot be parsed to extract artifact names
            Exception: If discovery, planning, or execution steps fail
        """
        # Log model configuration
        llm_params = get_llm_params()
        logger.info(f"LLM Config: provider={LLM_PROVIDER}, model={LLM_MODEL}, params={llm_params}")

        # Create goal specification with schema only if provided
        goal_spec = GoalSpecification(schema=goal_schema or "", instance=goal_instance)
        logger.debug(f"Starting agent lifecycle for goal: {goal_instance}")

        try:
            # Step 1: Parse artifact names from goal predicate instance
            logger.debug(f"Parsing artifact names from goal: {goal_instance}")
            artifact_names = parse_goal_artifacts(goal_instance)
            if not artifact_names:
                error_msg = f"{YELLOW}[ERROR]{RESET} Could not parse artifact names from goal: {goal_instance}"
                logger.error(error_msg)
                raise ValueError(f"Could not parse artifact names from goal: {goal_instance}")
            logger.debug(f"Parsed artifacts: {artifact_names}")

            # Step 2: Run agentic discovery
            logger.debug("Starting agentic discovery phase")
            discovery = AgenticDiscovery()
            capability_model = discovery.discover(goal_instance, artifact_names)
            logger.debug(f"Discovery phase complete. Found {len(capability_model.artifacts)} artifacts")
            for artifact_name, artifact in capability_model.artifacts.items():
                logger.debug(f"  - {artifact_name}: {len(artifact.actions)} actions, {len(artifact.properties)} properties")

            # Step 3: Generate BehaviorTree plan
            logger.debug("Starting BehaviorTree planning phase")
            planner = BTPlanner(capability_model, goal_spec=goal_spec)
            bt_plan = planner.plan(goal_spec)
            logger.debug(f"BehaviorTree plan generated successfully")
            # Log the full BT plan pretty-printed
            bt_plan_str = json.dumps(bt_plan, indent=2)
            logger.debug(f"BT plan:\n{bt_plan_str}")

            # Step 4: Optionally execute the plan
            execution_result = None
            if execute:
                logger.debug("Starting BehaviorTree execution phase")
                try:
                    executor = IRExecutor()
                    tree = executor.compile(bt_plan)
                    logger.debug("BT compiled successfully, beginning execution")
                    execution_result = executor.execute(tree)
                    logger.info(f"BT execution result: {execution_result['status']} ({execution_result['ticks']} ticks)")
                    # Log execution trace for debugging
                    for trace_entry in execution_result['trace'][:10]:  # Log first 10 entries
                        logger.debug(f"  [{trace_entry['tick']}] {trace_entry['node']}: {trace_entry['status']}")
                except Exception as e:
                    error_msg = f"{YELLOW}[ERROR]{RESET} BT execution failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise
            else:
                logger.debug("Skipping execution phase (execute=False)")

            logger.debug(f"Agent lifecycle complete for goal: {goal_instance}")
            return GoalResponse(
                goal=goal_instance,
                capability_summary=capability_model.to_prompt_context(),
                bt_plan=bt_plan,
                execution_result=execution_result,
            )

        except Exception as e:
            error_msg = f"{YELLOW}[ERROR]{RESET} Agent lifecycle failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    @staticmethod
    def _infer_goal_schema(goal_instance: str) -> str:
        """
        Infer the goal predicate schema from a goal instance.

        Converts a concrete instance like !carry("APAS", "DX10_output", "XY10_input")
        to a schema like !carry(RobotName, FromLocation, ToLocation).

        Args:
            goal_instance: The concrete goal instance

        Returns:
            The inferred goal schema
        """
        # Extract predicate name and argument count
        import re
        match = re.match(r"(!?\w+)\((.*)\)", goal_instance)
        if not match:
            return goal_instance

        predicate = match.group(1)
        args_str = match.group(2)

        # Count quoted arguments
        arg_count = len(re.findall(r'"[^"]*"', args_str))

        if arg_count == 0:
            return goal_instance

        # Generate generic parameter names based on count
        param_names = ["Param1", "Param2", "Param3", "Param4", "Param5"]
        params = ", ".join(param_names[:arg_count])

        return f"{predicate}({params})"
