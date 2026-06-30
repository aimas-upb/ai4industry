import logging
import json
from src.discovery.agentic_discovery import AgenticDiscovery, parse_goal_artifacts
from src.planning.bt_planner import BTPlanner
from src.planning.ir_executor import IRExecutor
from src.models import GoalResponse

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
RESET = "\033[0m"


class AgentLifecycle:
    """
    Orchestrate the agent lifecycle: discovery -> planning -> optional execution.

    Logs all major steps at DEBUG level and errors with a yellow header.
    """

    def solve(self, goal: str, execute: bool = False) -> GoalResponse:
        """
        Solve an achievement goal through the complete agent lifecycle.

        The goal follows a predicate schema (e.g., !carry(RobotName, FromLocation, ToLocation))
        with a concrete instance (e.g., !carry("APAS", "DX10_output", "XY10_input")).

        Args:
            goal: The goal predicate instance (e.g., "!carry('APAS', 'DX10_output', 'XY10_input')")
            execute: If True, execute the generated BehaviorTree; otherwise just plan

        Returns:
            GoalResponse containing the capability summary, BT plan, and optional execution result

        Raises:
            ValueError: If goal cannot be parsed to extract artifact names
            Exception: If discovery, planning, or execution steps fail
        """
        logger.debug(f"Starting agent lifecycle for goal: {goal}")

        try:
            # Step 1: Parse artifact names from goal predicate instance
            logger.debug(f"Parsing artifact names from goal: {goal}")
            artifact_names = parse_goal_artifacts(goal)
            if not artifact_names:
                error_msg = f"{YELLOW}[ERROR]{RESET} Could not parse artifact names from goal: {goal}"
                logger.error(error_msg)
                raise ValueError(f"Could not parse artifact names from goal: {goal}")
            logger.debug(f"Parsed artifacts: {artifact_names}")

            # Step 2: Run agentic discovery
            logger.debug("Starting agentic discovery phase")
            discovery = AgenticDiscovery()
            capability_model = discovery.discover(goal, artifact_names)
            logger.debug(f"Discovery phase complete. Found {len(capability_model.artifacts)} artifacts")
            for artifact_name, artifact in capability_model.artifacts.items():
                logger.debug(f"  - {artifact_name}: {len(artifact.actions)} actions, {len(artifact.properties)} properties")

            # Step 3: Generate BehaviorTree plan
            logger.debug("Starting BehaviorTree planning phase")
            planner = BTPlanner(capability_model, goal=goal)
            bt_plan = planner.plan(goal)
            logger.debug(f"BehaviorTree plan generated successfully")
            logger.debug(f"BT plan structure: {json.dumps(bt_plan, indent=2)[:500]}...")  # Log first 500 chars

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

            logger.debug(f"Agent lifecycle complete for goal: {goal}")
            return GoalResponse(
                goal=goal,
                capability_summary=capability_model.to_prompt_context(),
                bt_plan=bt_plan,
                execution_result=execution_result,
            )

        except Exception as e:
            error_msg = f"{YELLOW}[ERROR]{RESET} Agent lifecycle failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
