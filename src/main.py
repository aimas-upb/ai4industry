import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
from src.utils.models import GoalRequest, GoalResponse
from src.agent import AgentLifecycle

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
RESET = "\033[0m"

app = FastAPI(
    title="Industrial Manufacturing BT Planner",
    description="LLM-based agent for discovering affordances and planning behavior trees",
)


@app.post("/solve")
async def solve_goal(request: GoalRequest, background_tasks: BackgroundTasks):
    """
    Solve an achievement goal by:
    1. Discovering affordances from RDF knowledge graphs
    2. Generating a BehaviorTree plan
    3. Optionally executing the plan

    If callback_url is provided, solve runs asynchronously and result is PUT to that URL.
    Otherwise, solve runs synchronously and response is returned directly.

    Logs all major lifecycle steps at DEBUG level and all errors with a yellow header.
    """
    logger.debug(f"Received goal request: {request.goal}, execute={request.execute}, callback_url={request.callback_url}")

    try:
        if request.callback_url:
            # Async mode: launch in background, return 202 immediately
            background_tasks.add_task(
                _solve_and_callback,
                request.goal,
                request.execute,
                request.schema,
                request.callback_url
            )
            return {"status": "accepted"}

        else:
            # Sync mode: solve directly and return response (backwards compatible)
            agent = AgentLifecycle()
            response = agent.solve(request.goal, request.execute, goal_schema=request.schema)
            logger.debug(f"Goal solved successfully for: {request.goal}")
            return response

    except Exception as e:
        error_msg = f"{YELLOW}[ERROR]{RESET} Failed to solve goal '{request.goal}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


async def _solve_and_callback(goal: str, execute: bool, schema: str, callback_url: str):
    """
    Solve a goal asynchronously and PUT the result to callback_url.

    On exception, PUT a failure result to callback_url so the requester isn't left hanging.
    """
    try:
        agent = AgentLifecycle()
        response = agent.solve(goal, execute, goal_schema=schema)
        logger.debug(f"Goal solved successfully for: {goal}, sending callback to {callback_url}")

        # Convert response to dict for callback
        callback_payload = response.model_dump()
        async with httpx.AsyncClient() as client:
            await client.put(callback_url, json=callback_payload, timeout=10)
            logger.debug(f"Callback PUT succeeded for {goal}")

    except Exception as e:
        error_msg = f"{YELLOW}[ERROR]{RESET} Async solve/callback failed for '{goal}': {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Attempt to notify callback of failure
        try:
            failure_payload = {
                "goal": goal,
                "capability_summary": None,
                "bt_plan": None,
                "execution_result": {"status": "FAILURE", "ticks": 0, "trace": [], "error": str(e)}
            }
            async with httpx.AsyncClient() as client:
                await client.put(callback_url, json=failure_payload, timeout=10)
                logger.debug(f"Failure callback PUT succeeded for {goal}")
        except Exception as callback_error:
            logger.error(f"Failed to send failure callback: {callback_error}")


@app.get("/health")
async def health():
    return {"status": "ok"}
