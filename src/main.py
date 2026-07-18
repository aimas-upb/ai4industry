import logging
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
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
    start_time = time.time()
    logger.debug(f"Received goal request: {request.goal}, execute={request.execute}, callback_url={request.callback_url}")

    try:
        if request.callback_url:
            # Async mode: launch in background, return 202 immediately
            logger.debug(f"Launching async solve for goal: {request.goal}, callback_url={request.callback_url}")
            background_tasks.add_task(
                _solve_and_callback,
                request.goal,
                request.execute,
                request.schema,
                request.callback_url
            )
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Background task queued, returning 202 in {elapsed:.1f}ms")
            # Return immediately without waiting for background task
            return JSONResponse({"status": "accepted"}, status_code=202)

        else:
            # Sync mode: solve directly and return response (backwards compatible)
            logger.debug(f"Launching synchronous solve for goal: {request.goal}")
            agent = AgentLifecycle()
            response = agent.solve(request.goal, request.execute, goal_schema=request.schema)
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Goal solved successfully for: {request.goal} in {elapsed:.1f}ms")
            return JSONResponse(content=response.model_dump(), status_code=200)

    except Exception as e:
        error_msg = f"{YELLOW}[ERROR]{RESET} Failed to solve goal '{request.goal}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


def _solve_and_callback(goal: str, execute: bool, schema: str, callback_url: str):
    """
    Solve a goal and PUT the result to callback_url.

    This is intentionally a SYNCHRONOUS function. agent.solve() is a blocking call
    (RDF discovery, LLM requests, tree execution). FastAPI runs sync background tasks
    in a threadpool, which keeps the event loop free to flush the 202 response to the
    caller immediately. If this were `async def`, the blocking solve() would run on the
    event loop and starve it, so the 202 would not reach the caller until the entire
    solve finished.

    On exception, PUT a failure result to callback_url so the requester isn't left hanging.
    """
    try:
        agent = AgentLifecycle()
        response = agent.solve(goal, execute, goal_schema=schema)
        logger.debug(f"Goal solved successfully for: {goal}, sending callback to {callback_url}")

        # Convert response to dict for callback (synchronous httpx client)
        callback_payload = response.model_dump()
        with httpx.Client() as client:
            client.put(callback_url, json=callback_payload, timeout=10)
            logger.debug(f"Callback PUT succeeded for {goal}")

    except Exception as e:
        error_msg = f"{YELLOW}[ERROR]{RESET} Solve/callback failed for '{goal}': {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Attempt to notify callback of failure
        try:
            failure_payload = {
                "goal": goal,
                "capability_summary": None,
                "bt_plan": None,
                "execution_result": {"status": "FAILURE", "ticks": 0, "trace": [], "error": str(e)}
            }
            with httpx.Client() as client:
                client.put(callback_url, json=failure_payload, timeout=10)
                logger.debug(f"Failure callback PUT succeeded for {goal}")
        except Exception as callback_error:
            logger.error(f"Failed to send failure callback: {callback_error}")


@app.get("/health")
async def health():
    return {"status": "ok"}
