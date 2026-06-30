import logging
from fastapi import FastAPI, HTTPException
from src.models import GoalRequest, GoalResponse
from src.agent import AgentLifecycle

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
RESET = "\033[0m"

app = FastAPI(
    title="Industrial Manufacturing BT Planner",
    description="LLM-based agent for discovering affordances and planning behavior trees",
)


@app.post("/solve", response_model=GoalResponse)
async def solve_goal(request: GoalRequest):
    """
    Solve an achievement goal by:
    1. Discovering affordances from RDF knowledge graphs
    2. Generating a BehaviorTree plan
    3. Optionally executing the plan

    Logs all major lifecycle steps at DEBUG level and all errors with a yellow header.
    """
    logger.debug(f"Received goal request: {request.goal}, execute={request.execute}")

    try:
        agent = AgentLifecycle()
        response = agent.solve(request.goal, request.execute)
        logger.debug(f"Goal solved successfully for: {request.goal}")
        return response
    except Exception as e:
        error_msg = f"{YELLOW}[ERROR]{RESET} Failed to solve goal '{request.goal}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
