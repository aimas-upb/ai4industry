from pydantic import BaseModel
from typing import Optional


class GoalRequest(BaseModel):
    """
    Request to solve an achievement goal.

    Attributes:
        goal: The goal predicate instance (e.g., '!carry("APAS", "DX10_output", "XY10_input")')
        execute: If True, execute the generated BehaviorTree; if False, only plan
    """
    goal: str
    execute: bool = False


class GoalResponse(BaseModel):
    """
    Response containing discovery results, BT plan, and optional execution trace.

    Attributes:
        goal: The original goal predicate instance
        capability_summary: Markdown-formatted summary of discovered affordances
        bt_plan: BehaviorTree JSON-IR structure
        execution_result: Optional dict with execution status, ticks, and trace (only if execute=True)
    """
    goal: str
    capability_summary: str
    bt_plan: dict
    execution_result: Optional[dict] = None
