from pydantic import BaseModel
from typing import Optional


class GoalSpecification:
    """
    Encapsulates a goal specification with schema and instance.

    Attributes:
        schema: The goal predicate schema (e.g., '!carry(RobotName, FromLocation, ToLocation)')
        instance: The goal predicate instance (e.g., '!carry("APAS", "DX10_output", "XY10_input")')
    """
    def __init__(self, schema: str, instance: str):
        """
        Initialize a goal specification.

        Args:
            schema: The goal schema template
            instance: The concrete goal instance
        """
        self.schema = schema
        self.instance = instance

    def __str__(self) -> str:
        return self.instance


class GoalRequest(BaseModel):
    """
    Request to solve an achievement goal.

    Attributes:
        goal: The goal predicate instance (e.g., '!carry("APAS", "DX10_output", "XY10_input")')
        schema: Optional goal predicate schema (e.g., '!carry(RobotName, FromLocation, ToLocation)'). If not provided, will be inferred.
        execute: If True, execute the generated BehaviorTree; if False, only plan
    """
    goal: str
    schema: Optional[str] = None
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
