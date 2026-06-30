import py_trees
import logging
from typing import Optional

from src.planning.bt_nodes import ActionAffordanceNode, PropertyConditionNode

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
RESET = "\033[0m"


class IRExecutor:
    """
    Compile and execute BT JSON-IR with detailed tracing.

    Tracks node execution order, status transitions, and execution details
    for debugging and result reporting.
    """

    def __init__(self, max_ticks: int = 100):
        """
        Initialize the executor.

        Args:
            max_ticks: Maximum number of tree ticks before timeout
        """
        self.max_ticks = max_ticks
        self.execution_trace = []  # List of (node_name, status, details)

    def compile(self, ir: dict) -> py_trees.behaviour.Behaviour:
        """
        Recursively convert JSON-IR to py_trees behavior tree objects.

        Args:
            ir: Dictionary representing a node in the BT JSON-IR format

        Returns:
            py_trees Behaviour object (Composite or Leaf)

        Raises:
            ValueError: If the node type is unknown or required fields are missing
        """
        node_type = ir.get("type")
        name = ir.get("name", "UnnamedNode")

        if node_type == "sequence":
            children = [self.compile(child) for child in ir.get("children", [])]
            return py_trees.composites.Sequence(name=name, memory=False, children=children)

        elif node_type == "selector":
            children = [self.compile(child) for child in ir.get("children", [])]
            return py_trees.composites.Selector(name=name, memory=False, children=children)

        elif node_type == "parallel":
            children = [self.compile(child) for child in ir.get("children", [])]
            policy_str = ir.get("policy", "success_on_all")

            if policy_str == "success_on_one":
                policy = py_trees.common.ParallelPolicy.SuccessOnOne()
            else:
                policy = py_trees.common.ParallelPolicy.SuccessOnAll()

            return py_trees.composites.Parallel(
                name=name,
                policy=policy,
                children=children,
            )

        elif node_type == "action":
            url = ir.get("action_url")
            parameters = ir.get("parameters", {})
            return ActionAffordanceNode(name=name, url=url, parameters=parameters)

        elif node_type == "condition":
            url = ir.get("property_url")
            expected = ir.get("expected_value")
            operator = ir.get("operator", "==")
            return PropertyConditionNode(
                name=name,
                url=url,
                expected_value=expected,
                operator=operator,
            )

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def execute(self, tree: py_trees.behaviour.Behaviour) -> dict:
        """
        Execute the behavior tree with detailed tracing.

        Ticks the tree up to max_ticks times, recording execution trace at each step.
        Trace includes which nodes were visited, their status, and execution details.

        Args:
            tree: The compiled py_trees root node to execute

        Returns:
            Dictionary with:
              - status: "SUCCESS", "FAILURE", or "TIMEOUT"
              - ticks: Number of ticks executed
              - trace: List of (node_name, status_string, details) tuples
        """
        logger.debug("Starting BT execution")
        self.execution_trace = []
        tree.setup_with_descendants()

        for tick_num in range(self.max_ticks):
            logger.debug(f"BT tick {tick_num + 1}/{self.max_ticks}")

            # Tick the tree
            tree.tick_once()
            status = tree.status

            # Collect trace data for all visited nodes this tick
            self._collect_trace(tree, tick_num)

            if status == py_trees.common.Status.SUCCESS:
                logger.info("BT execution succeeded")
                return {
                    "status": "SUCCESS",
                    "ticks": tick_num + 1,
                    "trace": self.execution_trace,
                }
            elif status == py_trees.common.Status.FAILURE:
                logger.warning("BT execution failed")
                return {
                    "status": "FAILURE",
                    "ticks": tick_num + 1,
                    "trace": self.execution_trace,
                }

        logger.error(f"{YELLOW}[ERROR]{RESET} BT execution timed out after {self.max_ticks} ticks")
        return {
            "status": "TIMEOUT",
            "ticks": self.max_ticks,
            "trace": self.execution_trace,
        }

    def _collect_trace(self, node: py_trees.behaviour.Behaviour, tick_num: int) -> None:
        """
        Recursively collect execution trace from all visited nodes.

        Records node name, status, and any relevant details (e.g., action URL, condition result).

        Args:
            node: The current node to trace
            tick_num: The current tick number
        """
        status_str = str(node.status).split(".")[-1]  # e.g., "SUCCESS", "FAILURE", "RUNNING"

        # Extract details based on node type
        details = ""
        if isinstance(node, ActionAffordanceNode):
            details = f"Action: {node.url}"
        elif isinstance(node, PropertyConditionNode):
            details = f"Condition: {node.url} {node.operator} {node.expected_value}"

        self.execution_trace.append({
            "tick": tick_num,
            "node": node.name,
            "type": type(node).__name__,
            "status": status_str,
            "details": details,
        })

        # Recursively trace children for composite nodes
        if hasattr(node, "children"):
            for child in node.children:
                self._collect_trace(child, tick_num)
