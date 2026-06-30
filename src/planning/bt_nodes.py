import httpx
import py_trees
from typing import Any, Optional, List
from src.config import SIMULATOR_USERNAME, SIMULATOR_PASSWORD


class PropertyAffordanceNode(py_trees.behaviour.Behaviour):
    """
    Read a property via HTTP GET and store the value on the blackboard.

    The value is stored under a key extracted from the URL path, namespaced by
    artifact name to avoid conflicts across multiple artifacts.
    """

    def __init__(self, name: str, property_url: str, result_key: Optional[str] = None):
        """
        Initialize the property affordance node.

        Args:
            name: The name of this behavior tree node
            property_url: HTTP endpoint URL for the property affordance
            result_key: Optional custom blackboard key (auto-extracted from URL if not provided)
        """
        super().__init__(name=name)
        self.property_url = property_url
        # Extract artifact name and property name from URL if not provided
        if result_key is None:
            artifact_name, property_name = self._extract_names(property_url)
            self.result_key = f"artifacts/{artifact_name}/{property_name}" if artifact_name and property_name else f"properties/{name}"
        else:
            self.result_key = result_key
        self.last_value = None
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key=self.result_key, access=py_trees.common.Access.WRITE)

    @staticmethod
    def _extract_names(url: str) -> tuple:
        """
        Extract artifact and property names from a URL path.

        Parses URLs of the form '.../artifactName/properties/propertyName' to extract
        the artifact identifier and property name for namespacing blackboard keys.

        Args:
            url: The full property endpoint URL

        Returns:
            Tuple of (artifact_name, property_name) or (None, None) if URL format is invalid
        """
        if "/properties/" in url:
            parts = url.split("/properties/")
            property_name = parts[-1].split("?")[0].split("#")[0]
            # Extract artifact name (usually the part before /properties/)
            artifact_parts = parts[0].split("/")
            artifact_name = artifact_parts[-1] if artifact_parts else None
            return artifact_name, property_name
        return None, None

    def update(self) -> py_trees.common.Status:
        """
        Read the property value via HTTP GET and store on blackboard.

        Returns:
            Status.SUCCESS if property was read and stored successfully
            Status.FAILURE if the HTTP request failed or property is unreachable
        """
        try:
            response = httpx.get(
                self.property_url,
                auth=(SIMULATOR_USERNAME, SIMULATOR_PASSWORD),
                timeout=10
            )
            response.raise_for_status()
            self.last_value = response.json()
            self.blackboard.set(self.result_key, self.last_value)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.logger.error(f"Property read failed: {e}")
            return py_trees.common.Status.FAILURE


class ActionAffordanceNode(py_trees.behaviour.Behaviour):
    """
    Execute an action affordance via HTTP POST.

    Invokes a Thing Description action endpoint with the provided parameters
    and returns SUCCESS or FAILURE based on the HTTP response.
    """

    def __init__(self, name: str, url: str, parameters: Optional[dict] = None):
        """
        Initialize the action affordance node.

        Args:
            name: The name of this behavior tree node
            url: HTTP endpoint URL for the action affordance
            parameters: Dictionary of parameters to send in the POST request
        """
        super().__init__(name=name)
        self.url = url
        self.parameters = parameters or {}

    def update(self) -> py_trees.common.Status:
        """
        Execute the action affordance via HTTP POST.

        Returns:
            Status.SUCCESS if the POST request succeeded (2xx status code)
            Status.FAILURE if the request failed or endpoint is unreachable
        """
        try:
            response = httpx.post(
                self.url,
                json=self.parameters,
                auth=(SIMULATOR_USERNAME, SIMULATOR_PASSWORD),
                timeout=10,
            )
            response.raise_for_status()
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.logger.error(f"Action failed: {e}")
            return py_trees.common.Status.FAILURE


class PropertyConditionNode(py_trees.behaviour.Behaviour):
    """
    Read a property value and compare it against an expected value.

    Reads a property via HTTP GET and evaluates whether the actual value
    matches the expected value using the specified comparison operator.
    Used as a guard/precondition in behavior tree branches.
    """

    def __init__(
        self,
        name: str,
        url: str,
        expected_value: Any,
        operator: str = "==",
        value_path: Optional[List[str]] = None,
    ):
        """
        Initialize the property condition node.

        Args:
            name: The name of this behavior tree node
            url: HTTP endpoint URL for the property affordance
            expected_value: The value to compare the property against
            operator: Comparison operator (==, !=, >, <, >=, <=, in, not_in, contains, matches)
            value_path: Optional list of keys to navigate nested response objects
        """
        super().__init__(name=name)
        self.url = url
        self.expected_value = expected_value
        self.operator = operator
        self.value_path = value_path or []

    def update(self) -> py_trees.common.Status:
        """
        Read the property and compare against expected value.

        Returns:
            Status.SUCCESS if the comparison matches (or doesn't match if negate=True)
            Status.FAILURE if property cannot be read or comparison fails
        """
        try:
            response = httpx.get(
                self.url,
                auth=(SIMULATOR_USERNAME, SIMULATOR_PASSWORD),
                timeout=10
            )
            response.raise_for_status()
            actual_value = response.json()

            # Navigate to nested value if path specified (e.g., for state.hand in complex responses)
            for key in self.value_path:
                if isinstance(actual_value, dict):
                    actual_value = actual_value.get(key)
                else:
                    return py_trees.common.Status.FAILURE

            if self._compare(actual_value, self.expected_value):
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        except Exception as e:
            self.logger.error(f"Condition check failed: {e}")
            return py_trees.common.Status.FAILURE

    def _compare(self, actual: Any, expected: Any) -> bool:
        """
        Compare two values using the configured operator.

        Supports standard comparison operators (==, !=, >, <, etc.) and special
        operators for lists (in, not_in, contains) and regex matching.
        Also handles deep dictionary comparison for complex property values.

        Args:
            actual: The actual value read from the property
            expected: The expected value to compare against

        Returns:
            True if the comparison is satisfied, False otherwise
        """
        if self.operator == "==":
            if isinstance(actual, dict) and isinstance(expected, dict):
                return self._dicts_equal(actual, expected)
            return actual == expected
        elif self.operator == "!=":
            if isinstance(actual, dict) and isinstance(expected, dict):
                return not self._dicts_equal(actual, expected)
            return actual != expected
        elif self.operator == ">":
            return actual > expected
        elif self.operator == "<":
            return actual < expected
        elif self.operator == ">=":
            return actual >= expected
        elif self.operator == "<=":
            return actual <= expected
        elif self.operator == "in":
            return actual in expected
        elif self.operator == "not_in":
            return actual not in expected
        elif self.operator == "contains":
            return expected in actual
        elif self.operator == "matches":
            import re
            return bool(re.match(expected, str(actual)))
        else:
            return False

    def _dicts_equal(self, dict1: dict, dict2: dict) -> bool:
        """
        Compare two dictionaries for deep equality.

        Checks that both dictionaries have the same keys and that all corresponding
        values are equal. Used for comparing complex property values like coordinates.

        Args:
            dict1: First dictionary to compare
            dict2: Second dictionary to compare

        Returns:
            True if dictionaries have identical keys and values, False otherwise
        """
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        for key in dict1:
            if dict1[key] != dict2[key]:
                return False
        return True
