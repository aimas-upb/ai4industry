import logging
import threading
import uuid
import time
from dataclasses import dataclass, asdict
from typing import Optional
import flask
import httpx

# Configuration
SERVER_URL = "http://localhost"
SERVER_PORT = 5565
SOLVE_SERVICE = "solve"
STATUS_SERVICE = "status"
INPUT_DATA_PARAM = "input_data"
AGENT_URL = "http://localhost:8008/solve"
MAX_LIFE_IN_SECONDS = 600
SOLVE_TIMEOUT_SECONDS = 300

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

head = "<ML server> "
def log(*args):
    logger.info(f"{head}: {' '.join(map(str, args))}")

def logE(*args):
    logger.error(f"{head}: {' '.join(map(str, args))}")


@dataclass
class RequestStatus:
    """Represents the status of a solve request."""
    request_uri: str
    goal: Optional[str] = None
    capability_summary: Optional[str] = None
    bt_plan: Optional[dict] = None
    execution_result: Optional[str] = "RUNNING"  # RUNNING | SUCCESS | FAILURE | TIMEOUT | None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RequestManager:
    """
    Manages concurrent solve requests with TTL-based auto-expiry and timeout handling.

    Each request is stored with two timers:
    - TTL timer: auto-deletes the request after MAX_LIFE_IN_SECONDS
    - Solve-timeout timer: marks execution_result as TIMEOUT if no result by SOLVE_TIMEOUT_SECONDS
    """

    def __init__(self):
        self.requests: dict[str, RequestStatus] = {}
        self.ttl_timers: dict[str, threading.Timer] = {}
        self.timeout_timers: dict[str, threading.Timer] = {}
        self.lock = threading.Lock()

    def create_request(self, goal_instance: str) -> RequestStatus:
        """
        Create a new request status resource.

        Generates a unique request ID, builds the resource URI, schedules TTL and timeout timers.

        Args:
            goal_instance: The goal predicate instance

        Returns:
            RequestStatus with request_uri and initial execution_result="RUNNING"
        """
        req_id = uuid.uuid4().hex
        request_uri = f"{SERVER_URL}:{SERVER_PORT}/{STATUS_SERVICE}/{req_id}"

        status = RequestStatus(
            request_uri=request_uri,
            goal=goal_instance,
            capability_summary=None,
            bt_plan=None,
            execution_result="RUNNING"
        )

        with self.lock:
            self.requests[req_id] = status

            # Schedule TTL-based expiry
            ttl_timer = threading.Timer(
                MAX_LIFE_IN_SECONDS,
                self._expire,
                args=[req_id]
            )
            ttl_timer.daemon = True
            ttl_timer.start()
            self.ttl_timers[req_id] = ttl_timer

            # Schedule solve-timeout watchdog
            timeout_timer = threading.Timer(
                SOLVE_TIMEOUT_SECONDS,
                self._timeout,
                args=[req_id]
            )
            timeout_timer.daemon = True
            timeout_timer.start()
            self.timeout_timers[req_id] = timeout_timer

        log(f"Created request {req_id} for goal: {goal_instance}")
        return status

    def get(self, req_id: str) -> Optional[RequestStatus]:
        """Retrieve a request by ID."""
        with self.lock:
            return self.requests.get(req_id)

    def update(self, req_id: str, **fields) -> Optional[RequestStatus]:
        """
        Update a request's fields (partial update).

        Merges only the provided fields into the stored RequestStatus.
        If execution_result reaches a terminal state (SUCCESS/FAILURE/TIMEOUT),
        cancels the solve-timeout timer.

        Args:
            req_id: Request ID
            **fields: Fields to update (subset of goal, capability_summary, bt_plan, execution_result)

        Returns:
            Updated RequestStatus or None if not found
        """
        with self.lock:
            if req_id not in self.requests:
                return None

            status = self.requests[req_id]

            # Update provided fields
            for key, value in fields.items():
                if hasattr(status, key):
                    setattr(status, key, value)

            # Cancel solve-timeout timer if we've reached a terminal state
            if status.execution_result in ("SUCCESS", "FAILURE", "TIMEOUT"):
                if req_id in self.timeout_timers:
                    self.timeout_timers[req_id].cancel()
                    del self.timeout_timers[req_id]
                    log(f"Cancelled timeout timer for {req_id}, status: {status.execution_result}")

            log(f"Updated request {req_id}, execution_result: {status.execution_result}")
            return status

    def delete(self, req_id: str) -> bool:
        """
        Delete a request by ID.

        Cancels both TTL and timeout timers.

        Args:
            req_id: Request ID

        Returns:
            True if request existed and was deleted, False otherwise
        """
        with self.lock:
            if req_id not in self.requests:
                return False

            # Cancel timers
            if req_id in self.ttl_timers:
                self.ttl_timers[req_id].cancel()
                del self.ttl_timers[req_id]
            if req_id in self.timeout_timers:
                self.timeout_timers[req_id].cancel()
                del self.timeout_timers[req_id]

            del self.requests[req_id]
            log(f"Deleted request {req_id}")
            return True

    def _expire(self, req_id: str):
        """Called by TTL timer to auto-delete an expired request."""
        with self.lock:
            if req_id in self.requests:
                del self.requests[req_id]
                # Clean up timeout timer if present
                if req_id in self.timeout_timers:
                    self.timeout_timers[req_id].cancel()
                    del self.timeout_timers[req_id]
                log(f"Expired request {req_id} (TTL)")

    def _timeout(self, req_id: str):
        """Called by solve-timeout timer to mark a request as timed out."""
        with self.lock:
            if req_id in self.requests:
                status = self.requests[req_id]
                # Only mark TIMEOUT if not already in terminal state
                if status.execution_result not in ("SUCCESS", "FAILURE", "TIMEOUT"):
                    status.execution_result = "TIMEOUT"
                    log(f"Marked request {req_id} as TIMEOUT (solve took >300s)")


# Initialize Flask app and request manager
log("Creating server on port", SERVER_PORT, "...")
app = flask.Flask(__name__)
app.config["DEBUG"] = True
manager = RequestManager()


@app.route('/' + SOLVE_SERVICE, methods=['POST'])
def solve():
    """
    Accept a solve request, create a RequestStatus resource, and invoke the solver agent asynchronously.

    Returns 202 Accepted with the resource URI so the caller can poll the status.
    """
    log("Received /solve request")
    try:
        goal_instance = flask.request.form.get(INPUT_DATA_PARAM)
        if not goal_instance:
            logE("No input_data parameter provided")
            return flask.jsonify({'error': 'Missing input_data parameter'}), 400

        # Create a new request status resource
        status = manager.create_request(goal_instance)

        # Invoke the solver agent asynchronously with callback URL
        payload = {
            "goal": goal_instance,
            "execute": True,
            "callback_url": status.request_uri
        }

        try:
            # Send to agent - it queues the solve in a threadpool and returns 202 immediately
            with httpx.Client() as client:
                response = client.post(AGENT_URL, json=payload, timeout=5.0)
                log(f"Agent response status: {response.status_code}")
                response.raise_for_status()
            log(f"Agent accepted request for goal: {goal_instance}")
        except Exception as e:
            logE(f"Failed to invoke agent: {e}")
            # Mark the request as failed since we couldn't even send it
            manager.update(status.request_uri.split('/')[-1],
                          execution_result="FAILURE")
            return flask.jsonify({'error': f'Failed to invoke agent: {e}'}), 500

        # Return 202 with the resource location so caller can poll it
        return flask.jsonify(status.to_dict()), 202

    except Exception as e:
        logE(f'Exception: {e}')
        return flask.jsonify({'error': f'Exception: {e}'}), 500


@app.route('/' + STATUS_SERVICE + '/<req_id>', methods=['GET'])
def get_status(req_id):
    """
    Retrieve the status of a solve request.

    Returns 200 with the full RequestStatus dict (all fields, unfinalized ones as null).
    Returns 404 if request not found (expired or never existed).
    """
    status = manager.get(req_id)
    if status is None:
        return flask.jsonify({'error': 'Request not found'}), 404

    return flask.jsonify(status.to_dict()), 200


@app.route('/' + STATUS_SERVICE + '/<req_id>', methods=['PUT'])
def update_status(req_id):
    """
    Update the status of a solve request (callback route for the solver agent).

    Body should be a dict with subset/all of: goal, capability_summary, bt_plan, execution_result.
    - execution_result from the agent is a dict with "status" key (SUCCESS/FAILURE/TIMEOUT/etc)
    - We extract the status string and store it in RequestStatus.execution_result
    - Other values can be null for unfinalized fields.

    Returns 200 if updated.
    Returns 404 if request not found (expired already).
    """
    status = manager.get(req_id)
    if status is None:
        logE(f"PUT to non-existent request {req_id}")
        return flask.jsonify({'error': 'Request not found'}), 404

    try:
        # Parse callback payload from solver agent
        payload = flask.request.get_json() or {}

        # Update only fields present in payload
        update_fields = {}

        # Map simple fields
        for key in ['goal', 'capability_summary', 'bt_plan']:
            if key in payload:
                update_fields[key] = payload[key]

        # Handle execution_result: extract status from the dict if present
        if 'execution_result' in payload:
            exec_result = payload['execution_result']
            if isinstance(exec_result, dict) and 'status' in exec_result:
                # Extract status string from the execution result dict
                update_fields['execution_result'] = exec_result['status']
            elif isinstance(exec_result, str):
                # Already a string status
                update_fields['execution_result'] = exec_result
            elif exec_result is None:
                update_fields['execution_result'] = None

        if update_fields:
            manager.update(req_id, **update_fields)

        log(f"Updated request {req_id} via callback, execution_result: {update_fields.get('execution_result')}")
        return flask.jsonify({}), 200

    except Exception as e:
        logE(f"Error updating request {req_id}: {e}")
        return flask.jsonify({'error': str(e)}), 500


@app.route('/' + STATUS_SERVICE + '/<req_id>', methods=['DELETE'])
def delete_status(req_id):
    """
    Delete a solve request resource.

    Returns 204 No Content if deleted.
    Returns 404 if request not found.
    """
    deleted = manager.delete(req_id)
    if not deleted:
        return flask.jsonify({'error': 'Request not found'}), 404

    return '', 204


log("Server routes registered")
print(app.url_map)
app.run(port=SERVER_PORT)
