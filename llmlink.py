from importlib import import_module
from sys import stderr
import time
import flask
import httpx
import logging
from pprint import pprint

SERVER_URL = "http://localhost";
SERVER_PORT = 5565;
SOLVE_SERVICE = "solve";
STATUS_SERVICE = "status";
INPUT_DATA_PARAM = "input_data";

AGENT_URL = "http://localhost:8008/solve"


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

head = "<ML server> "
def log(*args): logger.info(f"{head}", *args)

def logE(*args): logger.error(f"{head}", *args)

log("creating server on port", SERVER_PORT, "... ")
app = flask.Flask(__name__)
app.config["DEBUG"] = True

status = None

@app.route('/' + SOLVE_SERVICE, methods=['POST'])
def solve():
    global status
    log("llm service")
    try:
        goal_instance = flask.request.form.get(INPUT_DATA_PARAM)
        # response = {'result': "result here"}
        # ret = flask.jsonify(response)
        # log("returned", ret)
        # return ret
        print("solving with input_data", goal_instance)
        
        # creating the payload for the agent
        payload = {
            "goal": goal_instance,
            "execute": True  # Set to True to execute the generated BT
        }
        
        response = httpx.post(AGENT_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        log("Agent response received successfully")

        # Print goal request results
        # The original goal
        print(f"\nGoal: {result['goal']}")

        # The results of the artifact discovery procedure - capability summary
        print(f"\n{'Capability Summary:'}")
        print("-" * 80)
        print(result['capability_summary'])

        # The BehaviorTree plan that was generated and executed
        print(f"\n{'Generated BehaviorTree Plan:'}")
        print("-" * 80)
        pprint(result['bt_plan'], width=80)
        
        # The execution result
        if result['execution_result']:
            exec_result = result['execution_result']
            print(f"\n{'Execution Result:'}")
            print("-" * 80)
            print(f"Status: {exec_result['status']}")
            print(f"Ticks: {exec_result['ticks']}")

            if exec_result['status'] == 'SUCCESS':
                print(f"\n Goal achieved!")
            elif exec_result['status'] == 'FAILURE':
                print(f"\n Goal execution failed")
            else:
                print(f"\n Goal execution timed out")

            # Print execution trace with proper indentation by depth
            if exec_result['trace']:
                print(f"\n{'Execution Trace:'}")
                print("-" * 80)
                for entry in exec_result['trace']:
                    status_symbol = "✓" if entry['status'] == 'SUCCESS' else "✗" if entry['status'] == 'FAILURE' else "→"
                    depth = entry.get('depth', 0)
                    indent = "  " * depth
                    print(f"  [{entry['tick']:2d}] {indent}{status_symbol} {entry['node']:30s} [{entry['type']:20s}] {entry['status']}")
                    if entry['details']:
                        detail_indent = "  " * (depth + 1)
                        print(f"       {detail_indent}└─ {entry['details']}")

        # return the result as JSON string and a 200 status code
        return flask.jsonify(result), 200

        # status = "result"
        # return "ok", 200
    except Exception as e:
         logE(f'Exception {e}.')
         return flask.jsonify({'error': f'Exception {e}.'}), 500

@app.route('/' + STATUS_SERVICE, methods=['GET'])
def status():
    global status
    if status is None:
        return flask.jsonify(status="processing"), 202
    return status, 200


@app.route('/' + STATUS_SERVICE, methods=['PUT'])
def update_status(new_status):
    global status
    status = new_status

print(app.url_map)
app.run(port = SERVER_PORT)