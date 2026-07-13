from importlib import import_module
from sys import stderr
import time
import flask

SERVER_URL = "http://localhost";
SERVER_PORT = 5565;
SOLVE_SERVICE = "solve";
STATUS_SERVICE = "status";
INPUT_DATA_PARAM = "input_data";


head = "<ML server> "
def log(*args): print(f"{head}", *args)

def logE(*args): print(f"{head}", *args, file = stderr, flush = True)

log("creating server on port", SERVER_PORT, "... ")
app = flask.Flask(__name__)
app.config["DEBUG"] = True

status = None

@app.route('/' + SOLVE_SERVICE, methods=['POST'])
def solve():
    global status
    log("llm service")
    try:
        input_data = flask.request.form.get(INPUT_DATA_PARAM)
        # response = {'result': "result here"}
        # ret = flask.jsonify(response)
        # log("returned", ret)
        # return ret
        print("solving with input_data", input_data)
        status = "result"
        return "ok", 200
    except Exception as e:
         flask.jsonify({'error': f'Exception {e}.'}), 500

@app.route('/' + STATUS_SERVICE, methods=['POST'])
def status():
    if status is None:
        return flask.jsonify(status="processing"), 202
    return status, 200

print(app.url_map)
app.run(port = SERVER_PORT)