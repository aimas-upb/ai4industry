from importlib import import_module
from sys import stderr
import time
import flask

SERVER_URL = "http://localhost";
SERVER_PORT = 5565;
SOLVE_SERVICE = "solve";
INPUT_DATA_PARAM = "input_data";


head = "<ML server> "
def log(*args): print(f"{head}", *args)

def logE(*args): print(f"{head}", *args, file = stderr, flush = True)


# def import_functionality(name, pippackage = None, critical = False, autoinstall = False):
#     components = name.split('.')
#     package = components[0]
#     log("importing", package)
#     try:
#         mod = import_module(package)
#         for comp in components[1:]:
#             mod = getattr(mod, comp)
#         return mod
#     except Exception as e:
#         pippackage = pippackage if pippackage is not None else name.split(".")[0]
#         log(package, "unavailable (use pip install", pippackage, "):", e)
#         # TODO check if should autoinstall, and then do
#         if critical: exit(1)
#     return None
#
# flask = import_functionality("flask")
log("creating server on port", SERVER_PORT, "... ")
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/' + SOLVE_SERVICE, methods=['POST'])
def solve():
    global models
    global datasets
    log("prediction service")
    try:
        input_data = flask.request.form.get(INPUT_DATA_PARAM)
        response = {'result': "result here"}
        ret = flask.jsonify(response)
        log("returned", ret)
        return ret
    except Exception as e:
         flask.jsonify({'error': f'Exception {e}.'}), 500

print(app.url_map)
app.run(port = SERVER_PORT)