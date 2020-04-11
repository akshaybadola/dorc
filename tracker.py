import json
import time
import atexit
import multiprocessing as mp
from subprocess import Popen, PIPE, TimeoutExpired
from flask import Flask, request
from werkzeug import serving


app = Flask(__name__)
daemons = {}


def check():
    for server, port in daemons:
        p = Popen(f"curl localhost:{port}", stdout=PIPE, stderr=PIPE)
        try:
            out, err = p.communicate(timeout=2)
        except TimeoutExpired:
            daemons[(server, port)] = False
        daemons[(server, port)] = True
    time.sleep(60)


@app.route("/", methods=["POST"])
def index():
    try:
        if isinstance(request.json, dict):
            data = request.json
        else:
            data = json.loads(request.json)
        if "put" in data:
            daemons[data["hostname"]] = data["port"]
            return "Success"
        elif "get" in data:
            return json.dumps(daemons)
    except Exception as e:
        print(e)
        return "Failure"


t = mp.Process(target=check)
t.start()
atexit.register(lambda: t.terminate())
serving.run_simple("127.0.0.1", 11111, app)
