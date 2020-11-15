import sys
import json
import time
import shlex
from threading import Thread, Event
from subprocess import Popen, PIPE
from flask import Flask, request
from werkzeug import serving


app = Flask(__name__)
daemons = {}


def check(flag):
    while flag.is_set():
        for server, port in daemons.items():
            print(f"Checking daemons {daemons}")
            if isinstance(port, tuple):
                print(f"Daemon {server} is already dead")
                continue
            p = Popen(shlex.split(f"curl localhost:{port}"), stdout=PIPE, stderr=PIPE)
            out, err = b"", b""
            try:
                out, err = p.communicate(timeout=2)
            except Exception:
                daemons[server] = (port, False)
                print(f"Dead Server {server}")
            if err and "failed connect" in err.decode("utf-8").lower():
                daemons[server] = (port, False)
                print(f"Dead Server {server}")
            p.kill()
        time.sleep(10)


@app.route("/", methods=["POST"])
def index():
    try:
        if isinstance(request.json, dict):
            data = request.json
        else:
            data = json.loads(request.json)
        if "put" in data:
            print(f"Registerd {data}")
            daemons[data["hostname"]] = data["port"]
            return "Success"
        elif "get" in data:
            return json.dumps(daemons)
    except Exception as e:
        print(e)
        return "Failure"


flag = Event()
flag.set()
t = Thread(target=check, args=[flag])
t.start()
try:
    serving.run_simple("127.0.0.1", 11111, app)
    flag.clear()
except KeyboardInterrupt:
    flag.clear()
    print("Exiting")
    sys.exit()
