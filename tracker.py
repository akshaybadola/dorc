import json
from flask import Flask, request
from werkzeug import serving


app = Flask(__name__)
daemons = {}


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


serving.run_simple("127.0.0.1", 11111, app)
