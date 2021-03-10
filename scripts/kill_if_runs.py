import requests
import psutil


pyprocs = [x for x in psutil.process_iter() if "python" in x.cmdline()]
for p in pyprocs:
    if "if_run.py" in p.cmdline()[1] and "/tests/" in p.cmdline()[4]:
        port = p.cmdline()[3]
        requests.get(f"http://localhost:{port}/_shutdown")
