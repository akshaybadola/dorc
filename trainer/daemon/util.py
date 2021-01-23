from typing import List, Union, Optional, Any
import os
import time
import shlex
import requests
import shutil
from threading import Thread
from subprocess import Popen, PIPE, TimeoutExpired
from pathlib import Path


def get_hostname() -> str:
    p = Popen("hostname", stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return out.decode("utf-8")


def check_ssh_port(host: str, port: int) -> int:
    timeout = 2
    while True:
        print(f"Checking port {port}")
        out, err = b"", b""
        ip_addr = host.split("@")[1]
        p = Popen(f"nc -z -v {ip_addr} 22", shell=True, stdout=PIPE, stderr=PIPE)
        try:
            out, err = p.communicate(timeout=timeout)
        except TimeoutExpired:
            port = "UNREACHABLE"
            break
        p = Popen(shlex.split(f"ssh -R {port}:localhost:20202 {host} hostname"),
                  stdout=PIPE, stderr=PIPE)
        try:
            out, err = p.communicate(timeout=timeout)
        except Exception:
            pass
        print(f"Got values {out}, {err}")
        if out.decode("utf-8") and "warn" in err.decode("utf-8").lower():
            port += 101
        elif out.decode("utf-8") and not err.decode("utf-8").lower():
            break
        p.kill()
    return port


def have_internet():
    auth_cmd = ('curl -L -k -d username="15mcpc15" -d password="unmission@123"' +
                ' -d mode=191 http://192.168.56.2:8090/login.xml')

    def communicate(p, vals):
        vals['out'], vals['err'] = p.communicate()

    def connect(auth_cmd):
        vals = {'out': None, 'err': None}
        p = Popen(auth_cmd, shell=True, stdout=PIPE, stderr=PIPE)
        t = Thread(target=communicate, args=[p, vals])
        t.start()
        t.join(timeout=5)
        p.kill()
        if vals['out'] and "You have successfully logged in" in vals['out'].decode('utf-8'):
            return True
        else:
            return False

    while True:
        vals = {'out': None, 'err': None}
        p = Popen("curl google.com".split(), stdout=PIPE, stderr=PIPE)
        t = Thread(target=communicate, args=[p, vals])
        t.start()
        t.join(timeout=5)
        p.kill()
        if vals['out']:
            if "the document has moved" not in vals['out'].decode('utf-8').lower():
                connect(auth_cmd)
        time.sleep(60)


def register_with_tracker(tracker, host, port):
    status = False
    fwd_port = 11111
    procs = []
    while not status:
        procs.append(Popen(shlex.split(f"ssh -N -L {fwd_port}:localhost:11111 {tracker}"),
                           stdout=PIPE, stderr=PIPE))
        time.sleep(3)
        try:
            print(f"Registering port {port} at {tracker}")
            resp = requests.request("POST", f"http://localhost:{fwd_port}/",
                                    json={"put": True,
                                          "hostname": host,
                                          "port": port}).content
            status = True
        except requests.ConnectionError as e:
            print(f"Connection refused from server {e}")
            resp = None
            status = True
        except Exception as e:
            print(f"Register request at port {fwd_port} with {tracker} failed {e}. Trying again")
            resp = None
    for p in procs:
        p.kill()
    return resp


def create_module(module_dir: Union[Path, str],
                  module_files: List[Union[Path, str]] = [],
                  env_str: str = ""):
    """Create a module named module_dir.

    Args:
        module_dir: The name of the module
        module_files: Files in the module
        env_str: Extra lines to be appended at the top of the file

    Usually an `import sys; sys.path.append('some_path')` can be appended to the
    file(s) to include any extra paths. Care has to be taken so that names are
    not overwrriten by the preceding lines. `env_str` will be appended to the
    top of the file.

    """
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write(env_str)
    for mf in module_files:
        with open(mf) as f:
            file_str = env_str + f.read()
        with open(os.path.join(module_dir, os.path.basename(mf)), "w") as f:
            f.write(file_str)
