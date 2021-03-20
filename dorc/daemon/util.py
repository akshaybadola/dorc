from typing import List, Union, Optional, Any, Dict
import os
import sys
import time
import shlex
import json
import requests
import multiprocessing as mp
from threading import Thread
from subprocess import Popen, PIPE, TimeoutExpired
from pathlib import Path


def load_json(data):
    if isinstance(data, dict):
        return data
    elif isinstance(data, str):
        return json.loads(data)
    else:
        return None


def check_ssh_port(host: str, port: Union[int, str]) -> Union[int, str]:
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
        print(f"Got values {out.decode('utf-8')}, {err.decode('utf-8')}")
        if out.decode("utf-8") and "warn" in err.decode("utf-8").lower():
            if isinstance(port, int):
                port += 101
            else:
                # FIXME: Should be init value
                port = 10101
        elif out.decode("utf-8") and not err.decode("utf-8").lower():
            break
        p.kill()
    return port


def have_internet(username: str, password: str):
    auth_cmd = (f'curl -L -k -d username="{username}" -d password="{password}"' +
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
        if vals['out'] is not None:
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


class Tracker:
    def __init__(self):
        self._trackers = {}
        self._maybe_fwd_ports()

    def _maybe_fwd_ports(self):
        self._fwd_ports_event = None
        self._fwd_ports_thread = None
        self._fwd_procs = None
        if self.register:
            self._fwd_ports = {}
            self._fwd_procs = {}
            self._fwd_ports_event = mp.Event()
            self.fwd_ports_event.set()
            self._fwd_ports_thread = mp.Process(target=self.fwd_ports_func)
            self.fwd_ports_thread.start()
        else:
            print("Not registering with trackers", file=sys.stderr)

    @property
    def fwd_ports(self) -> Dict:
        "A :class:`dict` mapping trackers and ports forwarded to them"
        return self._fwd_ports

    @property
    def fwd_procs(self) -> Dict:
        "A :class:`dict` mapping trackers and SSH :class:`~subprocess.Popen` processes"
        return self._fwd_procs

    @property
    def fwd_ports_event(self) -> mp.Event:  # type: ignore
        """Event :class:`~multiprocessing.Event` which controls `self.fwd_port_thread`"""
        return self._fwd_ports_event

    @property
    def fwd_ports_thread(self) -> mp.Process:
        """Process :class:`multiprocessing.Process` which checks if the ports are
        correctly forwarded to the trackers.

        """
        return self._fwd_ports_thread

    @property
    def trackers(self) -> List[str]:
        """List of `user@host` strings where a tracker is present.

        Trackers are http servers which map the hostnames to forwarded ports on
        that machine. Each `daemon` when started, can register with a list of
        trackers and forward its ports. The trackers can then be used to forward
        those ports back to user machine. Convoluted I know.

        """
        return self._trackers

    def fwd_ports_func(self) -> None:
        """Forward ports at one minute interval to `self.trackers`

        This function runs in a separate process and checks the SSH forwarded
        ports and forwards any stale/dead ports if required.

        """
        while self.fwd_ports_event.is_set():
            # self._logi("Checking port forwards")
            self.check_and_register_with_trackers()
            if not self.fwd_ports_event.is_set():
                # self._logi("Exiting from fwd_ports_func")
                return
            else:
                time.sleep(60)

    # FIXME: This thing will start in a thread and if this is available to call
    #        from an endpoint then there could be race conditions
    def check_and_register_with_trackers(self):
        """Checks if the all the ports are correctly forwarded.

        It checks all the ports with internal functions which use various shell
        commands. If any port is not correctly forwarded it forwards that port
        and registers that port correctly with the tracker.

        """
        if not self.trackers:
            self._logd(f"Empty tracker list. Will not do do anything")
            return
        if self.daemon_name is None:
            try:
                with open("daemon_name", "r") as f:
                    daemon_name = f.read().split("\n")[0].strip()
            except Exception:
                daemon_name = "No Nmae"
        else:
            daemon_name = self.daemon_name

        def _check_fwd_port(host, port):
            if port == "UNREACHABLE":
                return False
            try:
                self._logd(f"Checking {host}:{port}")
                p = Popen(shlex.split(f"ssh {host} \"curl http://localhost:{port}/_name\""),
                          stdout=PIPE, stderr=PIPE)
                out, err = p.communicate(timeout=3)
                if daemon_name == out.decode("utf-8"):
                    return True
                else:
                    self._logd(f"Incorrect port registerd with tracker {host}")
                    return False
            except TimeoutExpired:
                p.kill()
                return False

        def _fwd_port(host):
            if host in self.fwd_procs:
                self.fwd_procs[host].kill()
            self.fwd_ports[host] = port = check_ssh_port(host, self.fwd_port_start)
            if port != "UNREACHABLE":
                self.fwd_procs[host] = Popen(shlex.split(f"ssh -N -R {port}:localhost:20202 {host}"),
                                             stdout=PIPE, stderr=PIPE)
            return port

        def _register(host, daemon_name, port):
            resp = register_with_tracker(tracker, daemon_name, port)
            if resp is not None:
                self._logi(f"Forwarded port {port}, with name {daemon_name} to {tracker}." +
                           f"Response is {resp}")
            else:
                self._loge(f"Connection error from {daemon_name}. Could not forward port")

        _check = self.fwd_ports_event.is_set()
        print(f"Checking ports and Registering with {self.trackers}")
        if _check:
            for tracker in self.trackers:
                if tracker in self.fwd_ports:
                    port = self.fwd_ports[tracker]
                    if port == "UNREACHABLE":
                        # NOTE: Skip if unreachable
                        continue
                    elif _check_fwd_port(tracker, port):
                        _register(tracker, daemon_name, port)
                    else:
                        self._logi(f"Bad port {port}, with {daemon_name} to {tracker}.")
                        new_port = _fwd_port(tracker)
                        if new_port == "UNREACHABLE":
                            continue
                        else:
                            _register(tracker, daemon_name, new_port)
                else:
                    print(f"Finding new port for {tracker}")
                    port = _fwd_port(tracker)
                    if port == "UNREACHABLE":
                        continue
                    else:
                        _register(tracker, daemon_name, port)
