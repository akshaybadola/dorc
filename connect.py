import os
import json
import time
import shlex
import atexit
import argparse
import requests
from queue import Queue
from threading import Thread
from subprocess import Popen, run, PIPE, TimeoutExpired


hosts = ["joe@13.232.207.179", "joe@149.129.189.46", "15mcpc15@mars.uohyd.ac.in"]


def check_servers(host):
    # How many servers are up and the ports forwarded correctly
    p = Popen(shlex.split(f"ssh -N -L 11111:localhost:11111 {host}"),
              stdout=PIPE, stderr=PIPE)
    time.sleep(2)
    response = requests.request("POST", "http://localhost:11111/",
                                json={"get": True})
    output = json.loads(response.content)
    servers = {}
    for server, port in output.items():
        print(f"Checking for {host}, {server}")
        p = Popen(f"ssh {host} \"curl localhost:{port}\"", shell=True,
                  stdout=PIPE, stderr=PIPE)
        try:
            servers[(server, port)] = p.communicate(timeout=2)
            p.terminate()
        except TimeoutExpired:
            p.terminate()
            servers[(server, port)] = None
    p.terminate()
    return servers


def bleh(hosts, connect_procs={}):
    outputs = {}
    responses = []
    port = 8181
    for host in hosts:
        p = Popen(shlex.split(f"ssh -N -L 11111:localhost:11111 {host}"),
                  stdout=PIPE, stderr=PIPE)
        time.sleep(2)
        responses.append(requests.request("POST", "http://localhost:11111/",
                                          json={"get": True}))
        outputs[host] = json.loads(responses[-1].content)
        connect_procs[host] = {}
        for k, v in outputs[host].items():
            Popen(shlex.split(f"ssh -N -L {port}:localhost:{v} {host}"),
                  stdout=PIPE, stderr=PIPE)
        p.terminate()
    return outputs, connect_procs


def check_output(p, q):
    q.put(p.stdout.read(4).decode("utf-8"))


class Connector:
    def __init__(self, server, start_port, latency, parallel, server_port, use_paramiko):
        self.server = server
        self.start_port = start_port
        self.latency = latency
        self.parallel = parallel
        self.server_port = server_port
        self.use_paramiko = use_paramiko
        self.proc_q = Queue()
        self.procs = {}

    def forward_port(self, port, fwd_port):
        if not isinstance(port, tuple) and not isinstance(port, list):
            while True:
                print(f"Forwarding remote port {port} to local port {fwd_port}")
                p = Popen(f"ssh -L {fwd_port}:localhost:{port} {self.server}" +
                          " \"hostname && /bin/bash \"",
                          shell=True, stdout=PIPE, stderr=PIPE)
                q = Queue()
                t = Thread(target=check_output, args=[p, q])
                t.start()
                print(f"Waiting {self.latency} seconds for {port}")
                time.sleep(self.latency)
                if not q.empty():
                    print(f"Output from ssh: {q.get()}")
                    self.proc_q.put(p)
                    print(f"Successfully forwarded {fwd_port} from {self.server}:{port}")
                    break
                else:
                    p.kill()
                t.join()
                time.sleep(1)

    def ping(self):
        while True:
            for host, info in self.procs.items():
                port = info["local_port"]
                try:
                    requests.request("GET", f"http://localhost:{port}/_ping",
                                     timeout=5)
                except Exception:
                    print(f"Ping failed on host, port: {host}, {port}")
                    ports_info = self.get_ports_info()
                    remote_port = ports_info[host]
                    if remote_port != info["remote_port"]:
                        print(f"Remote port changed forwarding again")
                    info["proc"].kill()
                    self.forward_port(remote_port, port)
                    self.procs[host] = {"proc": self.proc_q.get(),
                                        "remote_port": remote_port,
                                        "local_port": port}
            time.sleep(10)

    def get_ports_info(self):
        if self.use_paramiko:
            import paramiko
            client = paramiko.SSHClient()
            client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
            user, remote_host = self.server.split("@")
            connect_port = 22
            client.connect(remote_host, connect_port, username=user,
                           key_filename=os.path.expanduser("~/.ssh/id_rsa"))
            cmd = 'curl -H "Content-Type: application/json" --data \'{"get": true}\' localhost:11111'
            stdin, stdout, stderr = client.exec_command(cmd)
            lines = [*stdout]
            client.close()
            return json.loads(lines[0])
        else:
            while True:
                print(f"Trying to forward port {self.server_port} from {self.server}")
                p = Popen(f"ssh -L {self.server_port}:localhost:{self.server_port} {self.server}" +
                          " \"hostname && /bin/bash \"", shell=True, stdout=PIPE, stderr=PIPE)
                time.sleep(2)
                out = p.stdout.read(4).decode("utf-8")
                if out:
                    print("Getting the hosts and ports")
                    resp = requests.request("POST", f"http://localhost:{self.server_port}/",
                                            json={"get": True})
                    ports = json.loads(resp.content)
                    print(f"Got ports info {ports}")
                    p.kill()
                    return ports
                p.kill()
                time.sleep(2)

    def main(self):
        remote_ports = self.get_ports_info()
        local_port = self.start_port
        if self.parallel:
            raise NotImplementedError
            # proc_q = Queue()
            # port_threads = []
            # for host, port in remote_ports.items():
            #     port_threads.append(Thread(target=forward_port, args=[server, port, latency,
            #                                                           proc_q, local_port]))
            #     local_ports.append(local_port)
            #     local_port += 101
            #     port_threads[-1].start()
            # for t in port_threads:
            #     t.join()
            #     self.procs.append(proc_q.get())
        else:
            for host, port in remote_ports.items():
                self.forward_port(port, local_port)
                self.procs[host] = {"proc": self.proc_q.get(),
                                    "remote_port": port,
                                    "local_port": local_port}
                local_port += 101
        # ping_thread = Thread(target=self.ping)
        # ping_thread.start()
        self.ping()
        for p in self.procs.values():
            p["proc"].wait()

        @atexit.register
        def bleh():
            for p in self.procs.values():
                p[0].terminate()


# curl -H "Content-Type: application/json" --data '{"get": true}' localhost:11111
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="")
    parser.add_argument("--all-servers", action="store_true")
    parser.add_argument("--start-port", type=int, default=8181)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--latency", type=int, default=2)
    parser.add_argument("--no-use-paramiko", dest="use_paramiko", action="store_false")
    parser.add_argument("--server-port", type=int, default=11111)
    args = parser.parse_args()
    if args.all_servers:
        print("Forwarding from all servers is not implemented yet")
    connector = Connector(args.server, args.start_port, args.latency,
                          args.parallel, args.server_port, args.use_paramiko)
    connector.main()
