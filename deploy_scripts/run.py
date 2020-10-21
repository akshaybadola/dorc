import os
from subprocess import Popen, PIPE
from trainer import Daemon


try:
    with open("daemon_name", "r") as f:
        daemon_name = f.read().split("\n")[0].strip()
except Exception as e:
    daemon_name = None
    print(f"{e}")
if not daemon_name:
    p = Popen("echo `whoami`@`hostname`", shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    daemon_name = out.decode("utf-8").split("\n")[0].strip()
try:
    with open("trackers_list") as f:
        trackers = [x.strip() for x in f.read().split("\n")[0].split(",")]
except Exception as e:
    print(f"{e}")
    trackers = []
daemon = Daemon(hostname="0.0.0.0", port=20202, data_dir=".daemon", production=True,
                template_dir=os.path.expanduser("~/trainer/dist"),
                static_dir=os.path.expanduser("~/trainer/dist"),
                root_dir=os.path.expanduser("~/trainer/"),
                trackers=trackers,
                daemon_name=daemon_name)
daemon._root_dir = os.path.expanduser("~/trainer/")
daemon.start()
