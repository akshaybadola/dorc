import os
from trainer import Daemon


with open("daemon_name", "r") as f:
    daemon_name = f.read().split("\n")[0].strip()
daemon = Daemon("0.0.0.0", 20202, ".daemon", True, os.path.expanduser("~/trainer/dist"),
                os.path.expanduser("~/trainer/dist"), os.path.expanduser("~/trainer/"),
                daemon_name=daemon_name)
daemon._root_dir = os.path.expanduser("~/trainer/")
daemon.start()
