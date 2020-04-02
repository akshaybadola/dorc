import os
from trainer import Daemon


daemon = Daemon("0.0.0.0", 20202, ".daemon", True, os.path.expanduser("~/trainer/dist"),
                os.path.expanduser("~/trainer/dist"), os.path.expanduser("~/trainer/"))
daemon._root_dir = os.path.expanduser("~/trainer/")
daemon.start()
