import os
import argparse
from subprocess import Popen, PIPE
from trainer.daemon import Daemon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        print("WILL register with trackers")
    else:
        print("WILL NOT register with trackers")
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
    daemon = Daemon(hostname="0.0.0.0", port=20202, data_dir=".demo_dir", production=False,
                    template_dir=os.path.abspath("dist"),
                    static_dir=os.path.abspath("dist"),
                    root_dir=os.path.abspath("."),
                    trackers=trackers,
                    daemon_name=daemon_name,
                    register=args.register)
    daemon.start()
