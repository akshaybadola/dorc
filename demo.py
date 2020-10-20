import os
import argparse
from trainer.daemon import _start_daemon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        print("WILL register with trackers")
        with open("daemon_name", "r") as f:
            daemon_name = f.read().split("\n")[0].strip()
        print("Daemon name", daemon_name)
    else:
        print("WILL NOT register with trackers")
        daemon_name = None
    daemon = _start_daemon("0.0.0.0", 20202, ".demo_dir", False,
                           os.path.abspath("dist"), os.path.abspath("dist"),
                           os.path.abspath("."), daemon_name=daemon_name,
                           register=args.register)
