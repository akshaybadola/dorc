import os
import json
import argparse
import sys
from trainer.interfaces import FlaskInterface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("hostname", type=str)
    parser.add_argument("port", type=int)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--production", type=str, default="False")
    parser.add_argument("--config-overrides", type=str, default="False")
    args, _ = parser.parse_known_args()
    hostname = args.hostname
    port = args.port
    data_dir = args.data_dir
    if args.production.lower() == "true":
        production = True
    else:
        production = False

    config_file = os.path.join(data_dir, "config_overrides.json")
    if os.path.exists(config_file):
        print("WE HAVE CONFIG")
        with open(config_file) as f:
            config = json.load(f)
    else:
        print("WE DIDN'T HAVE CONFIG. SO SAD")
        config = None
    sys.path.append(os.path.join(data_dir, "..", ".."))
    sys.argv = [sys.argv[0]]
    FlaskInterface(hostname, port, data_dir, production=production,
                   config_overrides=config)
