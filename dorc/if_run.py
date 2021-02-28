import os
import json
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dorc.interfaces import FlaskInterface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("hostname", type=str)
    parser.add_argument("port", type=int)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("gmods_dir", type=str)
    parser.add_argument("gdata_dir", type=str)
    parser.add_argument("daemon_url", type=str)
    parser.add_argument("--config-overrides", type=str, default="False")
    args, _ = parser.parse_known_args()
    hostname = args.hostname
    port = args.port
    data_dir = args.data_dir
    config_file = os.path.join(data_dir, "config_overrides.json")
    if os.path.exists(config_file):
        print("CONFIG OVERRIDES")
        with open(config_file) as f:
            config = json.load(f)
    else:
        print("NO CONFIG OVERRIDES")
        config = None
    # sys.path.append(os.path.join(data_dir, "..", ".."))
    sys.argv = [sys.argv[0]]
    FlaskInterface(hostname, port, data_dir, args.gmods_dir, args.gdata_dir,
                   args.daemon_url, config_overrides=config)
