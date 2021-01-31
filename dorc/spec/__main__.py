import os
import sys
import yaml
import argparse

from .. import util
from . import openapi_spec


parser = argparse.ArgumentParser("OpenAPI Spec Generator")
parser.add_argument("--no-daemon", action="store_false", dest="daemon",
                    help="Do not generate spec for daemon")
parser.add_argument("--no-interface", action="store_false", dest="interface",
                    help="Do not generate spec for trainer/interface")
parser.add_argument("--daemon-spec", type=str, default="daemon.yml",
                    help="Filename for daemon spec")
parser.add_argument("--interface-spec", type=str, default="iface.yml",
                    help="Filename for interface spec")
parser.add_argument("--excludes", type=str, default="",
                    help="Comma separated list of regexps to exclude")
args = parser.parse_args()
excludes = [r"^/_devices", r"/.*\<.*?filename\>", r"/static/.*", r"^/$"]
if args.excludes:
    excludes.extend(args.excludes.split(","))


if args.daemon:
    fname = args.daemon_spec
    print(f"Exclude patterns are: {excludes}")
    print(f"\n\nWill Output to: {fname}")
    dmn = util.make_test_daemon()
    out, err, ex = openapi_spec(dmn.app, excludes)
    with open(fname, "w") as f:
        yaml.dump(out, f)
    print("\n\nErrors:\n", "\n".join(map(str, err)), file=sys.stderr)
    print("\n\nExcluded rules:\n", ex, file=sys.stderr)
    util.stop_test_daemon()
if args.interface:
    fname = args.interface_spec
    setup_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../tests/_setup_local.py")
    autoloads_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../autoloads.py")
    trainer = util.make_test_interface(setup_path, autoloads_path)
    out, err, ex = openapi_spec(trainer.app, excludes)
    with open(fname, "w") as f:
        yaml.dump(out, f)
    print("\n\nErrors:\n", "\n".join(map(str, err)), file=sys.stderr)
    print("\n\nExcluded rules:\n", ex, file=sys.stderr)
    util.stop_test_interface()
