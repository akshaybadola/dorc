import sys
import yaml

from .. import util
from . import openapi_spec


excludes = [r"^/_devices", r"/.+\<.+?filename\>", r"/static/.*", r"^/$"]
fname = "spec.yml"
print(f"Exclude patterns are: {excludes}")
print(f"\n\nWill Output to: {fname}")
dmn = util.make_test_daemon()
out, err, ex = openapi_spec(dmn.app, excludes)
with open("spec.yml", "w") as f:
    yaml.dump(out, f)
print("\n\nErrors:\n", err, file=sys.stderr)
print("\n\nExcluded rules:\n", ex, file=sys.stderr)
util.stop_test_daemon()
