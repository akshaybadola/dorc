from typing import List
import os
import sys
import yaml
import argparse

from .. import util
from . import openapi_spec


parser = argparse.ArgumentParser("OpenAPI Spec Generator",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--no-daemon", action="store_false", dest="daemon",
                    help="Do not generate spec for daemon")
parser.add_argument("--no-interface", action="store_false", dest="interface",
                    help="Do not generate spec for trainer/interface")
parser.add_argument("-df", "--daemon-spec-file", dest="daemon_spec",
                    type=str, default="daemon.yml",
                    help="Filename for daemon spec")
parser.add_argument("-if", "--interface-spec-file", type=str, dest="interface_spec",
                    default="iface.yml",
                    help="Filename for interface spec")
parser.add_argument("-e", "--excludes", type=str, default="",
                    help="Comma separated list of regexps to exclude")
parser.add_argument("--gen-opid", dest="gen_opid", action="store_true",
                    help="Generate operationId also.")
parser.add_argument("-t", "--opid-template", type=str, dest="opid_template",
                    default="[__%M%r%n]_[_%p]_%h",
                    help="Template for generation of OpenAPI operationID if --gen-opid is given.\n" +
                    """
Default template is [__%%M%%R%%N]_[_%%p]_%%m, where:
- [_%%x] represents "_".join(x) and [__%%x] == "__".join(x) etc.
- %%M is the module name
- %%r is the redirected function name
- %%n is the endpoint's basename
- %%p are the parameters to the endpoint
- %%h is the name of the HTTP method (GET,POST)

The capitalized version of these %%R indicates to capitalize that token.

See https://swagger.io/docs/specification/paths-and-operations/
For details on operationId""")
parser.add_argument("--aliases", type=str, default="",
                    help="Comma separated list of aliases for modules, " +
                    "for substituting while generating operationID.\n" +
                    "Aliases themselves are pairs of 'a:b'.\n" +
                    "Example:\n    'SomeLongModuleName:ModName,OtherModName:Other'")
args = parser.parse_args()
excludes = [r"^/_devices", r"/.*\<.*?filename\>", r"/static/.*", r"^/$"]
if args.excludes:
    excludes.extend(args.excludes.split(","))


def fix_references(spec: str) -> str:
    """Fix yaml generated references.

    For some reason, some refs aren't included by default by pydantic and yaml
    then inserts `&id001` etc references in there. This replaces those with
    `$ref`s.

    Args:
        spec: Yaml dump of OpenAPI spec generated from pydantic

    Return:
        Fixed spec dump

    """
    splits = spec.split("\n")
    refs = {}
    for i, x in enumerate(splits):
        if "&id" in x:
            ref, _id = [_.strip() for _ in x.split(": ")]
            refs[_id[1:]] = ref.rstrip(":")
            splits[i] = x.split(": ")[0] + ":"
    for i, x in enumerate(splits):
        if "*id" in x:
            name, _id = [_.strip() for _ in x.split(": ")]
            ref = refs[_id[1:]]
            splits[i] = x.split(":")[0] + ":\n" + x.split(":")[0].replace(name, "  ") +\
                f"$ref: '#/components/schemas/{ref}'"
    return "\n".join(splits)


# yaml.Dumper.ignore_aliases = lambda *args: True
if args.daemon:
    fname = args.daemon_spec
    print(f"Exclude patterns are: {excludes}")
    print(f"\n\nWill Output to: {fname}")
    dmn = util.make_test_daemon()
    if args.gen_opid:
        try:
            aliases: Dict[str, str] = dict([*x.split(":")]
                                           for x in filter(None, args.aliases.split(",")))
        except Exception as e:
            AttributeError(f"Failed to parse aliases {args.aliases}. Error {e}")
    else:
        aliases = {}
    out, err, ex = openapi_spec(dmn.app, excludes, args.gen_opid,
                                args.opid_template, aliases)
    out_str = fix_references(yaml.safe_dump(out))
    with open(fname, "w") as f:
        f.write(out_str)
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
    if args.gen_opid:
        try:
            aliases = dict([*x.split(":")]
                           for x in filter(None, args.aliases.split(",")))
        except Exception as e:
            AttributeError(f"Failed to parse aliases {args.aliases}. Error {e}")
    else:
        aliases = {}
    out, err, ex = openapi_spec(trainer.app, excludes, args.gen_opid,
                                args.opid_template, aliases)
    out_str = fix_references(yaml.safe_dump(out))
    with open(fname, "w") as f:
        f.write(out_str)
    print("\n\nErrors:\n", "\n".join(map(str, err)), file=sys.stderr)
    print("\n\nExcluded rules:\n", ex, file=sys.stderr)
    util.stop_test_interface()
