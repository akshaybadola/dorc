import os
import sys
import argparse
import configargparse
import dorc
from . import daemon
from . import interfaces
from . import trainer


def generate_spec(arglist):
    """Generate OpenAPI spec."""
    from typing import Dict
    import os
    import sys
    import yaml

    from . import util
    from .spec import openapi_spec, fix_yaml_references, fix_redundancies

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
    parser.add_argument("--modules", dest="modules", default="")
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
    args = parser.parse_args(arglist)
    excludes = [r"^/_devices", r"/.*\<.*?filename\>", r"/static/.*", r"^/$"]
    if args.excludes:
        excludes.extend(args.excludes.split(","))
    print(f"Exclude patterns are: {excludes}")

    # yaml.Dumper.ignore_aliases = lambda *args: True
    if args.daemon:
        fname = args.daemon_spec
        print(f"\nWill Output to: {fname}")
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
                                    args.opid_template, {"dorc": dorc,
                                                         "trainer": trainer,
                                                         "daemon": daemon,
                                                         "interfaces": interfaces},
                                    [trainer.models.Return,
                                     trainer.models.ReturnBinary,
                                     trainer.models.ReturnExtraInfo,
                                     trainer.models.TrainerState,
                                     dorc.daemon.models.Session,
                                     dorc.daemon.models.SessionMethodResponseModel,
                                     dorc.daemon.models.CreateSessionModel],
                                    aliases)
        out_str = fix_yaml_references(yaml.safe_dump(out))
        with open(fname, "w") as f:
            f.write(out_str)
        if err:
            print("\nErrors:\n", "\n".join(map(str, err)), file=sys.stderr)
        else:
            print("\nNo errors!")
        print("\nExcluded rules:\n", ex, file=sys.stderr)
        util.stop_test_daemon()
    if args.interface:
        fname = args.interface_spec
        print(f"\nWill Output to: {fname}")
        setup_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../tests/_setup.py")
        autoloads_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "autoloads.py")
        iface = util.make_test_interface(setup_path, autoloads_path)
        if args.gen_opid:
            try:
                aliases = dict([*x.split(":")]
                               for x in filter(None, args.aliases.split(",")))
            except Exception as e:
                AttributeError(f"Failed to parse aliases {args.aliases}. Error {e}")
        else:
            aliases = {}
        out, err, ex = openapi_spec(iface.app, excludes, args.gen_opid,
                                    args.opid_template, {"dorc": dorc,
                                                         "trainer": trainer,
                                                         "daemon": daemon,
                                                         "interfaces": interfaces},
                                    [trainer.models.Return,
                                     trainer.models.ReturnBinary,
                                     trainer.models.TrainerState,
                                     trainer.models.ReturnExtraInfo],
                                    aliases)
        out_str = fix_yaml_references(yaml.safe_dump(out))
        with open(fname, "w") as f:
            f.write(out_str)
        if err:
            print("\nErrors:\n", "\n".join(map(str, err)), file=sys.stderr)
        else:
            print("\nNo errors!")
        print("\nExcluded rules:\n", ex, file=sys.stderr)
        util.stop_test_interface()


def load_or_create_config(arglist):
    parser = configargparse.ArgParser(
        "Run dorc Daemon",
        default_config_files=[os.path.expanduser("~/.dorc/config.ini")])
    parser.add("-c", "--config-file",
               is_config_file=True, help="Override the default config file path")
    parser.add("-r", "--root-dir", default=os.path.expanduser("~/.dorc/"),
               help="Root Directory where all the DORC sessions and data will be stored")
    parser.add("-H", "--daemon-host", default="127.0.0.1",
               help="Host on which to bind the server. Default is localhost")
    parser.add("-P", "--daemon-port", default=4444,
               help="Port on which to serve. Default is 444")
    parser.add("-p", "--trainer-port-start", default=20202,
               help="Port range to which trainers will bind start")
    parser.add("--daemon-name", default="dorc_" + os.environ.get("HOSTNAME"),
               help="Name of the server.")
    parser.add("-v", dest="verbose", help="verbose", action="store_true")
    opts = parser.parse_args(arglist)
    opts.config_file = opts.config_file or parser._default_config_files[0]
    if not os.path.exists(opts.config_file):
        config_file = opts.__dict__.pop("config_file")
        if not os.path.exists(os.path.dirname(config_file)):
            os.makedirs(os.path.dirname(config_file))
        parser.write_config_file(opts, [config_file])
    return opts


def run(arglist):
    config = load_or_create_config(arglist)
    dmn = daemon.Daemon(config.daemon_host,
                        int(config.daemon_port),
                        config.root_dir,
                        config.daemon_name,
                        int(config.trainer_port_start))
    dmn.start()


def main():
    parser = argparse.ArgumentParser("dorc", allow_abbrev=False, add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("cmd", help="""Command to run. One of 'run' or 'generate_spec'

Type dorc cmd --help to get help about the command.""")
    if len(sys.argv) == 1:
        print(f"No command given\n")
        parser.print_help()
        sys.exit(1)
    elif sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        sys.exit(0)
    try:
        args, sub_args = parser.parse_known_args()
    except Exception:
        parser.print_help()
        sys.exit(1)
    if args.cmd == "run":
        run(sub_args)
    elif args.cmd == "generate_spec":
        generate_spec(sub_args)
    else:
        print(f"Unknown command {args.cmd}\n")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
