import os
import argparse
import shutil
import shlex
from subprocess import Popen, PIPE, run

from build import main as build


def _exists(host, filename, switch):
    p = Popen(f"ssh {host}  '[[ {switch} {filename} ]] ; echo $?'",
              shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return "0" in out.decode("utf-8").split("\n")[0] and not len(err.decode("utf-8"))


def file_exists(host, filename):
    return _exists(host, filename, "-f")


def dir_exists(host, filename):
    return _exists(host, filename, "-d")


def create_dir(host, dirname):
    p = Popen(shlex.split(f"ssh {host} 'mkdir -p {dirname}'"),
              stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return not len(err.decode("utf-8"))


def exec_cmd(host, cmd):
    p = Popen(f"ssh {host} '{cmd}'", shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return out, err


def deploy(host, _init=False, copy_js=False, clean_env=False):
    if _init or not dir_exists(host, "~/trainer"):
        init(host, clean_env)
    run(f"scp build/trainer.cpython-37m-x86_64-linux-gnu.so {host}:~/trainer/", shell=True)
    run(f"scp trainer/autoloads.py {host}:~/trainer/", shell=True)
    if _init or copy_js:
        run(f"scp dist/* {host}:~/trainer/dist/", shell=True)
    exec_cmd(host, f"echo '{host}' > ~/trainer/daemon_name")


def update_venv(host, delete=False):
    if delete:
        shutil.rmtree("~/trainer/env")
    if not dir_exists(host, "~/trainer/env"):
        create_dir(host, "~/trainer/env")
        print(exec_cmd(host, "python3.7 -m virtualenv -p python3.7 ~/trainer/env"))
    run(f"scp deploy_scripts/requirements.txt {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/if_run.py {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/run.py {host}:~/trainer/", shell=True)
    run(f"scp ~/bin/myauth {host}:~/", shell=True)
    print(exec_cmd(host, "python3 ~/myauth; rm ~/myauth"))
    print(exec_cmd(host, "source ~/trainer/env/bin/activate; pip install -r ~/trainer/requirements.txt"))


def init(host, clean_env=False):
    if dir_exists(host, "~/trainer"):
        print("WARNING. Directory already exists. It must be removed manually. Exiting")
        return
    create_dir(host, "~/trainer")
    create_dir(host, "~/trainer/dist")
    update_venv(host, clean_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=lambda x: x.split(","),
                        default="mc15pc15@10.5.0.96,prototype@10.5.1.93,user@10.5.0.92")
    parser.add_argument("--clean-env", action="store_true")
    parser.add_argument("--copy-js", action="store_true")
    parser.add_argument("--force-init", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    curdir = os.path.abspath(os.curdir)
    if args.update:
        build()
    os.chdir(curdir)
    for host in args.hosts:
        deploy(host, args.force_init, args.copy_js, args.clean_env)


if __name__ == '__main__':
    main()
