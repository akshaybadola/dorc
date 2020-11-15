import os
import re
import time
import glob
import multiprocessing as mp
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


def deploy(host, _init=False, copy_js=False, clean_env=False, _restart=False):
    """DORC deployment:

    Deploy DORC servers on hosts. It uses SSH and it's preferable to have ssh
    keys installed on the remote machines as otherwise you may get password
    prompts. We use Popen as a simple interface to SSH.

    :param host: a string of the type of \"user@host\" on which to deploy
    :param _init: create all the directory structure, default is \"~/trainer\"
    :param copy_js: (deprecated) copy the JS files
    :param clean_env: Update the python environment for the trainer. It's installed in the
                      directory ~/trainer/env
    :param _restart: Restart the daemon if running after

    """
    print(f"Deploying to host {host}")
    if not dir_exists(host, os.path.expanduser("~/trainer")):
        print(f"Trainer directory doesn't exist on {host}. Initializing")
        _init = True
    if _init:
        init(host, clean_env)
    run(f"scp build/trainer.cpython-37m-x86_64-linux-gnu.so {host}:~/trainer/", shell=True)
    run(f"scp trainer/autoloads.py {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/if_run.py {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/run.py {host}:~/trainer/", shell=True)
    if _init or copy_js:
        print(f"Copying js files to {host}")
        run(f"scp dist/* {host}:~/trainer/dist/", shell=True)
    exec_cmd(host, f"echo '{host}' > ~/trainer/daemon_name")
    if _restart:
        restart(host)


def update_venv(host, delete=False):
    if delete:
        print(f"Removing env dir on {host}")
        shutil.rmtree(os.path.expanduser("~/trainer/env"))
    if not dir_exists(host, os.path.expanduser("~/trainer/env")):
        create_dir(host, os.path.expanduser("~/trainer/env"))
        print(f"Creating virtualenv on {host}")
        print(exec_cmd(host, "python3.7 -m virtualenv -p python3.7 ~/trainer/env"))
    run(f"scp deploy_scripts/requirements.txt {host}:~/trainer/", shell=True)
    run(f"scp ~/bin/myauth {host}:~/", shell=True)
    print(exec_cmd(host, "python3 ~/myauth; rm ~/myauth"))
    print(f"Installing requirements on {host}. Might take a while.")
    print(exec_cmd(host, "source ~/trainer/env/bin/activate; pip install -r ~/trainer/requirements.txt"))


def init(host, clean_env=False):
    if dir_exists(host, os.path.expanduser("~/trainer")):
        print("WARNING. Directory already exists. It must be removed manually. Exiting")
        return
    create_dir(host, os.path.expanduser("~/trainer"))
    create_dir(host, os.path.expanduser("~/trainer/dist"))
    update_venv(host, clean_env)
    dstr = """#! /bin/bash

source ~/new_env/bin/activate
cd ~/trainer
python run.py
"""
    with open("._tmp.sh", "w") as f:
        f.write(dstr)
    run(f"scp ._tmp.sh {host}:~/.daemon.sh", shell=True)
    exec_cmd(host, "chmod 755 ~/.daemon.sh")


def restart(host):
    print(f"Restarting daemon for {host}")
    exec_cmd(host, "killall ssh")
    time.sleep(1)
    out, err = exec_cmd(host, "ps -ef | grep -i \"python run.py\"")
    splits = [re.split(" +", x) for x in out.decode("utf-8").split("\n")]
    for s in splits:
        if len(s) > 1 and s[1]:
            print(f"Killing {' '.join(s)}")
            print(exec_cmd(host, f"kill -s SIGTERM {s[1]}"))
    p = mp.Process(target=exec_cmd, args=[host, f"nohup ~/.daemon.sh >> ~/nohup.out"])
    p.start()
    time.sleep(1)
    p.terminate()


def create_zip():
    if os.path.exists(".zip_temp"):
        shutil.rmtree(".zip_temp")
    os.mkdir(".zip_temp")
    os.mkdir(".zip_temp/trainer")
    os.mkdir(".zip_temp/trainer/dist")
    trainer_dir = ".zip_temp/trainer"
    shutil.copy("build/trainer.cpython-37m-x86_64-linux-gnu.so", trainer_dir)
    shutil.copy("trainer/autoloads.py", trainer_dir)
    shutil.copy("deploy_scripts/if_run.py", trainer_dir)
    shutil.copy("deploy_scripts/run.py", trainer_dir)
    shutil.copy("deploy_scripts/requirements.txt", trainer_dir)
    dist_files = glob.glob("dist/**", recursive=True)
    with open(os.path.join(trainer_dir, "README"), "w") as f:
        f.write("""Instructions:
1. Create a venv in any directory and install the requirements in requirements.txt
2. Default path is \"~/trainer\". If you decide to keep the contents of this
   directory anywhere else, you should change those paths in \"run.py\".
   Also add register=False as an option to \"Daemon\".
3.. Create a file "daemon_name" in the current directory and insert any name in it.
4. After installing the requirements, run \"python run.py\" in the current directory
""")
    for f in dist_files:
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(trainer_dir, "dist"))
    run("cd .zip_temp; 7z a trainer.zip trainer; mv trainer.zip ..", shell=True)


def main():
    parser = argparse.ArgumentParser(description="""DORC deployment:

    Deploy DORC servers on hosts with SSH. Default is to copy the compiled C module
    and update the global modules. See help for additional options.

    """)
    parser.add_argument("--hosts", type=lambda x: x.split(","),
                        default="mc15pc15@10.5.0.96,prototype@10.5.1.93,user@10.5.0.92,taruna@10.5.1.3",
                        help="Comma separated user@host to deploy")
    parser.add_argument("--clean-env", action="store_true",
                        help="Also clean the python venv for the trainer on the hosts")
    parser.add_argument("--copy-js", action="store_true",
                        help="Copy the javascript sources (deprecated)")
    parser.add_argument("--force-init", action="store_true",
                        help="""Initialize the trainer from scratch. Creates the directories and installs
                        dependencies using ssh. If the directory already exists,
                        then it has to be removed manually as a precaution.""")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the trainer module with cython")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile the C module, do nothing else")
    parser.add_argument("--create-zip", action="store_true",
                        help="Create the zip file of the package" +
                        "Doesn't do anything else")
    parser.add_argument("--restart", action="store_true", help="Restart the daemons on the hosts")
    parser.add_argument("--restart-only", action="store_true", help="Only restart the daemons" +
                        "on the hosts, don't do anything else")
    args = parser.parse_args()
    curdir = os.path.abspath(os.curdir)
    if args.compile or args.compile_only:
        build()
    if args.create_zip:
        create_zip()
        return
    if args.compile_only:
        return
    os.chdir(curdir)
    for host in args.hosts:
        if args.restart_only:
            restart(host)
        else:
            deploy(host, args.force_init, args.copy_js, args.clean_env, args.restart)


if __name__ == '__main__':
    main()
