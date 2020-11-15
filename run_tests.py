import os
from subprocess import Popen, PIPE, run

os.chdir("tests")
test_files = ["test_checks.py",
              "test_task.py",
              "test_epoch.py",
              "test_epoch_loop.py",
              "test_modules.py",
              "test_trainer_device.py",
              "test_trainer.py",
              "test_trainer_cuda.py",
              "*test_trainer_state.py, big issues",
              "test_daemon.py",
              "test_daemon_auth.py",
              "test_daemon_http.py",
              "test_daemon_load_unload.py",
              "test_daemon_modules_http.py",
              "test_clone_archive.py",
              "test_interfaces.py"]

test_files = [f for f in test_files if not f.startswith("*")]
out = []
err = []
for i, f in enumerate(test_files):
    # if i+1 > 12:
    #     break
    append = "-a" if not i else ""
    print(f"Testing file {f}, {i+1} out of {len(test_files)}")
    p = Popen(f"python -m coverage run {append} --source=.. -m unittest {f}",
              shell=True, stdout=PIPE, stderr=PIPE)
    x, y = p.communicate()
    out.append([f, x.decode("utf-8")])
    err.append([f, y.decode("utf-8")])
run("coverage report -i --include=/home/joe/projects/trainer/trainer/*", shell=True)
