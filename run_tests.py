import os
from subprocess import Popen, PIPE, run

cov_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer", "*")
os.chdir("tests")
test_files = ["test_checks.py",
              "test_task.py",
              "test_epoch_loop.py",
              "test_modules.py",
              "test_model.py",
              "test_trainer_device.py",
              "test_trainer_models.py",
              "test_trainer_training_steps.py",
              "*test_trainer_metrics.py",
              "*test_trainer.py",
              "test_epoch.py",
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
try:
    for i, f in enumerate(test_files):
        # CHECK: What was this for?
        # if i+1 > 12:
        #     break
        append = "-a" if not i else ""
        print(f"Testing file {f}, {i+1} out of {len(test_files)}")
        p = Popen(f"python -m coverage run {append} --source=.. -m unittest {f}",
                  shell=True, stdout=PIPE, stderr=PIPE)
        x, y = p.communicate()
        out.append([f, x.decode("utf-8")])
        err.append([f, y.decode("utf-8")])
        if out[-1]:
            print(out[-1][1])
        if err:
            print(err[-1][1])
    run(f"coverage report -i --include={cov_path}", shell=True)
except KeyboardInterrupt:
    run(f"coverage report -i --include={cov_path}", shell=True)
