import os
import sys
from argparse import ArgumentParser
from subprocess import Popen, PIPE, run

cov_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer", "*")
os.chdir("tests")
g_test_files = ["test_checks.py",
                "test_task.py",
                "test_epoch_loop.py",
                "test_modules.py",
                "test_model.py",
                "test_autoloads.py",
                "test_trainer_device.py",
                "test_trainer_models.py",
                "test_trainer_training_steps.py",
                "*test_trainer_metrics.py",
                "*test_trainer.py",
                "test_epoch.py",
                "*test_trainer_state.py, big issues",
                "test_interfaces.py",
                "test_daemon.py",
                "test_daemon_auth.py",
                "test_daemon_http.py",
                "test_daemon_load_unload.py",
                "test_daemon_modules_http.py",
                "test_clone_archive.py"]


def run_tests(force, ft, lt):
    if ft and not ft.endswith(".py"):
        ft = ft.rstrip(".") + ".py"
    if lt and not lt.endswith(".py"):
        lt = lt.rstrip(".") + ".py"
    test_files = [f for f in g_test_files if force or not f.startswith("*")]
    first = 0
    last = len(test_files)
    if ft:
        first = test_files.index(ft)
    if lt:
        last = test_files.index(lt) + 1
    out = []
    err = []
    try:
        for i in range(first, last):
            f = test_files[i]
            append = "-a" if not i == first else ""
            print(f"Testing file {f}, {i+1} out of {last}")
            p = Popen(f"python -m coverage run {append} --source=.. -m unittest {f}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            x, y = p.communicate()
            out.append([f, x.decode("utf-8")])
            err.append([f, y.decode("utf-8")])
            if out[-1]:
                print(out[-1][1])
            if err:
                print(err[-1][1])
        run(f"coverage report -i --include={cov_path} --omit='*/_trainer*'", shell=True)
    except KeyboardInterrupt:
        run(f"coverage report -i --include={cov_path} --omit='*/_trainer*'", shell=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-l", "--list-tests", action="store_true",
                        help="List tests and exit")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force run all tests")
    parser.add_argument("-sf", "--single-file", default="",
                        help="Run tests from only a single file")
    parser.add_argument("-lf", "--list-test-files", action="store_true",
                        help="List test files and exit")
    parser.add_argument("-ft", "--first-test-file", default="",
                        help="First test file to run. " +
                        "Test files are run in order listed by `list-test-files`")
    parser.add_argument("-lt", "--last-test-file", default="",
                        help="Last test file to run. " +
                        "Test files are run in order listed by `list-test-files`")
    args = parser.parse_args()
    if args.list_tests:
        print("Does nothing")
        return
    if args.list_test_files:
        print("\n".join(g_test_files))
        print("\nTest files beginning with * are buggy and are skipped by default.\n" +
              "Use -f to force running all of them.")
        return
    if args.single_file:
        if not args.single_file.endswith(".py"):
            args.single_file = args.single_file.rstrip(".") + ".py"
        if args.single_file in g_test_files:
            f = args.single_file
            print(f"Testing a single file {f}")
            p = Popen(f"python -m coverage run --source=.. -m unittest {f}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate()
            if err:
                print(err.decode("utf-8"))
            if out:
                print(out.decode("utf-8"))
            run(f"coverage report -i --include={cov_path} --omit='*/_trainer*'", shell=True)
            sys.exit(0)
        else:
            print(f"No such file {args.single_file}")
            sys.exit(1)
    run_tests(args.force, args.first_test_file, args.last_test_file)


if __name__ == '__main__':
    main()
