import numpy as np


def validate_func(cls):
    """Run the :code:`validation` loop."""
    cls._logd("Running post epoch validate")
    if cls.val_loader is not None:
        cls.validate(cls._epoch_runner)
    else:
        cls._logi("No val loader. Skipping")


def test_func(cls):
    """Run the :code:`test` loop."""
    cls._logd("Running post epoch test")
    if (cls.epoch+1) % cls.test_frequency == 0:
        if cls.test_loader is not None:
            cls.test(cls._epoch_runner)
        else:
            cls._logi("No test loader. Skipping")


def save_history_func(cls):
    """Save the history of training so far."""
    # TODO: What is history?
    raise NotImplementedError


def save_best_func(cls):
    """Update the save marked with :code:`best_epoch`."""
    cls._logd("Running save best")
    cls.check_and_save()


def save_checkpoint_func(cls):
    """Save the current state and :code:`best_epoch`."""
    cls._logd("Running save checkpoint")
    cls._save(cls._checkpoint_path)
    cls.check_and_save()


def log_func(cls):
    """Log items according to :code:`cls.artefacts_to_log`"""
    for x, f in cls.artefacts_to_log:
        cls._logi(f"Logging {x}")
        f(cls)


def gather_metrics_func_with_args(cls, runner):
    """Gather metrics from a particular task runner.

    runner: task_runner
    """
    retval = {}
    for step in cls.trainer_params.training_steps:
        if step != "iterations" and step in cls._metrics:
            metric_names = cls._metrics[step]
            retval[step] = {}
            retval[step]["num_datapoints"] = runner.total_samples[step]
            for m in metric_names:
                all_vals = [x[3] for x in runner.batch_vars
                            if x[0] == step and x[2] == m]
                if len(all_vals):
                    retval[step][m] = np.mean(all_vals)
    return retval


def update_metrics_func(cls):
    """Update the metrics being recorded.

    Gather the tuples from :attr:`epoch_runner` and update :attr:`metrics`

    """
    cls._logd("Updating the metrics")
    if "iterations" in cls.trainer_params.training_steps:
        update_key = cls.iterations / cls._hooks_run_iter_frequency
    else:
        update_key = cls.epoch
    for step in cls._metrics:
        metric_names = cls._metrics[step]
        cls._metrics[step]["num_datapoints"][update_key] =\
            cls._epoch_runner.total_samples[step]
        for m in metric_names:
            all_vals = [x[3] for x in cls._epoch_runner.batch_vars
                        if x[0] == step and x[2] == m]
            if len(all_vals):
                cls._metrics[step][m][update_key] = np.mean(all_vals)
