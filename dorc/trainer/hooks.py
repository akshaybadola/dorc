import numpy as np
from ..helpers import _log_metrics_for_step, get_proxy_dataloader


def val_post_epoch_hook(cls):
    cls.validate_post_epoch_hook(cls)


def validate_post_epoch_hook(cls):
    cls._logd("Running post epoch validate hook")
    if cls.val_loader is not None:
        cls.validate(cls._epoch_runner)
    else:
        cls._logi("No val loader. Skipping")


def test_post_epoch_hook(cls):
    cls._logd("Running post epoch test hook")
    if (cls.epoch+1) % cls.test_frequency == 0:
        if cls.test_loader is not None:
            cls.test(cls._epoch_runner)
        else:
            cls._logi("No test loader. Skipping")


def save_history_post_epoch_hook(cls):
    cls._logd("Running save history state post epoch hook")
    cls._save(cls._save_path_with_epoch)


def save_best_post_epoch_hook(cls):
    cls._logd("Running save best post epoch hook")
    cls.check_and_save()


def save_checkpoint_post_epoch_hook(cls):
    cls._logd("Running post epoch save hook")
    cls._save(cls._checkpoint_path)
    cls.check_and_save()


def gather_metrics(cls, runner):
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


# START: Stateless Functions
# DONE: There should be a separate definition of "steps" there where it
#       could be {train, val, test} or simply iterations
#       NOTE: Now iterations are also handled.
def _log_metrics(cls):
    if "iterations" in cls.trainer_params.training_steps:
        update_key = cls.iterations / cls._hooks_run_iter_frequency
        key_name = "iterations chunk"
    else:
        update_key = cls.epoch
        key_name = "epoch"
    log_func = cls._logd
    for step in cls._metrics:
        if getattr(cls, step + "_loader"):
            _log_metrics_for_step(step, key_name, getattr(cls, step + "_loader"),
                                  cls._metrics[step], update_key, log_func)
        else:
            cls._logd(f"No dataloader for {step}")


# FIXME: TRAINING_STEPS
# NOTE: For this a sample function has to be defined
def _log_samples(cls, fraction=0.01):
    """For a few randomly selected datapoints, log the datapoint_name and
    corresponding model output
    """
    if "iterations" in cls.trainer_params.training_steps:
        raise NotImplementedError
    for step in cls.trainer_params.training_steps:
        dataset = getattr(cls, step + "_loader").dataset
        loader = get_proxy_dataloader(dataset,
                                      cls.dataloader_params[step],
                                      10,  # seems like a good number
                                      cls.logger)
        step_func = getattr(cls, step + "_step_func")
        # reset, launch each in a separate thread, wait for finish
        # CHECK: Is this a good idea? Maybe separate runner from epoch
        getattr(cls._epoch_runner, "run_" + step)(step_func, loader, True)


# TODO: A lot of these controls and methods which depend on params will
#       have to be rewritten.
# TODO: multiplier can be a trainer_param
# FIXME: Annealing may depend on extra_metrics
# TODO: Annealing can be an external function like CheckFunc
def anneal_lr(cls, multiplier=.9):
    cls._logi("Annealing Learning Rate")
    check_losses = [loss[2] for loss in cls.losses if loss[0] == cls.save_on]
    if len(check_losses) >= 2:
        delta = check_losses[-2] - check_losses[-1]
        if delta < .01 * check_losses[-2]:
            for param_group in cls.optimizer.param_groups:
                param_group['lr'] *= multiplier
            cls._logi("Annealing...")


def dump_state_post_epoch_hook(cls):
    "Dump everything except weights"
    cls._dump_state()


# TODO: I should log some image names and output text also
#       That should be there in _log_samples
def log_post_epoch_hook(cls):
    """Summarizes and log the metrics/losses etc post epoch
    items_to_log_dict can be accessed and modified by the user

    :returns: None
    :rtype: None

    """
    cls._logi("Running post epoch log hook")
    # But these are certain transformations I'm doing to metrics
    for k, v in cls._items_to_log_dict.items():
        getattr(cls, "_log_" + k)()



def update_metrics_post_epoch_hook(cls):
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
