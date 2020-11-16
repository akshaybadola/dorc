import os
import torch


def _check_model_params(cls):
    assert isinstance(cls._model_params, dict), "_model_params has to be a dict"
    assert len(cls._model_params) > 0, "_model_params can't be empty"
    assert all(isinstance(x, dict) for x in cls._model_params.values()),\
        "all the _model_params should be dict"
    assert isinstance(cls._model_defs, dict),\
        "_model_defs has to be a dict"
    assert len(cls._model_defs) > 0, "_model_defs can't be empty"
    assert all(callable(x["model"]) for x in cls._model_defs.values()),\
        "_model_defs value have to be callabes"


def _check_trainer_params(cls):
    """Checks trainer params

    :returns: None
    :rtype: None

    """
    # optimizer: function now has to be of type Optimizer. Also criterion
    # has to have attribute forward.
    assert all(isinstance(x, dict)
               for x in [cls._trainer_params, cls._criteria_params,
                         cls._optimizer_params])
    assert all(len(x) > 0 for x in [cls._trainer_params, cls._criteria_params,
                                    cls._optimizer_params])
    assert all(isinstance(x, dict) and callable(x["function"])
               and hasattr(x["function"], "forward")
               for x in cls._criteria_params.values())
    assert all(isinstance(x, dict) and issubclass(x["function"], torch.optim.Optimizer)
               for x in cls._optimizer_params.values())
    # TODO: This is no longer relevant
    if "anneal" in cls._trainer_params:
        assert all(x in cls._trainer_params
                   for x in ["anneal_lr_after", "anneal_lr_factor", "anneal_lr_on"])
    if "test_frequency" not in cls._trainer_params:
        cls.test_frequency = 5
    assert "gpus" in cls._trainer_params
    assert "cuda" in cls._trainer_params
    assert "check_func" in cls._trainer_params
    cls._check_func = cls._trainer_params["check_func"]
    if not cls._have_resumed:
        cls._logd("Ignoring resume_params in while resuming")
        assert "init_weights" in cls._trainer_params
        assert "resume_weights" in cls._trainer_params
    assert "training_steps" in cls._trainer_params
    assert all(x in ["train", "val", "test", "iterations"]
               for x in cls._trainer_params["training_steps"])
    if "iterations" in cls._trainer_params["training_steps"]:
        # NOTE: Rest (test_every_k_iterations etc.) is checked in init_dataloaders
        # CHECK: Though should it be? Then that should be a training roadmap
        assert "max_iterations" in cls._trainer_params,\
            "training with iterations must provide max_iterations parameter"
        assert len(cls._trainer_params["training_steps"]) == 1,\
            "train, val, test or other steps cannot be included with iterations"
        assert "train" in cls._update_functions, "At least train update_function has to be present"
        assert "hooks_run_iter_frequency" in cls._trainer_params, "Training with iterations" +\
            " requires hooks_run_iter_frequency"
        cls._max_epochs = 0
        cls._trainer_params["max_epochs"] = 0
        cls._max_iterations = cls._trainer_params["max_iterations"]
        cls._hooks_run_iter_frequency = cls._trainer_params["hooks_run_iter_frequency"]
        assert cls._hooks_run_iter_frequency <= cls._max_iterations, "hooks_run_iter_frequency" +\
            " can be no more than max_iterations"
    else:
        assert "max_epochs" in cls._trainer_params, "max_epochs not in trainer params"
        cls._max_epochs = cls._trainer_params["max_epochs"]
        cls._trainer_params["max_iterations"] = 0
        cls._max_iterations = 0
        assert all(x in cls._update_functions
                   for x in cls._trainer_params["training_steps"]),\
            "Steps in update_functions and training_steps should match"


# assert anneal_lr_on in some metric
# check metric decease or increase?
def _check_resume_or_init_weights(cls):
    if ("init_weights" in cls._trainer_params and cls._trainer_params["init_weights"]):
        assert (not cls._trainer_params["resume_best"] and
                not cls._trainer_params["resume_weights"]),\
                "Cannot initialize from weights and resume from save data"
    if cls._trainer_params["init_weights"]:
        cls._logw("Warning! Loading weights directly to model")
        load_state = torch.load(cls._trainer_params["init_weights"])
        for name in cls._models.names:
            cls._models.load_weights(name, load_state["models"][name])
    elif cls._trainer_params["resume"]:  # implies resume from somewhere
        if cls._trainer_params["resume_best"]:
            # try to find and resume best weights
            cls._loge("Resume from best is not yet implemented")
            cls._resume_path = None
        elif cls._trainer_params["resume_weights"]:
            if os.path.exists(cls._trainer_params["resume_weights"]):
                cls._resume_path = cls._trainer_params["resume_weights"]
            else:
                cls._logw("Given resume weights do not exist")
                cls._resume_path = None  # set appropriate path
        else:
            if os.path.exists(cls._checkpoint_path):
                cls._logi("Checkpoint exists. Will resume from there")
                cls._resume_path = cls._checkpoint_path
            else:
                cls._logi("No checkpoint found. Will train from beginninng")
                cls._resume_path = None
    else:
        # Don't resume
        cls._resume_path = None
    if cls._trainer_params["resume"] and cls._resume_path:
        cls._logi(f"Resuming from {cls._resume_path}")
        cls._resume_from_path(cls._resume_path)


# TODO: What if there are other keys besides train/val/test
def _check_data_params(cls):
    """If cls._data is None, then data is extracted from the dataloader later.

    :returns: None
    :rtype: None

    """
    assert all([x in cls._dataloader_params for x in ["train", "val", "test"]])
    assert cls._dataloader_params["train"] is not None
    if cls._data is None:
        for x in ["train", "val", "test"]:
            if cls._dataloader_params[x] is not None:
                assert "function" in cls._dataloader_params[x],\
                    "dataloader_params for data subset cannot be None if data is None"
    else:
        assert all([x in cls._data for x in ["train", "val", "test"]])
        assert cls._data["train"] is not None, "Training data cannot be None"
