from typing import List, Dict, Iterable, Any, Union, Tuple, Callable, Optional
from pathlib import Path
import torch

from pydantic import BaseModel as PydanticBaseModel, validator
from .models import (BaseModel, TorchModule, DataModel,
                     StepModel, TrainerModel, OptimizerModel)
from .models import LogLevels, When, TrainingType


class LogLevelParams(BaseModel):
    """Parameters for log levels

    Args:
        file: Output log level to file
        stream: Output stream log level to stdout
    """
    file: Union[LogLevels, str]
    stream: Union[LogLevels, str]

    @validator("file", "stream")
    def log_levels_must_be_uppercase(cls, v):
        if isinstance(v, str):
            return LogLevels(v.upper())


def gpus_str_must_be_specific(v: str) -> str:
    if v in {"auto", "parallel", "distributed"}:
        return v
    else:
        raise ValueError(f"Unable to parse the gpus str {v}")


def gpus_must_evaluate_to_list_of_int(v: Union[List[int], int, str, None]) ->\
        List[int]:
    if not isinstance(v, int) and not v:
        return []
    elif isinstance(v, str):
        try:
            retval = [*map(int, v.split(","))]
            return retval
        except Exception:
            raise ValueError(f"Unable to parse the gpus str {v}")
    elif isinstance(v, int):
        if v >= -1:
            return [v]
        else:
            return []
    else:
        return v


def gpus_could_be_list_of_int_or_str(v: Union[List[int], int, str, None]) ->\
        Union[List[int], str]:
    try:
        v = gpus_must_evaluate_to_list_of_int(v)
        return v
    except ValueError:
        return gpus_str_must_be_specific(v)


def model_must_be_torch_nn_module(v: type) -> type:
    def pred(x):
        return x.__module__.startswith("torch") and x.__qualname__.endswith("Module")
    if any(pred(x) for x in v.__bases__):
        return v
    else:
        raise ValueError("Not a torch.nn.Module")


class ModelParams(BaseModel):
    """Model configuration.

    Args:
        model: A :class:`TorchModule`
        optimizer: An :class:`Optimizer`
        params: The parameters to model
        gpus: gpus on which to place the model
        loaded: Whether the model is to be loaded into memory or not.

    """

    model: TorchModule
    optimizer: str
    params: Dict
    gpus: Union[List[int], int, str, None]
    loaded: Optional[bool]
    _validate_gpus = validator("gpus", allow_reuse=True)(gpus_could_be_list_of_int_or_str)
    _validate_model = validator("model", allow_reuse=True)(model_must_be_torch_nn_module)


class Optimizer(BaseModel):
    """Optimizer configuration.

    Args:
        function: The optimizer function
        params: The optimizer params

    """
    function: OptimizerModel
    params: Dict[str, Any]


class Criterion(BaseModel):
    """Criterion configuration.

    Args:
        function: The criterion function
        params: The criterion params

    """
    function: Callable
    params: Dict[str, Any]


class UpdateFunctionsParams(BaseModel):
    """Parameters config for :class:`UpdateFunctions.params`.

    In case individual function isn't provided and
    :attr:`UpdateFunctions.function` (of type :class:`~trainer.model.ModelStep`)
    is given.

    Args:
        models: A dict mapping model names referred by the :class:`ModelStep`
                to available :class:`~torch.nn.Module`.
        criteria_map: A dict mapping model and criteria names
        checks: A dict mapping model names and checks for each model.
        logs: What to log for :class:`ModelStep`.

    """
    models: List[str]
    criteria_map: Dict[str, str]
    checks: Dict[str, Callable[[Any], bool]]
    logs: List[str]


class UpdateFunctions(BaseModel):
    """Update functions config class.

    Need specification for either three separate update steps, all subclasses of
    :class:`StepModel` with `train` required and `val`, `test` optional.

    OR,

    A :class:`~trainer.model.ModelStep`
    step function with params :class:`UpdateFunctionsParams`.

    """
    function: Optional[StepModel]
    params: Optional[UpdateFunctionsParams]
    train: Optional[Callable]
    val: Optional[Callable]
    test: Optional[Callable]

    @validator("params")
    def params_cannot_be_none_if_function_is_not_none(cls, v, values):
        if v is None and values["function"] is not None:
            raise ValueError("params cannot be none if function is not none")
        if v is not None and ("function" not in values or values["function"] is None):
            raise ValueError("params cannot be not none if function is none")
        return v

    @validator("train")
    def either_function_is_none_or_rest_are(cls, v, values):
        if v is not None and values["function"] is not None:
            raise ValueError("Both train step and function cannot be given")
        else:
            return v
        if v is None and values["function"] is None:
            raise ValueError("Both train step and function cannot be given")
        else:
            return v


class CustomDataLoader(BaseModel):
    """Parameters to initialize a custom `DataLoader`.

    The :class:`CustomDataLoader` consists of three functions which return the
    three dataloaders directly. The `DataLoader` can be something like
    :class:`~torch.utils.data.dataloader.DataLoader`, but has to only provide
    :meth:`__getitem__` and :meth:`__len__` methods in practice.

    """
    train: Callable
    train_params: Dict
    val: Optional[Callable]
    val_params: Optional[Dict]
    test: Optional[Callable]
    test_params: Optional[Dict]


class DataParams(BaseModel):
    """Parameters to initialize the data and dataloaders.

    Args:
        name: Name of the dataset
        loader: If `loader` is given then `train`, `val`, `test` are not given.
                In that case, the data is fetched and initialized with a custom
                function which prepares the data returns a dataloader.
                See: :class:`CustomDataLoader`
        train: `train` split of the dataset
        val: `val` split of the dataset
        test: `test` split of the dataset

    `train` dataset is required in either of case of initializing the data and
    dataloaders with :class:`DataParams` or :class:`CustomDataLoader`. Rest can
    be optional and the trainer will run accordingly.

    """
    name: str
    loader: Optional[CustomDataLoader]
    train: Optional[DataModel]
    val: Optional[DataModel]
    test: Optional[DataModel]

    @validator("train")
    def either_train_is_none_or_loader_is_none(cls, v, values):
        if v is not None and values["loader"] is not None:
            raise ValueError("Both training data and loader cannot be given")
        else:
            return v
        if v is None and values["loader"] is None:
            raise ValueError("Both training data and loader cannot be given")
        else:
            return v

    @validator("val")
    def val_has_to_be_none_if_loader_is_not_none(cls, v, values):
        if v is not None and values["loader"] is not None:
            raise ValueError("Both validation data and loader cannot be given")
        return v

    @validator("test")
    def test_has_to_be_none_if_loader_is_not_none(cls, v, values):
        if v is not None and values["loader"] is not None:
            raise ValueError("Both test data and loader cannot be given")
        return v


class TorchDataLoaderParams(BaseModel):
    """A config for parameters of :class:`~torch.utils.data.dataloader.DataLoader`

    See :class:`~torch.utils.data.dataloader.DataLoader` for details.

    """
    batch_size: int = 1
    shuffle: bool = False
    sampler: Optional[Callable] = None
    batch_sampler: Optional[Callable] = None
    num_workers: Optional[int] = 0
    collate_fn: Optional[Callable] = None
    pin_memory: Optional[bool] = False
    drop_last: Optional[bool] = False
    timeout: Optional[int] = 0
    worker_init_fn: Optional[Callable] = None
    multiprocessing_context: Optional[Callable] = None


class DataLoaderParams(BaseModel):
    """Parameters for the dataloaders

    """
    train: TorchDataLoaderParams
    val: Optional[TorchDataLoaderParams]
    test: Optional[TorchDataLoaderParams]


class TrainerParams(BaseModel):
    """A config for parameters :class:`~trainer.trainer`

    Args:
        gpus: A list or comma separated string of gpus
        cuda: Whether to use cuda (gpus) or not
        seed: Seed with which torch will be initialized
        resume_best: Resume from the previously best state
        resume_dict: Resume from given weights
        init_weights: Initialize the model from given weights
                      but don't resume from state
        resume: Whether to resume or not.
        test_frequency: How often in terms of `epoch` to run the test loop
        check_func: FIXME Don't recall what this does
        max_epochs: Maximum epochs to train
        load_all: Load all models into memory
                  If false, only the models which have `load == True` will be loaded.
        max_iterations: Maximum iterations to train if training with `iterations`
        training_steps: Training steps (usually `[train, val, test]`)
        training_type: Type of training loop `iterations` or `epoch`

    The behaviour of resuming from a previous state depends on both `resume` and
    the params given. In case `init_weights` are given then `resume` need not be
    `True`, the weights are loaded into the corresponsding models and the
    trainer starts from beginning.

    If however, `resume` is `True` then `resume_bset` is checked first and
    `trainer._resume_path` is set to that. Otherwise if `resume_dict` is
    given, then the state (including model weights) is resumed from there.

    Otherwise we resume from the last checkpoint.

    """
    gpus: Union[List[int], int, str, None]
    cuda: bool
    seed: int
    resume_best: Optional[Path]
    resume_dict: Optional[Path]
    init_weights: Optional[Path]
    resume: bool
    test_frequency: Optional[int] = 5
    check_func: Optional[Callable]
    max_epochs: Optional[int]
    # NOTE: I'm validating load_all here with ModelParams above. Is that good
    #       design?
    #       Maybe if nothing is loaded then load nothing?
    load_all: bool
    max_iterations: Optional[int]
    training_steps: List[str]
    training_type: TrainingType

    _validate_gpus = validator("gpus", allow_reuse=True)(gpus_must_evaluate_to_list_of_int)

    @validator("resume_dict")
    def resume_weights_must_be_path_and_exist(cls, v: Union[Path, str, None]) -> Optional[Path]:
        if not v:
            return None
        else:
            if Path(v).exists():
                return Path(v)
            else:
                raise AttributeError(f"Path {v} doesn't exist")

    @validator("resume_best")
    def resume_best_must_be_path_and_exist(cls, v: Union[Path, str, None]) -> Optional[Path]:
        if not v:
            return None
        else:
            if Path(v).exists():
                return Path(v)
            else:
                raise AttributeError(f"Path {v} doesn't exist")

    @validator("resume")
    def only_one_of_resume_or_init_weights_can_be_given(cls, v, values) -> bool:
        if "init_weights" in values and values["init_weights"]:
            raise ValueError("Only one of init_weights or resume can be given")
        elif not v:
            if (("resume_best" in values and values["resume_best"])
                    or ("resume_dict" in values and values["resume_dict"])):
                return True
            return v
        else:
            return v

    @validator("training_steps")
    def training_steps_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("training_steps can't be empty")
        return v

    @validator("training_type")
    def training_with_iterations_should_have_max_iterations_but_not_max_epochs(cls, v, values):
        if v == "iterations":
            if not values["max_iterations"]:
                raise ValueError("max_iterations cannot be" +
                                 f" {values['max_iterations']} with iterations")
            if "max_epochs" in values and values["max_epochs"]:
                raise ValueError("training with iterations cannot have" +
                                 f" max_epochs {values['max_epochs']}")
            values["max_epochs"] = 0
        return v

    @validator("training_type")
    def training_with_epochs_should_have_max_epochs_but_not_max_iterations(cls, v, values):
        if v == "epoch":
            if not values["max_epochs"]:
                raise ValueError("max_epochs cannot be" +
                                 f" {cls.max_epochs} while training with epochs")
            if "max_iterations" in values and values['max_iterations']:
                raise ValueError("training with epochs cannot have" +
                                 f" max_iterations {values['max_iterations']}")
            values["max_iterations"] = 0
        return v

    # @validator("training_steps")
    # def training_with_iterations_should_not_have_other_steps(cls, v):
    #     if "iterations" in v:
    #         if len(v) > 1:
    #             raise ValueError("train, val, test or other steps" +
    #                              " cannot be included with iterations")
    #     return v


class Metric(BaseModel):
    """Custom metrics specification

    Args:
        steps: Name of steps (train, val, test) on which the metric is collected
        function: function which gathers the metric.
        inputs: Inputs required to the function
        when: batch or epoch

    """
    steps: List[str]
    function: Optional[Callable]
    inputs: List[Any]                # FIXME: input_variables_to_function
    when: When

    @validator("when")
    def when_should_be_upper_case(cls, v):
        if isinstance(v, str):
            return When(v.upper())
        else:
            return v


class Config(PydanticBaseModel):
    """Config class.

    Args:
        model_params: Dictionary of model names and parameters.
                      paramters is an instance of :class:`ModelParams`
        trainer_params: Instance of :class:`TrainerParams`
        log_levels: Instance of :class:`LogLevelParams`
        optimizers: Dictionary mapping model names to instance of :class:`Optimizer`
        criteria: Dictionary mapping criteria names to instance of :class:`Criterion`
        update_functions: Instance of :class:`UpdateFunctions`
        data_params: Instance of :class:`DataParams`
        dataloader_params: Instance of :class:`DataLoaderParams`
        extra_metrics: Dictionary mapping metric names to instances of :class:`Metric`
        data_dir: Path where all the trainer state will be stored
        production: (deprecated) Whether to run in production mode.
    """
    model_params: Dict[str, ModelParams]
    trainer_params: TrainerParams
    log_levels: LogLevelParams
    optimizers: Dict[str, Optimizer]
    criteria: Dict[str, Criterion]
    update_functions: UpdateFunctions
    data_params: DataParams
    extra_metrics: Dict[str, Metric]
    dataloader_params: DataLoaderParams
    # NOTE: rest are supplied by interface
    data_dir: Path
    production: bool

    @validator("model_params")
    def model_params_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("model_params can't be empty")
        return v

    @validator("criteria")
    def all_criteria_functions_must_have_forward_attr(cls, v):
        for k, x in v.items():
            if not hasattr(x.function, 'forward'):
                raise AttributeError(f"{k}, {x} has no attribute 'forward'")
        return v

    class Config:
        validate_assignment = True

    # TODO: `assert all(x in self._update_functions)` should be in update_functions
    # @validator("update_functions")
    # def training_with_iterations_should_not_have_other_steps(cls, v):
    #     if "iterations" in v:
    #         if len(v) > 1:
    #             raise ValueError("train, val, test or other steps cannot be included with iterations")
    #     assert all(x in self._update_functions
    #                for x in self._trainer_params["training_steps"]),\
    #                    "Steps in update_functions and training_steps should match"

    # def not_sure_trainer_params(cls):
    #     if not self._have_resumed:
    #         self._logd("Ignoring resume_params in while resuming")
    #         assert "init_weights" in self._trainer_params
    #         assert "resume_dict" in self._trainer_params
    #     if "iterations" in self._trainer_params["training_steps"]:
    #         # NOTE: Rest (test_every_k_iterations etc.) is checked in init_dataloaders
    #         # CHECK: Though should it be? Then that should be a training roadmap
    #         assert "hooks_run_iter_frequency" in self._trainer_params, "Training with iterations" +\
    #             " requires hooks_run_iter_frequency"
    #         self._hooks_run_iter_frequency = self._trainer_params["hooks_run_iter_frequency"]
    #         assert self._hooks_run_iter_frequency <= self._max_iterations, "hooks_run_iter_frequency" +\
    #             " can be no more than max_iterations"
