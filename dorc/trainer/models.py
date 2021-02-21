from typing import List, Dict, Iterable, Any, Union, Tuple, Callable as TCallable, Optional
from pydantic import BaseModel as PydanticBaseModel, validator
import torch
from enum import Enum
from numbers import Number
import numpy

from ..spec.models import add_nullable, remove_attr, remove_prop_titles
from . import model


class LogLevels(str, Enum):
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARN = "WARN"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


# FIXME: Should be defined dynamically
class When(str, Enum):
    """When to do something (maybe run some hook)

    Batch or Epoch

    """
    BATCH = "BATCH"
    EPOCH = "EPOCH"


class TrainingType(str, Enum):
    """What type of training loop are we running?

    `iterations` or `epoch`

    """
    epoch = "epoch"
    iterations = "iterations"


class DataEnum(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: PydanticBaseModel) -> None:
            add_nullable(schema, model)
            remove_prop_titles(schema, model)


class NumpyNDArray(numpy.ndarray):
    """Config for :class:`~numpy.ndarray`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> Any:
        # validate data...
        return v

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="numpy.ndarray"
        )


class DataModel(BaseModel):
    """Config for :class:`~torch.utils.data.Data`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> Any:
        if iter(v) and len(v) and not isinstance(v, str):
            return v
        else:
            raise TypeError((f"{v} of type {type(v)} is not a Sequence"))

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="torch.utils.data.Data"
        )


class OptimizerModel(torch.optim.Optimizer):
    """Config for :class:`~torch.optim.Optimizer`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> torch.optim.Optimizer:
        if type(v) == torch.optim.Optimizer or\
           issubclass(v, torch.optim.Optimizer):
            return v
        else:
            raise TypeError(f"{v} of type {type(v)} is not of type {torch.optim.Optimizer}")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="torch.optim.Optimizer"
        )


# class OptimizerModel(torch.optim.optimizer.Optimizer):
#     """Config for :class:`~torch.optim.optimizer.Optimizer`

#     A simple parser and validator that only checks the type.

#     """
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     @classmethod
#     def validate(cls, v: Any) -> torch.optim.optimizer.Optimizer:
#         if isinstance(v, torch.optim.optimizer.Optimizer) or\
#            issubclass(type(v), torch.optim.optimizer.Optimizer):
#             return v
#         else:
#             raise TypeError("Expected type of torch.optim.optimizer.Optimizer")

#     @classmethod
#     def __modify_schema__(cls, field_schema):
#         field_schema.update(
#             type="type",
#             default="torch.optim.optimizer.Optimizer"
#         )


class TorchModule(torch.nn.Module):
    """Config for :class:`~torch.nn.Module`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> torch.nn.Module:
        if type(v) == torch.nn.Module or issubclass(v, torch.nn.Module):
            return v
        else:
            raise TypeError(f"{v} of type {type(v)} is not a type of torch.nn.Module")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="torch.nn.Module"
        )


class TorchTensor(torch.Tensor):
    """Config for :class:`~torch.Tensor`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor) or issubclass(type(v), torch.Tensor):
            return v
        else:
            raise TypeError(f"{v} of type {type(v)} is not a type of {torch.Tensor}")


    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="torch.Tensor"
        )


class TrainerModel(model.Model):
    """Config for :class:`~trainer.trainer.model.Model`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> model.Model:
        # validate data...
        if isinstance(v, model.Model) or issubclass(type(v), model.Model):
            return v
        else:
            raise TypeError(f"{v} of type {type(v)} is not a type of {model.Model}")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="trainer.trainer.model.Model"
        )


class StepModel(model.ModelStep):
    """Config for :class:`~trainer.trainer.ModelStep`

    A simple parser and validator that only checks the type.

    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> model.ModelStep:
        if type(v) == model.ModelStep or issubclass(v, model.ModelStep):
            return v
        else:
            raise TypeError(f"{v} of type {type(v)} is not a type of {model.ModelStep}")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="type",
            default="trainer.trainer.model.ModelStep"
        )


class AdhocEvalParams(BaseModel):
    """Params spec for calling an adhoc evaluation or run.

    Args:
        epoch: :class:`int` num or current,
        num_or_fraction: Either `int` or `float`. If `float` then the
                         fraction of the data points from the dataset
                         are sampled, else the number of points.
        concurrent: Whether to pause the current loop or run concurrently
        seed: Set the random seed if given.
        data: One of train val or test
        callback: name of callback function

    `data` Can be other than train, val or test, though it's not implemented
    right now.

    `callaback` can be any one of the existing user functions or available
    functions in :class:`Trainer`

    """
    epoch: Union[int, str]
    num_or_fraction: float
    concurrent: bool
    seed: Optional[int]
    data: DataEnum
    callback: str

    @validator("epoch")
    def epoch_must_be_no_more_than_current(cls, v):
        if isinstance(v, str):
            return "current"
        elif isinstance(v, int):
            return v
        else:
            raise TypeError(f"Invalid type {type(v)}")

    @validator("num_or_fraction")
    def num_or_fraction_must_be_greater_than_zero(cls, v):
        if not isinstance(v, Number):
            raise TypeError(f"Invalid type {type(v)}")
        elif v <= 0:
            raise ValueError(f"num_or_fraction must be greater than 0")
        elif v >= 1:
            return int(v)


class CallableModel(TCallable):
    """Config for :class:`Callable`

    A simple parser and validator that only checks the type.

    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> Any:
        if callable(v):
            return v
        else:
            raise TypeError(f"Invalid type {type(v)}")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="function",
            default="Some function"
        )


class StateEnum(str, Enum):
    lite = "lite"
    full = "full"
    complete = "complete"


class TrainerState(BaseModel):
    """Config for the state returned by :meth:`Trainer._get_state`"""
    mode: StateEnum
    epoch: int
    max_epochs: int
    given_name: str
    iterations: int
    max_iterations: int
    saves: Union[List[str], Dict[str, str]]  # name or dict of save file
    devices: Dict[str, List[int]]
    allocated_devices: List[int]
    active_models: Dict[str, str]
    loaded_models: Union[List[str], Dict[str, Any]]
    models: Union[List[str], Dict[str, Any]]
    metrics: Dict[str, Dict[str, Any]]
    extra_metrics: Dict[str, Dict[str, Any]]
    data: str
    funcs: List[str]            # Dict[str, Dict[str, Any]]
    extra_items: Optional[Dict]


class Return(BaseModel):
    """Config for a return value from a remote `method`.

    Args:
        status: The status of the request
        message: The associated message

    """
    status: bool
    message: str


class ReturnBinary(BaseModel):
    """Config for a return value with Image data from a remote `method`.

    Args:
        status: The status of the request
        image: The requested image
        mimetype: The mimetype for the image

    """
    status: bool
    data: bytes
    mimetype: str


class ReturnExtraInfo(BaseModel):
    """Config for a return value with extra information data from a remote `method`.

    Args:
        status: The status of the request
        message: The associated message
        data: The data object

    """
    status: bool
    message: str
    data: Dict
