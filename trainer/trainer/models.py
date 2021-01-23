from typing import List, Dict, Iterable, Any, Union, Tuple, Callable, Optional
from pydantic import BaseModel as PydanticBaseModel
from ..spec.models import add_nullable, remove_attr, remove_prop_titles
import torch
import numpy
from . import model


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: PydanticBaseModel) -> None:
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
        if iter(v) and len(v):
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
