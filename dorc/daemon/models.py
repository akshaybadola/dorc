from typing import List, Dict, Iterable, Any, Union, Tuple, Callable, Optional
from ..spec.models import add_nullable, remove_prop_titles, BaseModel as SpecModel
from ..trainer.models import TrainerState


class BaseModel(SpecModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model) -> None:
            add_nullable(schema, model)


class Session(BaseModel):
    loaded: bool
    port: Optional[int]
    state: TrainerState
    finished: bool


class Sessions(BaseModel):
    default: Union[str, Dict[str, Session]]


class CreateSessionModel(BaseModel):
    """Session creation specification.

    Args:
        name: Name of the session
        overrides: Configuration overrides over the primary session config
        config: A :class:`dict` object conforming to :class:`~config.Config`
        saves: A :class:`dict` of save file names to copy in the savedir
               The value of the saves is a base64 encoded bytes object.

    :code:`config` can be either parsed from json or dynamically loaded from a
    given python or zip file.

    """
    name: str
    overrides: Optional[Dict]
    config: Union[Dict, str]
    load: bool
    saves: Optional[Dict[str, str]]


class SessionMethodResponseModel(BaseModel):
    status: bool
    message: str
    task_id: int
