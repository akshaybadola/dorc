from typing import List, Dict, Iterable, Any, Union, Tuple, Callable, Optional
from ..trainer import config
from ..spec.models import BaseModel as SpecModel
import numpy


class BaseModel(SpecModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any]) -> None:
            for prop in schema.get('properties', {}).values():
                prop.pop('title', None)


class CreateSessionModel(BaseModel):
    """Session creation spec.

    Args:
        name: Name of the session
        overrides: Configuration overrides over the primary session config
        config: A :class:`dict` (parsed from json) object conforming to :class:`~config.Config`
        saves: A :class:`dict` of save file names to copy in the savedir
               The value of the saves is a base64 encoded bytes object.

    """
    name: str
    overrides: Optional[Dict]
    config: Dict
    saves: Optional[Dict[str, str]]
