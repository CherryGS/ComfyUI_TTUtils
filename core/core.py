import json
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Iterable, Literal, Generic, TypeVar, Type
from typing_extensions import Unpack, assert_type
import torch


class JsonEncoder(json.JSONEncoder):
    def encode(self, o):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {"__tuple__": True, "items": [hint_tuples(i) for i in item]}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}

            return item

        return super(JsonEncoder, self).encode(hint_tuples(o))


def hinted_tuple_hook(obj):
    if "__tuple__" in obj:
        return tuple(obj["items"])
    else:
        return obj


class ParamType(str, Enum):
    clip = "CLIP"
    cond = "CONDITIONING"
    model = "MODEL"
    vae = "VAE"
    latent = "LATENT"
    image = "IMAGE"
    mask = "MASK"
    net = "CONTROL_NET"
    int = "INT"
    float = "FLOAT"
    string = "STRING"
    select = "SELECT"
    bool = "BOOLEAN"
    custom = "CUSTOM"

    def __str__(self) -> str:
        return f"{self.value}"


InputType = dict[
    Literal["required"] | Literal["hidden"] | Literal["optional"],
    dict[
        str,
        tuple[ParamType, dict[str, Any] | None] | tuple[list[str]],
    ],
]


@dataclass
class Param:
    key: str
    _type: ParamType

    def __init__(self, key, t, **kwargs):
        self.key = key
        self._type = t
        self.__dict__.update(kwargs)

    def props(self, ignore: Iterable = []):
        ignore = set(ignore) | set(["key", "_type"])
        return {
            i: j
            for i, j in filter(
                lambda x: x[0] not in ignore and x[1] is not None,
                self.__dict__.items(),
            )
        }

    def get_type(self):
        """
        获取该参数的类型，返回一个和该类型相等的字符串
        """
        return str(self._type)

    def res(self):
        p = self.props()
        if p:
            return (self.get_type(), p)
        else:
            return (self.get_type(),)

    def schema(self):

        p = JsonEncoder().encode(self.res())
        return str(json.loads(p, object_hook=hinted_tuple_hook))


@dataclass
class Custom(Param):
    _type: ParamType = field(default=ParamType.custom, init=False)
    type: Type[object] | str

    def __init__(self, key: str, type: Type[object] | str):
        super().__init__(key, ParamType.custom)
        self.type = type

    def props(self, ignore: Iterable = []):
        return super().props(["type"])

    def get_type(self):
        return self.type if isinstance(self.type, str) else self.type.__name__


@dataclass
class Select(Param):
    _type: ParamType = field(default=ParamType.select, init=False)
    select: list[str] = field(default_factory=list, kw_only=True)
    default: str | None = field(default=None)


@dataclass
class Clip(Param):
    _type: ParamType = field(default=ParamType.clip, init=False)


@dataclass
class Int(Param):
    _type: ParamType = field(default=ParamType.int, init=False)

    default: int = field(default=0)

    min: int = field(default=0)
    """Minimum value"""
    max: int = field(default=4096)
    """Maximum value"""
    step: int = field(default=64)
    """Slider's step"""
    display: Literal["number", "slider"] = field(default="number")
    """Cosmetic only: display as "number" or "slider"""

    def hints(self) -> int: ...


@dataclass
class Float(Param):
    _type: ParamType = field(default=ParamType.float, init=False)

    default: float = field(default=0.0)
    min: float = field(default=0.0)
    """Minimum value"""
    max: float = field(default=10.0)
    """Maximum value"""
    step: float = field(default=0.01)
    """Slider's step"""
    round: float = field(default=0.001)
    """The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding."""
    display: Literal["number", "slider"] = field(default="number")
    """Cosmetic only: display as "number" or "slider"""


@dataclass
class Bool(Param):
    _type: ParamType = field(default=ParamType.bool, init=False)

    default: bool = field(default=True)


@dataclass
class String(Param):
    _type: ParamType = field(default=ParamType.string, init=False)

    forceInput: bool = field(default=False)
    multiline: bool = field(default=False)
    """True if you want the field to look like the one on the ClipTextEncode node"""
    default: str = field(default="")

    def hints(self) -> str: ...


@dataclass
class Conditioning(Param):
    _type: ParamType = field(default=ParamType.cond, init=False)


@dataclass
class Image(Param):
    _type: ParamType = field(default=ParamType.image, init=False)


class Node:

    required: Iterable[Param]
    hidden: Iterable[Param]
    optional: Iterable[Param]
    output: Iterable[Param]

    OUTPUT_NODE: bool = True
    FUNCTION: str
    CATEGORY: str

    def __init_subclass__(cls, *args, **kwargs):
        """
        控制 `RETURN_TYPES` 和 `RETURN_NAMES` 的生成
        """
        super().__init_subclass__(*args, **kwargs)

        # if "__iter__" not in cls.__dict__:
        #     cls.output = ()

        RETURN_TYPES = tuple([i.get_type() for i in (cls.output if cls.output else [])])
        RETURN_NAMES = tuple(
            [
                (i.key if i.key else i.get_type())
                for i in (cls.output if cls.output else [])
            ]
        )

        if cls.FUNCTION == None:
            cls.FUNCTION = "run"
        if cls.CATEGORY == None:
            cls.CATEGORY = cls.__name__
        if len(RETURN_TYPES):
            cls.OUTPUT_NODE = False

        assert len(RETURN_NAMES) == len(RETURN_TYPES)
        setattr(cls, "RETURN_TYPES", RETURN_TYPES)
        setattr(cls, "RETURN_NAMES", RETURN_NAMES)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {i.key: i.schema() for i in cls.required},
            "hidden": {i.key: i.schema() for i in cls.hidden},
            "optional": {i.key: i.schema() for i in cls.optional},
        }

    @classmethod
    def IS_CHANGED(cls, *args):
        return ""


__all__ = [
    "Bool",
    "Clip",
    "Conditioning",
    "Custom",
    "Float",
    "Image",
    "Int",
    "Node",
    "String",
    "Select",
]


if __name__ == "__main__":
    from rich import print

    # p = Custom("option", BaseProp)
    # print(p.schema())

    p = (Int("t"), Int("q"), String("r"))
    # do some typing for q with p then pylance could get
    q: tuple[int, int, str]

    # def test(*args):
    #     assert_type(args, p)

    # print(Int("t").hints())
    # print((1, None) == (1,))
