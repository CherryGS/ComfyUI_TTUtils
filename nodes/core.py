import json
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Iterable, Literal
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


class PropType(str, Enum):
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


propA = Literal[PropType.int, PropType.float, PropType.string]

InputType = dict[
    Literal["required"] | Literal["hidden"] | Literal["optional"],
    dict[
        str,
        tuple[PropType, dict[str, Any] | None] | tuple[list[str]],
    ],
]


@dataclass
class BaseProp:
    _type: PropType
    key: str

    def filter_props(self, ignore: Iterable = []):
        ignore = set(ignore) | set(["key"])
        return {
            i: j
            for i, j in filter(
                lambda x: x[0] not in ignore and x[1] is not None,
                self.__dict__.items(),
            )
        }

    def get_type(self):
        return str(self._type).lower()

    def res(self):
        p = self.filter_props()
        if p:
            return (self.get_type(), p)
        else:
            return (self.get_type(),)

    def schema(self) -> str:

        p = JsonEncoder().encode(self.res())
        return json.loads(p, object_hook=hinted_tuple_hook)


@dataclass
class Custom(BaseProp):
    _type: PropType = field(default=PropType.custom, init=False)
    type: type[object] | str

    def filter_props(self, ignore: Iterable = []):
        return super().filter_props(["type"])

    def get_type(self):
        return self.type if isinstance(self.type, str) else self.type.__name__


@dataclass
class Select(BaseProp):
    _type: PropType = field(default=PropType.select, init=False)
    select: list[str] = field(default_factory=list, kw_only=True)
    default: str | None = field(default=None)


@dataclass
class Clip(BaseProp):
    _type: PropType = field(default=PropType.clip, init=False)


@dataclass
class Int(BaseProp):
    _type: PropType = field(default=PropType.int, init=False)

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
class Float(BaseProp):
    _type: PropType = field(default=PropType.float, init=False)

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
class Bool(BaseProp):
    _type: PropType = field(default=PropType.bool, init=False)

    default: bool = field(default=True)


@dataclass
class String(BaseProp):
    _type: PropType = field(default=PropType.string, init=False)

    forceInput: bool = field(default=False)
    multiline: bool = field(default=False)
    """True if you want the field to look like the one on the ClipTextEncode node"""
    default: str = field(default="")

    def hints(self) -> str: ...


@dataclass
class Conditioning(BaseProp):
    _type: PropType = field(default=PropType.cond, init=False)


@dataclass
class Image(BaseProp):
    _type: PropType = field(default=PropType.image, init=False)


# class BaseMetaClass(type):
#     def __init__(cls, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         if "input" not in cls.__dict__:
#             raise TypeError(f"'input' prop must be defined within class {cls.__name__}")
#         if "output" not in cls.__dict__:
#             raise TypeError(
#                 f"'output' prop must be defined within class {cls.__name__}"
#             )

#         d: tuple[BaseProp, ...] = cls.__dict__["output"]
#         RETURN_TYPES = tuple([i._type.value for i in d])
#         RETURN_NAMES = tuple(
#             [(i.key if i.key else str(i._type.value).lower()) for i in d]
#         )

#         assert len(RETURN_NAMES) == len(RETURN_TYPES)
#         setattr(cls, "RETURN_TYPES", RETURN_TYPES)
#         setattr(cls, "RETURN_NAMES", RETURN_NAMES)


class BaseNode:

    required: Iterable[BaseProp] = list()
    hidden: Iterable[BaseProp] = list()
    optional: Iterable[BaseProp] = list()
    output: Iterable[BaseProp] = list()

    OUTPUT_NODE: bool = True
    FUNCTION: str = "encode"
    CATEGORY: str = ""

    def __init_subclass__(cls, *args, **kwargs):
        """
        控制 `RETURN_TYPES` 和 `RETURN_NAMES` 的生成
        """
        super().__init_subclass__(*args, **kwargs)

        if "__iter__" not in cls.__dict__:
            cls.output = ()

        RETURN_TYPES = tuple([i.get_type() for i in cls.output])
        RETURN_NAMES = tuple([(i.key if i.key else i.get_type()) for i in cls.output])

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


if __name__ == "__main__":
    from rich import print

    # p = Custom("option", BaseProp)
    # print(p.schema())

    p = (Int("t"), Int("q"), String("r"))
    # do some typing for q with p then pylance could get
    q: tuple[int, int, str]

    # def test(*args):
    #     assert_type(args, p)

    print(Int("t").hints())
    # print((1, None) == (1,))
