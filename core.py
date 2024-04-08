from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
import json


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


@dataclass()
class BaseProp:
    select: list[str] = field(default_factory=list, init=False)
    _type: PropType
    key: str

    def _gen(self):
        return {
            i: j
            for i, j in filter(
                lambda x: x[0] not in ["key", "select"] and x[1] is not None,
                self.__dict__.items(),
            )
        }

    def _res(self):
        if self._type != PropType.select:
            return (self._type, self._gen())
        return (self.select, self._gen())

    def val(self) -> str:

        p = JsonEncoder().encode(self._res())
        return json.loads(p, object_hook=hinted_tuple_hook)


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
class String(BaseProp):
    _type: PropType = field(default=PropType.string, init=False)

    forceInput: bool = field(default=False)
    multiline: bool = field(default=False)
    """True if you want the field to look like the one on the ClipTextEncode node"""
    default: str = field(default="")


@dataclass
class Conditioning(BaseProp):
    _type: PropType = field(default=PropType.cond, init=False)


class BaseMetaClass(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "input" not in cls.__dict__:
            raise TypeError(f"'input' prop must be defined within class {cls.__name__}")
        if "output" not in cls.__dict__:
            raise TypeError(
                f"'output' prop must be defined within class {cls.__name__}"
            )

        d: tuple[BaseProp, ...] = cls.__dict__["output"]
        RETURN_TYPES = tuple([i._type.value for i in d])
        RETURN_NAMES = tuple(
            [(i.key if i.key else str(i._type.value).lower()) for i in d]
        )

        assert len(RETURN_NAMES) == len(RETURN_TYPES)
        setattr(cls, "RETURN_TYPES", RETURN_TYPES)
        setattr(cls, "RETURN_NAMES", RETURN_NAMES)


class BaseNode:

    input: tuple[BaseProp, ...]
    output: tuple[BaseProp, ...]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {i.key: i.val() for i in cls.input}}

    @classmethod
    def IS_CHANGED(cls, *args):
        return ""


if __name__ == "__main__":
    from rich import print

    class TestNode(BaseNode, metaclass=BaseMetaClass):

        input = (Int("int"), String("str"))
        output = (Int("int"), String("str"))
        FUNCTION = "solve"
        CATEGORY = "TT"

    t = Select("select", default="a", select=["a", "b", "c"])
    print(t._res())
    print(TestNode.INPUT_TYPES())
    print(TestNode.__dict__)
