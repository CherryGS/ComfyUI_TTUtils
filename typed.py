import typing as tp


clip = tp.TypeVar("clip")

input_type = dict[str, dict[str, tuple[str | list, dict[str, int | str | bool] | None]]]


class TypedClass(tp.Protocol):
    @classmethod
    def INPUT_TYPES(cls) -> input_type: ...


T_NODE_CLASS_MAPPINGS = dict[str, TypedClass]
