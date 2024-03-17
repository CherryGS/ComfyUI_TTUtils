import typing as tp
tp.IO

INPUT_TYPE = dict[str, tuple]

class TypedClass:
    @classmethod
    def INPUT_TYPES(cls) -> INPUT_TYPE: ...
