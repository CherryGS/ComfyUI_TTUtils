# from .typed import TypedClass
from .core import *


class ClipTextEncoderPlus(BaseNode):

    required = (
        Clip("clip"),
        String("text"),
        Conditioning("cond_in"),
        Select("opt", select=["combine", "average", "ignore"]),
    )

    output = (
        Clip("clip"),
        Conditioning("cond_out"),
    )
    OUTPUT_NODE = True
    FUNCTION = "solve"
    CATEGORY = "Tickt"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def solve(self, clip, text: str, cond_in, opt: str):
        match opt:
            case "combine":
                return (
                    clip,
                    cond_in + self.encode(clip, text),
                )
            case "ignore":
                return (
                    clip,
                    self.encode(clip, text),
                )
            case _:
                raise NotImplemented


from . import register

register.ezupd([(ClipTextEncoderPlus,)])


if __name__ == "__main__":
    """"""
    ClipTextEncoderPlus()
    # def check(x: TypedClass):
    #     pass

    # for i in NODE_CLASS_MAPPINGS.values():
    #     check(i)
