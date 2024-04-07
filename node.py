# from .typed import TypedClass
from core import *


class ClipTextEncoderPlus(BaseNode, metaclass=BaseMeta):

    input = (
        Clip("clip"),
        String("text"),
        BaseProp(PropType.cond, "cond"),
        Select("opt", select=["combine", "average"]),
    )

    output = (BaseProp(PropType.cond, ""),)
    OUTPUT_NODE = True
    FUNCTION = "solve"
    CATEGORY = "TT"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)

    def solve(self, clip, text: str, cond, opt: str):
        match opt:
            case "combine":
                return [cond + self.encode(clip, text)]
            case "average":
                pass


NODE_CLASS_MAPPINGS = {"ClipTextEncoderPlus": ClipTextEncoderPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"ClipTextEncoderPlus": "Clip Text Encoder +"}

if __name__ == "__main__":
    """"""
    # def check(x: TypedClass):
    #     pass

    # for i in NODE_CLASS_MAPPINGS.values():
    #     check(i)
