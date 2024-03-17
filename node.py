# from .typed import TypedClass
from typed import *


class ClipTextEncoderPlus:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiple": True}),
                "cond": ("CONDITIONING",),
                "opt": (["combine", "average"],),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ()
    FUNCTION = "solve"
    CATEGORY = "Tickten"

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


NODE_CLASS_MAPPINGS: T_NODE_CLASS_MAPPINGS = {
    "ClipTextEncoderPlus": ClipTextEncoderPlus
}
NODE_DISPLAY_NAME_MAPPINGS = {"ClipTextEncoderPlus": "Clip Text Encoder +"}

if __name__ == "__main__":
    """"""
    # def check(x: TypedClass):
    #     pass

    # for i in NODE_CLASS_MAPPINGS.values():
    #     check(i)
