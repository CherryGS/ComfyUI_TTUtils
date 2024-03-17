class ClipTextEncoder:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Input your prompt."},
                ),
            },
        }

    RETURN_TYPES = (
        "CLIP",
        "CONDITIONING",
    )
    RETURN_NAMES = (
        "clip",
        "cond_out",
    )
    FUNCTION = "solve"
    CATEGORY = "Tickt"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def solve(self, clip, text: str):
        return (clip, self.encode(clip, text))


class ClipTextEncoderPlus:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Input your prompt."},
                ),
                "cond_in": ("CONDITIONING",),
                "opt": (["combine", "average", "ignore"],),
            },
        }

    RETURN_TYPES = (
        "CLIP",
        "CONDITIONING",
    )
    RETURN_NAMES = (
        "clip",
        "cond_out",
    )
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


NODE_CLASS_MAPPINGS = {
    "ClipTextEncoder": ClipTextEncoder,
    "ClipTextEncoderPlus": ClipTextEncoderPlus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipTextEncoder": "Clip Text Encoder",
    "ClipTextEncoderPlus": "Clip Text Encoder +",
}

if __name__ == "__main__":
    """"""
    # def check(x: TypedClass):
    #     pass

    # for i in NODE_CLASS_MAPPINGS.values():
    #     check(i)
