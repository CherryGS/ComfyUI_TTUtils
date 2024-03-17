from .typed import TypedClass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class ClipTextEncoderPlus(TypedClass):

    @classmethod
    def INPUT_TYPES(cls):
        return 1
        return {
            "required": {
                "pre_clip": ("CLIP",),
                "extra_propmt": ("STRING", {"multiple": True})
            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ()
    FUNCTION = "combine"
    CATEGORY = "Tickten"

    def combine(self, pre_clip, extra_propmt):
        pass
