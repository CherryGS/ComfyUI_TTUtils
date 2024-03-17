class Gg123:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target": ("INT",),
                "param": ("STRING",{}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ()
    FUNCTION = "test"
    CATEGORY = "tt"

    def test(self, target, param):
        return (123,)
        raise Exception(target_model)
