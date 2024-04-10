from collections import namedtuple
from .core import BaseNode, Bool, Float, Image, Int, Select, String

siz = {"min": 64, "max": 1728, "step": 64}


class TxtToImg(BaseNode):
    required = (
        Bool("free_size"),
        String("prompt"),
        String("uc"),
        Select("uc_preset", select=["None", "Light", "Heavy", "Human Focus"]),
        Int("height", default=1216, min=64, max=1728, step=64),
        Int("width", default=832, min=64, max=1728, step=64),
        Int("step", default=28, min=1, max=50, step=1),
        Int("seed", -1, -1, 9999999999, 1),
        Select(
            "sampler",
            "k_euler_ancestral",
            select=[
                "k_euler",
                "k_euler_ancestral",
                "k_dpmpp_2s_ancestral",
                "k_dpmpp_2m",
                "k_dpmpp_sde",
                "ddim",
            ],
        ),
        Select("smea", "none", select=["none", "SMEA", "SMEA+DYN"]),
        Float("prompt_guidance", 6, 0, 10),
        Float("prompt_guidance_rescale", 0, 0, 1),
        Float("uc_strength", 1, 0, 10, round=0.01),
        Select("noise", "native", select=["native", "exponential", "polyexponential"]),
    )

    output = (Image("image"),)
