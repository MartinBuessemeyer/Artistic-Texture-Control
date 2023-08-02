from effects.arbitrary_style import ArbitraryStyleEffect
from effects.xdog import XDoGEffect
from helpers.visual_parameter_def import arbitrary_style_vp_ranges, arbitrary_style_presets, portrait_preset

xdog_params = ["blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness"]
toon_params = xdog_params + ["finalSmoothing", "colorBlur", "colorQuantization"]  # this is the color quant. version
arbitrary_style_params = [x[0] for x in arbitrary_style_vp_ranges]


def get_effect_short_name(name: str) -> str:
    if name == 'arbitrary_style':
        return 'ab_style'


def get_default_settings(name):
    if name == "xdog":
        effect = XDoGEffect()
        presets = portrait_preset
        params = xdog_params
    elif name == "arbitrary_style":
        effect = ArbitraryStyleEffect()
        presets = arbitrary_style_presets
        params = arbitrary_style_params
    else:
        raise ValueError(f"effect {name} not found")
    return effect, presets, params
