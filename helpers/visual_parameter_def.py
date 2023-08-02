import torch

from helpers.scale_visual_parameters import ScaleVisualParameters

arbitrary_style_vp_ranges = [("bilateral_sigma_r_1", 0.0, 10.0),  # edge persevation
                             ("bilateral_sigma_d_1", 0.0, 10.0),  # range
                             ("contour", 0.0, 100.0),
                             ("contour_opacity", 0.0, 1.0),
                             ("contrast", 1.0, 4.0),
                             ("bump_phong_specular", 1.0, 100.0),
                             ("bump_phong_shininess", 1.0, 100.0),
                             ("bump_opacity", 0.0, 1.0),
                             ("bump_scale", 0.0, 17.0),
                             ]

arbitrary_style_presets = [("bilateral_sigma_r_1", 3.0),
                           ("bilateral_sigma_d_1", 3.0),
                           ("contour", 50.0),
                           ("contour_opacity", 0.0),
                           ("contrast", 1.0),
                           ("bump_phong_specular", 10.0),
                           ("bump_phong_shininess", 10.0),
                           ("bump_opacity", 0.0),
                           ('bump_scale', 0.0)
                           ]

toon_vp_ranges = [("brightness", -0.15, 0.5),
                  ("contrast", 0.2, 3.0),
                  ("saturation", 0.0, 3.0),
                  ("colorQuantization", 5.0, 30.0),
                  ("colorBlur", 0.0125, 0.14),
                  ("details", 0.0, 3.0),
                  ("strokeWidth", 0.3, 1.2),
                  ("contour", 0.0, 3.0),
                  ("blackness", 0.01, 0.5),
                  ("spotColor_r", 0.0, 255.0),
                  ("spotColor_g", 0.0, 255.0),
                  ("spotColor_b", 0.0, 255.0),
                  ("spotColorAmount", 0.0, 1.0),
                  ("finalSmoothing", 1.0, 10.0)]

portrait_preset = [("depthPower", 1.0),
                   ("depthColorTransfer", 0.0),
                   ("brightness", 0.1),
                   ("contrast", 1.15),
                   ("saturation", 1.02),
                   ("colorQuantization", 18.93),
                   ("colorBlur", 0.14),
                   ("strokeWidth", 0.54),
                   ("contour", 1.0),
                   ("details", 2.27),
                   ("blackness", 0.15),
                   ("finalSmoothing", 1.0),
                   ("spotColor_r", 153.0),
                   ("spotColor_g", 197.0),
                   ("spotColor_b", 208.0),
                   ("spotColorAmount", 0.0),
                   ("adaptHuePreprocess", 0.5),
                   ("adaptHuePostprocess", 0.5)]


class VisualParameterDef:
    def __init__(self, vp_ranges):
        self.name2idx = {}
        self.vp_ranges = vp_ranges
        self.scale_parameters = ScaleVisualParameters(vp_ranges)
        # Note that this is the only place where parameters should be really scaled

        i = 0
        for n, _, _ in vp_ranges:
            self.name2idx[n] = i
            i += 1

    def select_parameter(self, tensor, name):
        return tensor[:, self.name2idx[name]:self.name2idx[name] + 1]

    def select_parameters(self, tensor, parameter_names):
        result = tensor.new_empty((tensor.size(0), len(parameter_names), *tensor.size()[2:]))

        for i, pn in enumerate(parameter_names):
            result[:, i] = tensor[:, self.name2idx[pn]]

        return result

    def preset_tensor(self, preset, reference_tensor, add_local_dims=False):
        if add_local_dims:
            dims = (reference_tensor.size(0), len(self.name2idx), reference_tensor.size(2), reference_tensor.size(3))
        else:
            dims = (reference_tensor.size(0), len(self.name2idx))

        result = reference_tensor.new_empty(dims)
        for n, v in preset:
            result[:, self.name2idx[n]] = v

        return self.scale_parameters(result, True)  # scale back

    def update_visual_parameters(self, vp_tensor, parameter_names, update_tensor, support_cascading=False):
        if support_cascading:
            print("CASCADING should not be used anymore")
            raise RuntimeError("cascading")

        vp_tensor = vp_tensor.clone()
        for i, pn in enumerate(parameter_names):
            vp_tensor[:, self.name2idx[pn]] = update_tensor[:, i]

        return vp_tensor

    @staticmethod
    def get_param_range():
        return -0.5, 0.5

    @staticmethod
    def clamp_range(vp):
        return torch.clamp(vp, -0.5, 0.5)

    @staticmethod
    def rand_like(vp):
        return torch.rand_like(vp) - 0.5

    @staticmethod
    def rand(shape):
        return torch.rand(shape) - 0.5
