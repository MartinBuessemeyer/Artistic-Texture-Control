from dataclasses import dataclass

import torch

from effects.bilateral import BilateralEffect
from effects.bump_mapping import BumpMappingEffect
from effects.edge_blend import EdgeBlendEffect
from effects.flow_aligned_bilateral import FlowAlignedBilateralEffect
from effects.flow_aligned_smoothing import FlowAlignedSmoothingEffect
from effects.gauss2d_xy_separated import Gauss2DEffect
from effects.lab_to_rgb import LabToRGBEffect
from effects.noise import NoiseEffect
from effects.rgb_to_lab import RGBToLabEffect
from effects.structure_tensor import StructureTensorEffect
from effects.tangent_flow_map import TangentFlowEffect
from effects.xdog_pass0 import XDoGPass0Effect
from effects.xdog_pass1 import XDoGPass1Effect
from helpers.effect_base import EffectBase
from helpers.index_helper import IndexHelper
from helpers.visual_parameter_def import arbitrary_style_vp_ranges


@dataclass
class PipelineConfiguration:
    name: str = True
    use_bilateral: bool = True
    use_contrast: bool = True
    use_xdog: bool = True
    use_bump_mapping: bool = True
    learnable_xdog_contour: bool = True
    learnable_bump_specular: bool = True
    learnable_bump_scale: bool = True
    use_regional_consistency_losses: bool = False


class ArbitraryStyleAblationStudyEffect(EffectBase):
    def __init__(self, config: PipelineConfiguration):
        super().__init__(arbitrary_style_vp_ranges)
        self.config = config
        # locals
        dim_kernsize = 10

        # effects
        self.rgb_to_lab = RGBToLabEffect()
        self.lab_to_rgb = LabToRGBEffect()
        self.bilateral = BilateralEffect(dim_kernsize=dim_kernsize)

        self.structureTensorPass = StructureTensorEffect()
        self.gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=dim_kernsize)
        self.gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=dim_kernsize)
        self.tangent_flow = TangentFlowEffect()

        self.xDoGPass0 = XDoGPass0Effect(dim_kernsize=dim_kernsize)  # TODO XDoG LIC ???
        self.xDoGPass1 = XDoGPass1Effect()

        self.edge_blend = EdgeBlendEffect()

        self.bilateralPass0 = FlowAlignedBilateralEffect(False, dim_kernsize=dim_kernsize)
        self.bilateralPass1 = FlowAlignedBilateralEffect(True, dim_kernsize=dim_kernsize)

        self.noise = NoiseEffect()

        self.noise_smoothing = FlowAlignedSmoothingEffect(True)

        self.bump = BumpMappingEffect()

    def forward_effect(self, x, visual_parameters):
        i = IndexHelper(x)

        contrast = self.vpd.select_parameter(visual_parameters, "contrast")
        # luminosity_offset = self.vpd.select_parameter(visual_parameters, "luminosity_offset")

        bilateral_sigma_r_1 = self.vpd.select_parameter(visual_parameters, "bilateral_sigma_r_1")
        bilateral_sigma_d_1 = self.vpd.select_parameter(visual_parameters, "bilateral_sigma_d_1")

        contour = self.vpd.select_parameter(visual_parameters, "contour")
        contour_opacity = self.vpd.select_parameter(visual_parameters, "contour_opacity")

        bump_phong_specular = self.vpd.select_parameter(visual_parameters, "bump_phong_specular")
        bump_opacity = self.vpd.select_parameter(visual_parameters, "bump_opacity")
        bump_scale = self.vpd.select_parameter(visual_parameters, "bump_scale")

        if not self.config.learnable_xdog_contour:
            contour = torch.full_like(contour, 50.0)
        if not self.config.learnable_bump_specular:
            bump_phong_specular = torch.full_like(bump_phong_specular, 10.0)
        if not self.config.learnable_bump_scale:
            bump_scale = torch.full_like(bump_scale, 5.0)

        if self.config.use_bilateral:
            stylized_image = self.smoothing_part_bilateral(x, bilateral_sigma_d_1,
                                                           bilateral_sigma_r_1)
        else:
            stylized_image = x
        if self.config.use_xdog:
            stylized_image, xdog_intermediate = self.contours_part(x, stylized_image, i,
                                                                   contour,
                                                                   contour_opacity)

        if self.config.use_bump_mapping:
            bump = self.paint_texture_part(stylized_image, bump_scale, bump_phong_specular)
            edge_blend = i.cat(i.idx(stylized_image, 'xyz'), bump_opacity)
            stylized_image = self.composition_part(bump, edge_blend)

        if self.config.use_contrast:
            stylized_image = self.contrast_adjustment(stylized_image, contrast)

        return stylized_image

    def paint_texture_part(self, x, bump_scale, bump_phong_specular):
        noise_smoothing_step_size = torch.tensor(1.0, device=x.device)
        noise_smoothing_step_size_scaling_factor = torch.tensor(0.0, device=x.device)
        colorSmoothing = torch.tensor(10, device=x.device)  # the "Size/details" of noise (smaller looks better)
        tf_1_gauss_sigma = torch.tensor(16.0, device=x.device)  # The smoothness of noise 'strokeLength'

        bump_sample_distance = torch.tensor(0.2626, device=x.device)
        bump_phong_shininess = torch.tensor(14.0, device=x.device)
        brushScale = torch.tensor(1.0, device=x.device)

        noise_smoothing_sigma = colorSmoothing / 3.8086
        bump_sigma_color = colorSmoothing / 3.8086
        noise_scale = brushScale / 3.8086

        precisionFactor = torch.tensor(1.0 / 3.8086, device=x.device)
        structure_tensor_sigma = torch.tensor(1.0 / 3.8086, device=x.device)
        tf_1, _ = self.tf_map(self.run(self.rgb_to_lab, x), structure_tensor_sigma,
                              tf_1_gauss_sigma, precisionFactor)
        noise = self.noise(x, noise_scale)
        noise_smoothing = self.noise_smoothing(noise, tf_1,
                                               noise_smoothing_sigma,
                                               noise_smoothing_step_size,
                                               noise_smoothing_step_size_scaling_factor)

        bump = self.run(self.bump, noise_smoothing, bump_sigma_color, bump_scale, bump_phong_shininess,
                        bump_phong_specular, bump_sample_distance)
        return bump

    def composition_part(self, bump, color_smoothing_upsampled):
        output = self.run(self.compose, color_smoothing_upsampled, bump)
        return output

    def contrast_adjustment(self, image: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
        grayscale_factor = torch.tensor([0.2989, 0.587, 0.114], device=image.device).view(1, -1, 1, 1)
        grayscale_image = (image * grayscale_factor).sum(dim=1, keepdim=True)
        mean = torch.mean(grayscale_image, dim=(-3, -2, -1), keepdim=True)
        return (contrast * image + (1.0 - contrast) * mean).clamp(0.0, 1.0)

    def bump_mapping(self, stylized_image: torch.Tensor, phong_img: torch.Tensor, shininess: torch.Tensor,
                     specular: torch.Tensor, opacity: torch.Tensor, i: IndexHelper) -> torch.Tensor:
        bump_sample_distance = torch.tensor(0.2626, device=stylized_image.device)
        bump_scale = torch.tensor(3.0, device=stylized_image.device)
        bump_sigma_color = torch.tensor(30.0 / 3.8086, device=stylized_image.device)

        bump = self.run(self.bump, phong_img, bump_sigma_color, bump_scale, shininess, specular, bump_sample_distance)
        stylized_image = i.cat(stylized_image, opacity)
        return self.run(self.compose, stylized_image, bump)

    def contours_part(self, image, stylized_image, i,
                      contour, contour_opacity):
        xdog_epsilon = torch.tensor(0.6, device=image.device)
        xdog_phi = torch.tensor(1.0, device=image.device)
        structure_tensor_sigma = torch.tensor(8.0, device=image.device)
        tf_1_gauss_sigma = torch.tensor(0.5, device=image.device)
        xdog_sigma_narrow = torch.tensor(0.4201, device=image.device)
        xdog_sigma_edge = torch.tensor(0.7877, device=image.device)
        xdog_wide_kernel_weight = contour / 100.0

        lab_image = self.run(self.rgb_to_lab, image)
        tf, _ = self.tf_map(lab_image, structure_tensor_sigma, tf_1_gauss_sigma)
        xdog_contours_intermediate = self.run(self.xDoGPass0, lab_image, tf, xdog_wide_kernel_weight,
                                              xdog_sigma_narrow)
        xdog_contours = self.run(self.xDoGPass1, xdog_contours_intermediate, tf, xdog_epsilon,
                                 xdog_sigma_edge, xdog_phi)
        xdog_contours = i.idx(xdog_contours, "xxx")
        edge_blend = self.run(self.edge_blend, stylized_image, xdog_contours, contour_opacity)
        return edge_blend, xdog_contours_intermediate

    def smoothing_part_bilateral(self, x, bilateral_sigma_d, bilateral_sigma_r):
        rgb_2_lab_0 = self.run(self.rgb_to_lab, x)
        bilateral = self.run(self.bilateral, rgb_2_lab_0, bilateral_sigma_d, bilateral_sigma_r)
        return self.run(self.lab_to_rgb, bilateral)

    def tf_map(self, xlab, sigma_sst, sigma_gauss, precisionFactor=1.0):
        sst = self.run(self.structureTensorPass, xlab, sigma_sst)
        gauss_1 = self.run(self.gauss2dx, sst, sigma_gauss, precisionFactor)
        gauss_2 = self.run(self.gauss2dy, gauss_1, sigma_gauss, precisionFactor)
        return self.run(self.tangent_flow, gauss_2), sst
