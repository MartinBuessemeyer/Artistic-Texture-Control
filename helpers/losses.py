from functools import reduce

import clip
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import MSELoss, L1Loss
from torch.nn.functional import mse_loss, l1_loss, interpolate
from torchvision import transforms
from torchvision.transforms import Resize, Compose
from torchvision.transforms import ToTensor

from effects.rgb_to_lab import RGBToLabEffect
from effects.structure_tensor import StructureTensorEffect
from helpers.color_conversion import rgb_to_yuv
from helpers.hist_layers import SingleDimHistLayer
from helpers.hist_metrics import DeepHistLoss, EarthMoversDistanceLoss
from helpers.index_helper import IndexHelper
from helpers.ms_ssim import MixLoss
from helpers.template import imagenet_templates
from helpers.vgg_feature_extractor import Vgg19FeatureExtractor
from parameter_optimization.strotss_org import Vgg16_Extractor, tensor_resample, sample_indices, make_laplace_pyramid, \
    fold_laplace_pyramid, calculate_loss
from parameter_prediction_network.specific_losses.vgg16_loss import Vgg16Loss


def gram_matrix(input):
    a, b, c, d = input.size()  # keep batch dim separate!

    features = input.view(a, b, c * d)
    G = torch.matmul(features, features.transpose(1, 2))

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b * c * d)


def get_individual_syle_weight(gram_matrices):
    return 3 / (sum(torch.linalg.norm(style_gram_matrix) for style_gram_matrix in gram_matrices) / len(gram_matrices))


class PerceptualLoss(nn.Module):
    def __init__(self, style_image_path, image_dim, style_weight=1e10, content_weight=1e5,
                 lightning_module=None,
                 **kwargs):
        super().__init__()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("style_image_path", "style_weight", "content_weight")

        self.vgg = Vgg19FeatureExtractor()
        self.mse_loss = MSELoss()

        self.content_weight = content_weight
        self.style_weight = style_weight

        style_image = Image.open(style_image_path).convert("RGB")
        style_image = ToTensor()(style_image)
        if image_dim:
            # style_image = style_image.resize(image_dim)
            style_image = Resize(image_dim)(style_image)
        self.target_styles = self.generate_targets(style_image.unsqueeze(0))

    def generate_targets(self, style_image):
        features = self.vgg(style_image)
        gram_matrices = [gram_matrix(f) for f in features]
        return [style_gram_matrix for style_gram_matrix in gram_matrices]

    def forward(self, x, y):
        y_feat = self.vgg(y)
        x_feat = self.vgg(x)
        content_loss = self.content_weight * self.mse_loss(x_feat.conv_4, y_feat.conv_4)

        y_grams = [gram_matrix(f) for f in y_feat]
        style_loss = self.style_weight * torch.stack(
            [self.mse_loss(y_gram, target_style.to(x.device).repeat(y_gram.size(0), 1, 1))
             for y_gram, target_style in zip(y_grams, self.target_styles)]).sum()

        return content_loss, style_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--style_weight', type=float, default=1e10)
        parser.add_argument('--content_weight', type=float, default=1e5)
        parser.add_argument('--style_image_path', default="manga_style.png")
        return parent_parser


class PerceptualStyleLoss(nn.Module):
    def __init__(self, style_image_path, image_dim=1024, style_weight=1.0, lightning_module=None, **kwargs):
        super().__init__()

        style_image = Image.open(style_image_path).convert("RGB").resize((image_dim, image_dim))
        style_image = ToTensor()(style_image).unsqueeze(0)

        self.vgg = Vgg19FeatureExtractor()
        self.style_weight = style_weight
        self.target_style = self.generate_targets(style_image)
        self.mse_loss = MSELoss()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("style_weight", "style_image_path")

    def generate_targets(self, style_image):
        features = self.vgg(style_image)
        gram_matrices = [gram_matrix(f) for f in features]
        return [style_gram_matrix * get_individual_syle_weight(gram_matrices)
                for style_gram_matrix in gram_matrices]

    def forward(self, y):
        assert self.target_style is not None
        y_feat = self.vgg(y)
        y_grams = [gram_matrix(f) for f in y_feat]

        style_loss = self.style_weight * torch.stack(
            [self.mse_loss(e[0], e[1].to(e[0].device).repeat(e[0].size(0), 1, 1))
             for e in zip(y_grams, self.target_style)]).sum()

        return style_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--style_weight', type=float, default=1e5)
        return parent_parser


class PerceptualContentLoss(nn.Module):
    def __init__(self, content_weight=1.0, lightning_module=None, **kwargs):
        super().__init__()
        self.content_weight = content_weight
        self.mse_loss = MSELoss()
        self.vgg = Vgg19FeatureExtractor()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("content_weight")

    def forward(self, x, y):
        y_feat = self.vgg(y)
        x_feat = self.vgg(x)

        return self.content_weight * self.mse_loss(x_feat.conv_4, y_feat.conv_4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--content_weight', type=float, default=1)
        return parent_parser


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_to_lab = RGBToLabEffect()
        self.structure_tensor = StructureTensorEffect()

    def forward(self, x, y):
        sigma = torch.tensor(0.43137, device=x.device)
        sst_x = torch.clamp(self.structure_tensor(self.rgb_to_lab(x), sigma), -5, 5)
        sst_y = torch.clamp(self.structure_tensor(self.rgb_to_lab(y), sigma), -5, 5)

        return mse_loss(sst_x, sst_y)


# Loss from dehazing paper
class DehazingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_loss = GradientLoss()
        self.content_loss = PerceptualContentLoss()

        for param in self.parameters():
            param.requires_grad = False

        self.lambda_l2 = 1.0
        self.lambda_g = 0.5
        self.lambda_f = 0.8

    def forward(self, x, y):
        l2 = mse_loss(x, y)
        gl = self.grad_loss(x, y)
        cl = self.content_loss(x, y)

        return self.lambda_l2 * l2 + self.lambda_g * gl + self.lambda_f * cl


class TotalVariationLoss(nn.Module):
    def __init__(self, regularizer_weight=1.0, lightning_module=None, **kwargs):
        super().__init__()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("regularizer_weight")

        self.regularizer_weight = regularizer_weight

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--regularizer_weight', type=float, default=3.0)
        return parent_parser

    def forward(self, x):
        return self.regularizer_weight * (
                torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
                torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )


class TotalVariationExponentLoss(nn.Module):
    def __init__(self, regularizer_weight=1.0, error_exponent=1.0, root_exponent=1.0, lightning_module=None, **kwargs):
        super().__init__()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("regularizer_weight")

        self.regularizer_weight = regularizer_weight
        self.error_exponent = error_exponent
        self.root_exponent = root_exponent

    def forward(self, x):
        x_scaled = x * 255.0
        '''return self.regularizer_weight * torch.pow(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]), \
        self.error_exponent).mean() + torch.pow(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]), \
        self.error_exponent).mean()'''
        return self.regularizer_weight * IndexHelper.safe_pow(
            IndexHelper.safe_pow(torch.abs(x_scaled[:, :, 1:, :-1] - x_scaled[:, :, 1:, 1:]), self.error_exponent) + \
            IndexHelper.safe_pow(torch.abs(x_scaled[:, :, :-1, 1:] - x_scaled[:, :, 1:, 1:]), self.error_exponent),
            self.root_exponent).mean()


class HueLoss(nn.Module):
    def __init__(self, hue_weight):
        super().__init__()
        self.hue_weight = hue_weight
        self.loss = torch.nn.L1Loss()

    def forward(self, x, y):
        return self.hue_weight * self.loss(x, y)


class TotalVariationKernelLoss(nn.Module):
    def __init__(self, regularizer_weight=1.0, kernel_size=2, lightning_module=None, **kwargs):
        super().__init__()
        self.regularizer_weight = regularizer_weight

        self.kernel_size = kernel_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--regularizer_weight', type=float, default=3.0)
        return parent_parser

    def forward(self, x):
        loss = torch.tensor(0.0, device=x.device)
        for offset in range(1, self.kernel_size + 1):
            loss += 1 / offset * (torch.mean(torch.abs(x[:, :, :, :-offset] - x[:, :, :, offset:])) +
                                  torch.mean(torch.abs(x[:, :, :-offset, :] - x[:, :, offset:, :])))

        return self.regularizer_weight * loss


# This seems to work well
class DeepHistL1Loss(nn.Module):
    # i.e. histogram (color part) + l1 loss
    def __init__(self, emd_factor=0.2, l1_factor=1.0, l2_factor=0.0):
        super().__init__()
        self.single_dim_hist_layer = SingleDimHistLayer()
        self.emd = EarthMoversDistanceLoss()
        self.l1_loss = L1Loss()
        self.l2_loss = MSELoss()
        self.tv_loss = TotalVariationLoss()

        self.emd_factor = emd_factor
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

    def forward(self, output, target):
        # x and y are RGB images
        assert output.size(1) == 3
        assert target.size(1) == 3

        # this does not work
        # x = resize_keep_aspect_ratio(x)
        # y = resize_keep_aspect_ratio(y)

        x = rgb_to_yuv(output)
        y = rgb_to_yuv(target)

        hist_x = [self.single_dim_hist_layer(x[:, i]) for i in range(3)]
        hist_y = [self.single_dim_hist_layer(y[:, i]) for i in range(3)]

        emd_loss = reduce(torch.add, [self.emd(hist_x[i], hist_y[i]) for i in range(3)]) / 3.0
        return self.emd_factor * emd_loss.mean() + self.l1_factor * self.l1_loss(output, target) \
            + self.l2_factor * self.l2_loss(output, target)  # + 0.0 * self.tv_loss(output)


class CLIPstylerLoss(nn.Module):
    def __init__(self, content_weight=150, lambda_dir=500, num_crops=64, threshold=0.7, crop_size=128,
                 lambda_patch=9000, lambda_tv=2e-3):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, self.preprocess = clip.load("ViT-B/32", jit=False)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(self.device)

        self.source = "a Photo"
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.threshold = threshold
        self.lambda_patch = lambda_patch
        self.lambda_tv = lambda_tv

        self.content_weight = content_weight
        self.lambda_dir = lambda_dir
        self.content_loss = PerceptualContentLoss()

        self.raw_content_image = None

        template_source = self.compose_text_with_templates(self.source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(self.device)
        text_source = self.clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        self.text_source = text_source

        self.cropper = transforms.Compose([
            transforms.RandomCrop(self.crop_size)
        ])
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224)
        ])

    def get_image_prior_losses(self, inputs_jit):
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        return loss_var_l2

    def compose_text_with_templates(self, text, templates=imagenet_templates):
        return [template.format(text) for template in templates]

    def set_text_prompt(self, prompt):
        self.prompt = prompt
        template_text = self.compose_text_with_templates(prompt, imagenet_templates)
        tokens = clip.tokenize(template_text).to(self.device)
        self.text_features = self.clip_model.encode_text(tokens).detach()
        self.text_features = self.text_features.mean(axis=0, keepdim=True)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def set_source(self, image):
        self.source_image = image
        self.source_features = self.clip_model.encode_image(self.clip_normalize(image))
        self.source_features /= self.source_features.clone().norm(dim=-1, keepdim=True)

    def set_raw_content_image(self, image):
        self.raw_content_image = image

    def clip_normalize(self, image):
        image = interpolate(image, size=224, mode='bicubic')
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        image = (image - mean) / std
        return image

    def forward(self, out, *args):

        """clip_loss = (1-torch.cosine_similarity(
            self.clip_model.encode_image(self.clip_normalize(out)),
            self.text_features,
            dim=1
        )).mean()"""

        if self.raw_content_image is None:
            content_loss = self.content_loss(out, self.source_image)
        else:
            content_loss = self.content_loss(out, self.raw_content_image)

        loss_patch = 0
        img_proc = []
        for i in range(self.num_crops):
            target_crop = self.cropper(out)
            target_crop = self.augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc, dim=0)
        img_aug = img_proc

        image_features = self.clip_model.encode_image(self.clip_normalize(img_aug))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        img_direction = image_features - self.source_features
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (self.text_features - self.text_source).repeat(image_features.shape[0], 1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp < self.threshold] = 0
        loss_patch += loss_temp.mean()

        glob_features = self.clip_model.encode_image(self.clip_normalize(out))
        glob_features /= glob_features.clone().norm(dim=-1, keepdim=True)

        glob_direction = (glob_features - self.source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        reg_tv = self.lambda_tv * self.get_image_prior_losses(out)

        total_loss = self.lambda_patch * loss_patch + self.content_weight * content_loss + self.lambda_dir * loss_glob + reg_tv
        # total_loss = self.lambda_patch * loss_patch + self.lambda_dir * loss_glob + reg_tv
        # total_loss = self.content_weight * content_loss
        # total_loss = self.lambda_dir * loss_glob

        return total_loss


def l1_loss_ignore_minus_one(prediction, target):
    return torch.where(target >= -0.5, l1_loss(prediction, target), torch.zeros_like(target))


def categorical_proxy_loss(prediction, target):
    ce_loss = torch.nn.CrossEntropyLoss()
    target = torch.floor((target + 0.5) * prediction.size(2)).to(torch.long)
    loss = torch.tensor(0.0, device=prediction.device)

    for i in range(prediction.size(1)):
        loss += ce_loss(prediction[:, i], target[:, i])

    return loss / prediction.size(1)


def loss_from_string(loss):
    if loss == "dehazing":
        loss_f = DehazingLoss()
    elif loss == "perceptual_content":
        loss_f = PerceptualContentLoss()
    elif loss == "mix":
        loss_f = MixLoss()
    elif loss == "histogram":
        loss_f = DeepHistLoss()
    elif loss == "histogram_l1":
        loss_f = DeepHistL1Loss()
    elif loss == "l2":
        loss_f = mse_loss
    elif loss == "l1":
        loss_f = l1_loss
    elif loss == "l1_loss_ignore_minus_one":
        loss_f = l1_loss_ignore_minus_one
    elif loss == "categorical_proxy":
        loss_f = categorical_proxy_loss
    elif loss == "clipstyler":
        loss_f = CLIPstylerLoss()
    else:
        raise ValueError(f"{loss} is invalid loss")

    return loss_f


# Converts loss(y_hat, target) to loss(source, y_hat) calls
class FlippedVgg16Loss(Vgg16Loss):
    def __init__(self, content_path, style_img_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tv_loss = TotalVariationLoss(regularizer_weight=0)
        self.content_img = Image.open(content_path).convert("RGB")
        transforms = Compose([Resize(style_img_size), ToTensor()])
        self.content_img = transforms(self.content_img).unsqueeze(0).cuda()

    def forward(self, y, style_img):
        # resize content image to match y
        content_resized = torch.nn.functional.interpolate(self.content_img, size=y.shape[-2:], mode='bilinear')
        content_loss, style_loss = super().forward(content_resized, y)
        reg_loss = self.tv_loss(y)
        return content_loss + style_loss + reg_loss


# Converts loss(y_hat, target) to loss(source, y_hat) calls
class StrottsLoss(torch.nn.Module):
    def __init__(self, content_path, style_path, content_weight=16.0, device='cuda', space='uniform'):
        super().__init__()
        self.extractor = Vgg16_Extractor(space=space).to(device)
        transforms = Compose([ToTensor()])
        content_img = Image.open(content_path).convert("RGB")
        content_img = transforms(content_img).unsqueeze(0).cuda()
        content_img = self._scale_to_strotts_interval(content_img)
        style_img = Image.open(style_path).convert("RGB")
        style_img = transforms(style_img).unsqueeze(0).cuda()
        style_img = self._scale_to_strotts_interval(style_img)
        width, height = content_img.shape[-2:]
        scales = []
        for scale in range(10):
            divisor = 2 ** scale
            if min(width, height) // divisor >= 33:
                scales.insert(0, divisor)
        self.scales = scales
        with torch.no_grad():
            content_imgs = [tensor_resample(content_img, [content_img.shape[2] // scale,
                                                          content_img.shape[3] // scale])
                            for scale in self.scales]
            style_imgs = [tensor_resample(style_img, [style_img.shape[2] // scale,
                                                      style_img.shape[3] // scale])
                          for scale in self.scales]
            self.content_img_sizes = [content_img.shape[-2:] for content_img in content_imgs]
            self.feat_contents = [self.extractor(content_img) for content_img in content_imgs]
            self.feat_styles = [torch.cat([self.extractor.forward_samples_hypercolumn(style_img, samps=1000)
                                           for i in range(5)], dim=2)
                                for style_img in style_imgs]
            self.indices = [sample_indices(feat_content[0], feat_style)
                            for feat_content, feat_style in zip(self.feat_contents, self.feat_styles)]
            self.content_weights = [content_weight / max(2.0 * i, 1.0) for i in range(len(self.scales))]

    def forward(self, y, style_img_provided):
        y = y * 2.0 - 1.0
        loss = torch.tensor(0.0, device=y.device)
        for scale, feat_content, feat_style, (xx, xy), content_weight, content_img_size in \
                zip(self.scales, self.feat_contents, self.feat_styles, self.indices, self.content_weights,
                    self.content_img_sizes):
            result_correct_resolution = tensor_resample(y, content_img_size)
            result_pyramid = make_laplace_pyramid(result_correct_resolution, 5)
            stylized = fold_laplace_pyramid(result_pyramid)
            feat_result = self.extractor(stylized)
            loss += calculate_loss(feat_result, feat_content, feat_style, [xx, xy], content_weight)
        self._shuffle_indices()
        return loss / len(self.scales)

    # transform range [0,1] -> [-1.0, 1.0]
    @staticmethod
    def _scale_to_strotts_interval(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2.0 - 1.0

    def _shuffle_indices(self):
        for xx, xy in self.indices:
            np.random.shuffle(xx)
            np.random.shuffle(xy)
