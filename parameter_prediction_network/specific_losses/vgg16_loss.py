# TAKEN FROM: https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/utils.py
# AND https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/vgg.py
# AT 23.03.2022 16.44h

# LICENCE: BSD-3-Clause License


# Slight adaptions to accumulate loss


from collections import namedtuple

import torch
from PIL import Image
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision.transforms import transforms


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class Vgg16Loss(torch.nn.Module):
    def __init__(self, style_image_path, image_dim, style_weight=1e10, content_weight=1e5):
        super(Vgg16Loss, self).__init__()
        self.vgg16 = Vgg16()
        self.vgg16.eval()
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.mse_loss = torch.nn.MSELoss()
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), persistent=False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), persistent=False)

        if image_dim:
            style_transform = transforms.Compose([
                transforms.Resize(image_dim),
                transforms.ToTensor(),
            ])
        else:
            style_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        style_image = Image.open(style_image_path).convert("RGB")
        style_image = style_transform(style_image)[None, :]

        features_style = self.vgg16(self.normalize(style_image))
        self.gram_style = [gram_matrix(y) for y in features_style]

    def normalize(self, x):
        # normalize using imagenet mean and std
        return (x - self.mean) / self.std

    def forward(self, content_img, y):
        n_batch = len(content_img)
        content_img = self.normalize(content_img)
        y = self.normalize(y)

        features_x = self.vgg16(content_img)
        features_y = self.vgg16(y)
        content_loss = self.content_weight * self.mse_loss(features_y.relu2_2, features_x.relu2_2)
        style_loss = 0.0
        for ft_y, gm_s in zip(features_y, self.gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += self.mse_loss(gm_y, gm_s.to(y.device).repeat(n_batch, 1, 1))
        style_loss *= self.style_weight
        return content_loss, style_loss
