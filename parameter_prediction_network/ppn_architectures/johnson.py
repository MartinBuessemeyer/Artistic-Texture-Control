import torch

from helpers.effect_base import EffectBase


class ResidualBlock(torch.nn.Module):
    def __init__(self, feature_map_size: int, use_batchnorm):
        super().__init__()
        self.l1 = torch.nn.Conv2d(feature_map_size, feature_map_size, 3, padding='same')
        self.batch_norm_1 = torch.nn.BatchNorm2d(feature_map_size) if use_batchnorm else torch.nn.Identity()
        self.activation = torch.nn.ReLU(inplace=True)
        self.l2 = torch.nn.Conv2d(feature_map_size, feature_map_size, 3, padding='same')
        self.batch_norm_2 = torch.nn.BatchNorm2d(feature_map_size) if use_batchnorm else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x = self.l1(x)
        x = self.batch_norm_1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.batch_norm_2(x)
        return x + x_res


class JohnsonPPN(torch.nn.Module):
    def __init__(self, num_outs: int, use_batchnorm=False):
        super().__init__()
        num_parameter_masks_to_predict = num_outs

        self.l1 = torch.nn.Conv2d(3, 32, 9, stride=1, padding='same')
        self.l2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.l3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.res_block_1 = ResidualBlock(128, use_batchnorm)
        self.res_block_2 = ResidualBlock(128, use_batchnorm)
        self.res_block_3 = ResidualBlock(128, use_batchnorm)
        self.res_block_4 = ResidualBlock(128, use_batchnorm)
        self.res_block_5 = ResidualBlock(128, use_batchnorm)

        self.l4 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.l5 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.l6 = torch.nn.Conv2d(32, num_parameter_masks_to_predict, 9, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)

        x = self.l4(x)
        x = self.l5(x)
        out = self.l6(x)
        return out