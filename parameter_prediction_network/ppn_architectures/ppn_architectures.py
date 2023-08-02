import torch
from segmentation_models_pytorch import Unet

from parameter_prediction_network.ppn_architectures import PPNArchitectures
from parameter_prediction_network.ppn_architectures.johnson_instance_normalization import JohnsonInstanceNorm
from parameter_prediction_network.ppn_architectures.johnson import JohnsonPPN


def replace_batch_norm_with_other_layer(model: torch.nn.Module, layer) -> None:
    bn_submodule_names = (name for name, k in model.named_modules()
                          if type(model.get_submodule(name)) == torch.nn.modules.batchnorm.BatchNorm2d)
    for bn_submodule_name in bn_submodule_names:
        batch_norm = model.get_submodule(bn_submodule_name)
        parent_module_name = '.'.join(bn_submodule_name.split('.')[:-1])
        bn_name = bn_submodule_name.split('.')[-1]
        setattr(model.get_submodule(parent_module_name), bn_name,
                layer(batch_norm.num_features, affine=True))


def get_unet(num_outs: int, encoder_name: str, use_instance_norm: bool) -> torch.nn.Module:
    module = Unet(classes=num_outs,
                  encoder_weights="imagenet", activation="identity",
                  encoder_name=encoder_name)
    replace_batch_norm_with_other_layer(module, torch.nn.InstanceNorm2d if use_instance_norm else torch.nn.Identity)
    return module


def get_ppn_architecture(num_outs: int, architecture_name: PPNArchitectures) -> torch.nn.Module:
    if architecture_name == PPNArchitectures.johnson:
        return JohnsonPPN(num_outs, use_batchnorm=True)
    if architecture_name == PPNArchitectures.johnson_no_batch_norm:
        return JohnsonPPN(num_outs, use_batchnorm=False)
    if architecture_name == PPNArchitectures.johnson_instance_norm:
        return JohnsonInstanceNorm(num_outs)
    if architecture_name == PPNArchitectures.unet_small_instance_norm:
        return get_unet(num_outs, 'efficientnet-b0', use_instance_norm=True)
    if architecture_name == PPNArchitectures.unet_small_no_batch_norm:
        return get_unet(num_outs, 'efficientnet-b0', use_instance_norm=False)
    if architecture_name == PPNArchitectures.unet_large_instance_norm:
        return get_unet(num_outs, 'efficientnet-b7', use_instance_norm=True)
    if architecture_name == PPNArchitectures.unet_large_no_batch_norm:
        return get_unet(num_outs, 'efficientnet-b7', use_instance_norm=False)
    else:
        raise AttributeError(f'Requested invalid architecture: {architecture_name}')
