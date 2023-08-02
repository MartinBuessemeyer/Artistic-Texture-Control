import torch

from parameter_optimization.strotss_org import tensor_resample
from parameter_prediction_network.johnson_nst.johnson_nst_model import JohnsonNSTModel


class JohnsonL1Loss(torch.nn.Module):
    def __init__(self, trained_johnson_path: str, weight: float = 1.0, loss_func=torch.nn.L1Loss(),
                 with_network_grads: bool = False):
        super().__init__()
        johnson_model = JohnsonNSTModel.load_from_checkpoint(trained_johnson_path, strict=False)
        self.johnson_network = johnson_model.network
        self.johnson_network.eval()
        self.weight = weight
        self.with_network_grads = with_network_grads
        self.loss_func = loss_func

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.with_network_grads:
            johnson_stylized = tensor_resample(self.johnson_network(x), y.shape[2:])
        else:
            with torch.no_grad():
                johnson_stylized = tensor_resample(self.johnson_network(x), y.shape[2:]).detach()
        return self.weight * self.loss_func(y, johnson_stylized)
