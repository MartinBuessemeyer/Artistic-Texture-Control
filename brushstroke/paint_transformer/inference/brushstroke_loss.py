import torch

from brushstroke.paint_transformer.inference.paint_transformer import PaintTransformer


class BrushStrokeLoss(torch.nn.Module):
    def __init__(self, brushstroke_weight: float, model_path: str = './brushstroke/paint_transformer/train/'
                                                                    'checkpoints/painter_grayscale/latest_net_g.pth'):
        super().__init__()
        self.paint_transformer = PaintTransformer(model_path,
                                                  brush_vertical_path='./brushstroke/paint_transformer/inference/brush/brush_large_vertical.png',
                                                  brush_horizontal_path='./brushstroke/paint_transformer/inference/brush/brush_large_horizontal.png',
                                                  num_color_channels=1,  num_skip_first_drawing_layers = 0)#, num_skip_last_drawing_layers = 0, patch_size=16)
        self.brushstroke_weight = brushstroke_weight
        self.loss_func = torch.nn.L1Loss()

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=masks.device)
        if self.brushstroke_weight == 0.0:
            return loss
        for batch_idx in range(masks.shape[0]):
            for mask_idx in range(masks.shape[1]):
                cur_mask = masks[batch_idx:batch_idx + 1, mask_idx: mask_idx + 1]
                # skip masks that only have one color
                example_color = cur_mask[0, 0, 0, 0]
                if not torch.all(cur_mask == example_color):
                    result = self.paint_transformer(cur_mask)
                    result = result.detach()
                    loss += self.loss_func(result, cur_mask)
        loss = loss / masks.shape[0] / masks.shape[1]
        return self.brushstroke_weight * loss
