from enum import Enum

import torch
import torch.nn.functional as F

from brushstroke.paint_transformer.inference import morphology
from brushstroke.paint_transformer.inference import network
from brushstroke.paint_transformer.inference.paint_transformer import PaintTransformer, param2stroke, \
    pad, crop


class BrushstrokeType(Enum):
    CIRCLE = 1
    RECTANGLE = 2
    BRUSH = 3

    def __str__(self):
        return self.name


# 0 => vertical, 1 => horizontal
BRUSHSTROKE_TYPE_2_BRUSHSTROKE_PATHS = {
    str(BrushstrokeType.CIRCLE): ['./brushstroke/paint_transformer/inference/brush/circle_large.png',
                                  './brushstroke/paint_transformer/inference/brush/circle_large.png'],
    str(BrushstrokeType.RECTANGLE): ['./brushstroke/paint_transformer/inference/brush/rectangle_large.png',
                                     './brushstroke/paint_transformer/inference/brush/rectangle_large.png'],
    str(BrushstrokeType.BRUSH): ['./brushstroke/paint_transformer/inference/brush/brush_large_vertical.png',
                                 './brushstroke/paint_transformer/inference/brush/brush_large_horizontal.png']
}

BRUSHSTROKE_TYPE_2_CHECKPOINT = {
    str(BrushstrokeType.CIRCLE): './brushstroke/paint_transformer/train/checkpoints/painter_rgb_circle/latest_net_g.pth',
    str(BrushstrokeType.RECTANGLE): './brushstroke/paint_transformer/train/checkpoints/painter_rgb_rectangle_large/latest_net_g.pth',
    str(BrushstrokeType.BRUSH): './brushstroke/paint_transformer/inference/model.pth'
}


def partial_render_with_details(this_canvas: torch.Tensor, foregrounds: torch.Tensor, alphas: torch.Tensor,
                                decision: torch.Tensor,
                                patch_coord_y: torch.Tensor, patch_coord_x: torch.Tensor, patch_size_y: int,
                                patch_size_x: int,
                                num_color_channels: int, batch_size: int, num_strokes: int,
                                h: int, w: int, details_mask: torch.Tensor, use_detail_decision: bool):
    canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x),
                            stride=(patch_size_y // 2, patch_size_x // 2))
    # canvas_patch: b, 3 * py * px, h * w
    canvas_patch = canvas_patch.view(batch_size, num_color_channels, patch_size_y, patch_size_x, h, w).contiguous()
    canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
    # canvas_patch: b, h, w, 3, py, px
    if use_detail_decision:
        one_channeled_valid_alphas = alphas[..., 0:1, :, :]
        # convert alpha to 0 or 1
        one_channeled_valid_alphas = (one_channeled_valid_alphas != 0).float()

        number_of_pixels_of_stroke = one_channeled_valid_alphas.sum(dim=(4, 5, 6), keepdim=True)
        # calculate whether to render detailed strokes based on the mask.
        if details_mask:
            details_mask_patch = F.unfold(details_mask, (patch_size_y, patch_size_x),
                                          stride=(patch_size_y // 2, patch_size_x // 2))
            # canvas_patch: b, 3 * py * px, h * w
            details_mask_patch = details_mask_patch.view(batch_size, 1, patch_size_y, patch_size_x, h,
                                                         w).contiguous()
            details_mask_patch = details_mask_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
            # Add num_strokes dimension of alpha.
            details_mask_patch = details_mask_patch.unsqueeze(3)
            number_of_detailed_pixels_of_stroke = (one_channeled_valid_alphas * details_mask_patch).sum(dim=(4, 5, 6),
                                                                                                        keepdim=True)

            detail_decision = (number_of_detailed_pixels_of_stroke / number_of_pixels_of_stroke) > 0.5
            decision = torch.logical_and(decision, detail_decision)

    selected_canvas_patch = canvas_patch[:, patch_coord_y, patch_coord_x, :, :, :]
    selected_foregrounds = foregrounds[:, patch_coord_y, patch_coord_x, :, :, :, :]
    selected_alphas = alphas[:, patch_coord_y, patch_coord_x, :, :, :, :]
    selected_decisions = decision[:, patch_coord_y, patch_coord_x, :, :, :, :]
    for i in range(num_strokes):
        cur_foreground = selected_foregrounds[:, :, :, i, :, :, :]
        cur_alpha = selected_alphas[:, :, :, i, :, :, :]
        cur_decision = selected_decisions[:, :, :, i, :, :, :]
        selected_canvas_patch = cur_foreground * cur_alpha * cur_decision + selected_canvas_patch * (
                1 - cur_alpha * cur_decision)
    this_canvas = selected_canvas_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
    # this_canvas: b, 3, h_half, py, w_half, px
    h_half = this_canvas.shape[2]
    w_half = this_canvas.shape[4]
    this_canvas = this_canvas.view(batch_size, num_color_channels, h_half * patch_size_y,
                                   w_half * patch_size_x).contiguous()
    # this_canvas: b, 3, h_half * py, w_half * px
    return this_canvas


def param2img_parallel_with_detail_decision(param: torch.Tensor, decision: torch.Tensor, meta_brushes: torch.Tensor,
                                            cur_canvas: torch.Tensor, details_mask: torch.Tensor,
                                            use_detail_decision: bool):
    """
        Input stroke parameters and decisions for each patch, meta brushes, current canvas, frame directory,
        and whether there is a border (if intermediate painting results are required).
        Output the painting results of adding the corresponding strokes on the current canvas.
        Args:
            param: a tensor with shape batch size x patch along height dimension x patch along width dimension
             x n_stroke_per_patch x n_param_per_stroke
            decision: a 01 tensor with shape batch size x patch along height dimension x patch along width dimension
             x n_stroke_per_patch
            meta_brushes: a tensor with shape 2 x 3 x meta_brush_height x meta_brush_width.
            The first slice on the batch dimension denotes vertical brush and the second one denotes horizontal brush.
            cur_canvas: a tensor with shape batch size x 3 x H x W,
             where H and W denote height and width of padded results of original images.
            details_mask: A tensor with shape batch size x 3 x H x W,
             where H and W denote height and width of padded details mask used to decide whether to render a detailed brushstroke.

        Returns:
            cur_canvas: a tensor with shape batch size x 3 x H x W, denoting painting results.
        """
    # param: b, h, w, stroke_per_patch, param_per_stroke
    # decision: b, h, w, stroke_per_patch
    b, h, w, s, p = param.shape
    num_color_channels = p - 5
    param = param.view(-1, p).contiguous()
    decision = decision.view(-1).contiguous().to(torch.bool)
    H, W = cur_canvas.shape[-2:]
    is_odd_y = h % 2 == 1
    is_odd_x = w % 2 == 1
    patch_size_y = 2 * H // h
    patch_size_x = 2 * W // w
    even_idx_y = torch.arange(0, h, 2, device=cur_canvas.device)
    even_idx_x = torch.arange(0, w, 2, device=cur_canvas.device)
    odd_idx_y = torch.arange(1, h, 2, device=cur_canvas.device)
    odd_idx_x = torch.arange(1, w, 2, device=cur_canvas.device)
    even_y_even_x_coord_y, even_y_even_x_coord_x = torch.meshgrid([even_idx_y, even_idx_x])
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = torch.meshgrid([odd_idx_y, odd_idx_x])
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = torch.meshgrid([even_idx_y, odd_idx_x])
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = torch.meshgrid([odd_idx_y, even_idx_x])
    cur_canvas = F.pad(cur_canvas, [patch_size_x // 4, patch_size_x // 4,
                                    patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0])
    details_mask = F.pad(details_mask, [patch_size_x // 4, patch_size_x // 4,
                                        patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0]) if details_mask else None
    foregrounds = torch.zeros(param.shape[0], num_color_channels, patch_size_y, patch_size_x, device=cur_canvas.device)
    alphas = torch.zeros(param.shape[0], num_color_channels, patch_size_y, patch_size_x, device=cur_canvas.device)
    valid_foregrounds, valid_alphas = param2stroke(param[decision, :], patch_size_y, patch_size_x,
                                                   meta_brushes)

    foregrounds[decision, :, :, :] = valid_foregrounds
    alphas[decision, :, :, :] = valid_alphas
    # foreground, alpha: b * h * w * stroke_per_patch, 3, patch_size_y, patch_size_x
    foregrounds = foregrounds.view(-1, h, w, s, num_color_channels, patch_size_y, patch_size_x).contiguous()
    alphas = alphas.view(-1, h, w, s, num_color_channels, patch_size_y, patch_size_x).contiguous()
    # foreground, alpha: b, h, w, stroke_per_patch, 3, render_size_y, render_size_x
    decision = decision.view(-1, h, w, s, 1, 1, 1).contiguous()

    # decision: b, h, w, stroke_per_patch, 1, 1, 1

    if even_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render_with_details(cur_canvas, foregrounds, alphas, decision, even_y_even_x_coord_y,
                                             even_y_even_x_coord_x,
                                             patch_size_y, patch_size_x, num_color_channels, b, s, h, w, details_mask,
                                             use_detail_decision)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render_with_details(cur_canvas, foregrounds, alphas, decision, odd_y_odd_x_coord_y,
                                             odd_y_odd_x_coord_x,
                                             patch_size_y, patch_size_x, num_color_channels, b, s, h, w, details_mask,
                                             use_detail_decision)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
        canvas = torch.cat([cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render_with_details(cur_canvas, foregrounds, alphas, decision, odd_y_even_x_coord_y,
                                             odd_y_even_x_coord_x,
                                             patch_size_y, patch_size_x, num_color_channels, b, s, h, w, details_mask,
                                             use_detail_decision)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render_with_details(cur_canvas, foregrounds, alphas, decision, even_y_odd_x_coord_y,
                                             even_y_odd_x_coord_x,
                                             patch_size_y, patch_size_x, num_color_channels, b, s, h, w, details_mask,
                                             use_detail_decision)
        canvas = torch.cat([cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, patch_size_x // 4:-patch_size_x // 4]

    return cur_canvas


class FirstStageImagePainter(PaintTransformer):
    def __init__(self, brushstroke_type: BrushstrokeType, num_skip_first_drawing_layers: int = 0,
                 num_skip_last_drawing_layers: int = 0, num_of_detail_layers: int = 1):
        super().__init__(
            model_path=BRUSHSTROKE_TYPE_2_CHECKPOINT[str(brushstroke_type)],
            brush_vertical_path=BRUSHSTROKE_TYPE_2_BRUSHSTROKE_PATHS[str(brushstroke_type)][0],
            brush_horizontal_path=BRUSHSTROKE_TYPE_2_BRUSHSTROKE_PATHS[str(brushstroke_type)][1],
            num_color_channels=3, num_skip_first_drawing_layers=num_skip_first_drawing_layers,
            num_skip_last_drawing_layers=num_skip_last_drawing_layers)
        self.num_of_detail_layers = num_of_detail_layers

    def forward(self, img: torch.Tensor, detail_mask: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            batch_size, channel_size, original_h, original_w = img.shape
            K = max(int(torch.ceil(torch.log2(torch.tensor(max(original_h, original_w) / self.patch_size)))), 0)
            original_img_pad_size = int(self.patch_size * (2 ** int(K)))
            original_img_pad = pad(img, original_img_pad_size, original_img_pad_size)
            detail_mask_pad = pad(detail_mask, original_img_pad_size, original_img_pad_size) if detail_mask else None
            final_result = torch.zeros_like(original_img_pad)
            last_layer = K - self.num_skip_last_drawing_layers
            for layer in range(self.num_skip_first_drawing_layers, last_layer + 1):
                is_detail_layer = K - self.num_of_detail_layers < layer
                layer_size = int(self.patch_size * (2 ** layer))
                img = F.interpolate(original_img_pad, (layer_size, layer_size))
                result = F.interpolate(final_result, (layer_size, layer_size))
                img_patch = F.unfold(img, (self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
                result_patch = F.unfold(result, (self.patch_size, self.patch_size),
                                        stride=(self.patch_size, self.patch_size))
                # There are patch_num * patch_num patches in total
                patch_num = (layer_size - self.patch_size) // self.patch_size + 1

                # img_patch, result_patch: b, 3 * output_size * output_size, h * w
                img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, channel_size, self.patch_size,
                                                                         self.patch_size).contiguous()
                result_patch = result_patch.permute(0, 2, 1).contiguous().view(
                    -1, channel_size, self.patch_size, self.patch_size).contiguous()
                shape_param, stroke_decision = self.network(img_patch, result_patch)
                stroke_decision = network.sign_without_sigmoid_grad(stroke_decision)

                grid = shape_param[:, :, :2].view(img_patch.shape[0] * self.stroke_num, 1, 1, 2).contiguous()
                img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, self.stroke_num, 1, 1, 1).view(
                    img_patch.shape[0] * self.stroke_num, channel_size, self.patch_size, self.patch_size).contiguous()
                color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(
                    img_patch.shape[0], self.stroke_num, channel_size).contiguous()
                stroke_param = torch.cat([shape_param, color], dim=-1)
                # stroke_param: b * h * w, stroke_per_patch, param_per_stroke
                # stroke_decision: b * h * w, stroke_per_patch, 1
                param = stroke_param.view(batch_size, patch_num, patch_num, self.stroke_num,
                                          stroke_param.shape[-1]).contiguous()
                decision = stroke_decision.view(batch_size, patch_num, patch_num, self.stroke_num).contiguous().to(
                    torch.bool)
                # param: b, h, w, stroke_per_patch, 8
                # decision: b, h, w, stroke_per_patch
                param[..., :2] = param[..., :2] / 2 + 0.25
                param[..., 2:4] = param[..., 2:4] / 2
                if torch.any(decision):
                    final_result = param2img_parallel_with_detail_decision(param, decision, self.meta_brushes,
                                                                           final_result, detail_mask_pad,
                                                                           is_detail_layer)

            layer = last_layer
            is_detail_layer = K - self.num_of_detail_layers < layer
            layer_size = int(self.patch_size * (2 ** layer))
            patch_num = (layer_size - self.patch_size) // self.patch_size + 1

            border_size = original_img_pad_size // (2 * patch_num)
            img = F.interpolate(original_img_pad, (layer_size, layer_size))
            result = F.interpolate(final_result, (layer_size, layer_size))
            img = F.pad(img, [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2,
                              0, 0, 0, 0])
            result = F.pad(result,
                           [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2,
                            0, 0, 0, 0])
            img_patch = F.unfold(img, (self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
            result_patch = F.unfold(result, (self.patch_size, self.patch_size),
                                    stride=(self.patch_size, self.patch_size))
            final_result = F.pad(final_result, [border_size, border_size, border_size, border_size, 0, 0, 0, 0])
            detail_mask_pad = F.pad(detail_mask_pad, [border_size, border_size, border_size, border_size, 0, 0, 0,
                                                      0]) if detail_mask else None
            h = (img.shape[2] - self.patch_size) // self.patch_size + 1
            w = (img.shape[3] - self.patch_size) // self.patch_size + 1
            # img_patch, result_patch: b, 3 * output_size * output_size, h * w
            img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, channel_size, self.patch_size,
                                                                     self.patch_size).contiguous()
            result_patch = result_patch.permute(0, 2, 1).contiguous().view(-1, channel_size, self.patch_size,
                                                                           self.patch_size).contiguous()
            shape_param, stroke_decision = self.network(img_patch, result_patch)
            grid = shape_param[:, :, :2].view(img_patch.shape[0] * self.stroke_num, 1, 1, 2).contiguous()
            img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, self.stroke_num, 1, 1, 1).view(
                img_patch.shape[0] * self.stroke_num, channel_size, self.patch_size, self.patch_size).contiguous()
            color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(
                img_patch.shape[0], self.stroke_num, channel_size).contiguous()
            stroke_param = torch.cat([shape_param, color], dim=-1)
            # stroke_param: b * h * w, stroke_per_patch, param_per_stroke
            # stroke_decision: b * h * w, stroke_per_patch, 1
            param = stroke_param.view(batch_size, h, w, self.stroke_num, stroke_param.shape[-1]).contiguous()
            decision = stroke_decision.view(batch_size, h, w, self.stroke_num).contiguous().to(torch.bool)
            # param: b, h, w, stroke_per_patch, 8
            # decision: b, h, w, stroke_per_patch
            param[..., :2] = param[..., :2] / 2 + 0.25
            param[..., 2:4] = param[..., 2:4] / 2
            if torch.any(decision):
                final_result = param2img_parallel_with_detail_decision(param, decision, self.meta_brushes, final_result,
                                                                       detail_mask_pad, is_detail_layer)
            final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]
            final_result = crop(final_result, original_h, original_w)
            final_result = morphology.dilation(final_result)
            final_result = morphology.erosion(final_result)
            return final_result
