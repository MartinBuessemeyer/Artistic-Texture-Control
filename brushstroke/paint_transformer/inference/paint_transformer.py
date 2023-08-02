import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from brushstroke.paint_transformer.inference import morphology
from brushstroke.paint_transformer.inference import network


def param2stroke(param: torch.Tensor, H: int, W: int, meta_brushes: torch.Tensor):
    """
    Input a set of stroke parameters and output its corresponding foregrounds and alpha maps.
    Args:
        param: a tensor with shape n_strokes x n_param_per_stroke. Here, param_per_stroke is 8:
        x_center, y_center, width, height, theta, R, G, and B.
        H: output height.
        W: output width.
        meta_brushes: a tensor with shape 2 x 3 x meta_brush_height x meta_brush_width.
         The first slice on the batch dimension denotes vertical brush and the second one denotes horizontal brush.

    Returns:
        foregrounds: a tensor with shape n_strokes x 3 x H x W, containing color information.
        alphas: a tensor with shape n_strokes x 3 x H x W,
         containing binary information of whether a pixel is belonging to the stroke (alpha mat), for painting process.
    """
    # Firstly, resize the meta brushes to the required shape,
    # in order to decrease GPU memory especially when the required shape is small.
    meta_brushes_resize = F.interpolate(meta_brushes, (H, W))
    b = param.shape[0]
    # Extract shape parameters and color parameters.
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    color = param[:, 5:]
    num_channels = color.shape[1]
    # Pre-compute sin theta and cos theta
    sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    # index means each stroke should use which meta stroke? Vertical meta stroke or horizontal meta stroke.
    # When h > w, vertical stroke should be used. When h <= w, horizontal stroke should be used.
    index = torch.full((b,), -1, device=param.device, dtype=torch.long)
    index[h > w] = 0
    index[h <= w] = 1
    brush = meta_brushes_resize[index.long()]

    # Calculate warp matrix according to the rules defined by pytorch, in order for warping.
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    # Conduct warping.
    grid = F.affine_grid(warp, [b, num_channels, H, W], align_corners=False)
    brush = F.grid_sample(brush, grid, align_corners=False)
    # alphas is the binary information suggesting whether a pixel is belonging to the stroke.
    alphas = (brush > 0).float()
    brush = brush.repeat(1, num_channels, 1, 1)
    alphas = alphas.repeat(1, num_channels, 1, 1)
    # Give color to foreground strokes.
    color_map = color.unsqueeze(-1).unsqueeze(-1)
    foreground = brush * color_map
    # Dilation and erosion are used for foregrounds and alphas respectively to prevent artifacts on stroke borders.
    # foreground = morphology.dilation(foreground)
    # alphas = morphology.erosion(alphas)
    return foreground, alphas


def partial_render(this_canvas: torch.Tensor, foregrounds: torch.Tensor, alphas: torch.Tensor, decision: torch.Tensor,
                   patch_coord_y: torch.Tensor, patch_coord_x: torch.Tensor, patch_size_y: int, patch_size_x: int,
                   num_color_channels: int, batch_size: int, num_strokes: int,
                   h: int, w: int):
    canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x),
                            stride=(patch_size_y // 2, patch_size_x // 2))
    # canvas_patch: b, 3 * py * px, h * w
    canvas_patch = canvas_patch.view(batch_size, num_color_channels, patch_size_y, patch_size_x, h, w).contiguous()
    canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
    # canvas_patch: b, h, w, 3, py, px
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


def param2img_parallel(param: torch.Tensor, decision: torch.Tensor, meta_brushes: torch.Tensor,
                       cur_canvas: torch.Tensor):
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
        canvas = partial_render(cur_canvas, foregrounds, alphas, decision, even_y_even_x_coord_y, even_y_even_x_coord_x,
                                patch_size_y, patch_size_x, num_color_channels, b, s, h, w)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, foregrounds, alphas, decision, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x,
                                patch_size_y, patch_size_x, num_color_channels, b, s, h, w)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
        canvas = torch.cat([cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, foregrounds, alphas, decision, odd_y_even_x_coord_y, odd_y_even_x_coord_x,
                                patch_size_y, patch_size_x, num_color_channels, b, s, h, w)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, foregrounds, alphas, decision, even_y_odd_x_coord_y, even_y_odd_x_coord_x,
                                patch_size_y, patch_size_x, num_color_channels, b, s, h, w)
        canvas = torch.cat([cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, patch_size_x // 4:-patch_size_x // 4]

    return cur_canvas


def read_img(img_path, img_type='RGB', h=None, w=None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img


def pad(img: torch.Tensor, H: int, W: int):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), device=img.device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), device=img.device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), device=img.device)], dim=-1)
    return img


def crop(img: torch.Tensor, h: int, w: int):
    H, W = img.shape[-2:]
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = img[:, :, pad_h:H - pad_h - remainder_h, pad_w:W - pad_w - remainder_w]
    return img


class PaintTransformer(torch.nn.Module):
    def __init__(self, model_path: str, brush_vertical_path: str, brush_horizontal_path: str,
                 num_color_channels: int, num_skip_first_drawing_layers: int = 4, num_skip_last_drawing_layers: int = 0,
                 patch_size: int = 32, stroke_num: int = 8):
        super().__init__()
        self.num_skip_first_drawing_layers = num_skip_first_drawing_layers
        self.num_skip_last_drawing_layers = num_skip_last_drawing_layers

        self.patch_size = patch_size
        self.stroke_num = stroke_num
        brush_large_vertical = read_img(brush_vertical_path, 'L')
        brush_large_horizontal = read_img(brush_horizontal_path, 'L')
        self.register_buffer('meta_brushes', torch.cat([brush_large_vertical, brush_large_horizontal], dim=0),
                             persistent=False)
        self.network = network.Painter(5, stroke_num, 256, 8, 3, 3, input_nc=num_color_channels)
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_size, channel_size, original_h, original_w = img.shape
            K = max(int(torch.ceil(torch.log2(torch.tensor(max(original_h, original_w) / self.patch_size)))), 0)
            original_img_pad_size = int(self.patch_size * (2 ** int(K)))
            original_img_pad = pad(img, original_img_pad_size, original_img_pad_size)
            final_result = torch.zeros_like(original_img_pad)
            last_layer = K - self.num_skip_last_drawing_layers
            for layer in range(self.num_skip_first_drawing_layers, last_layer + 1):
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
                    final_result = param2img_parallel(param, decision, self.meta_brushes, final_result)

            layer = last_layer
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
                final_result = param2img_parallel(param, decision, self.meta_brushes, final_result)
            final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]
            final_result = crop(final_result, original_h, original_w)
            final_result = morphology.dilation(final_result)
            final_result = morphology.erosion(final_result)
            return final_result
