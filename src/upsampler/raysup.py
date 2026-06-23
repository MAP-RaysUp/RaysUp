import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.layers.convolutions import ImageEncoderDecoupled

try:
    from natten import na2d
except ImportError:
    na2d = None


def functional_cross_attention(q, k, v, num_heads, kernel_size=(7, 7), use_global=False):
    dim = q.shape[1]
    head_dim = dim // num_heads
    scale = head_dim**-0.5
    query_h, query_w = q.shape[-2:]

    if use_global:
        q = rearrange(q, "b (h c) x y -> b h (x y) c", h=num_heads)
        k = rearrange(k, "b (h c) x y -> b h (x y) c", h=num_heads)
        v = rearrange(v, "b (h c) x y -> b h (x y) c", h=num_heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        return rearrange(out, "b h (x y) c -> b (h c) x y", x=query_h, y=query_w)

    if na2d is None:
        raise ImportError("NATTEN is required when use_global=False.")

    key_h, key_w = k.shape[-2:]
    dilation = (max(1, query_h // key_h), max(1, query_w // key_w))

    def resize_for_natten(x, size, dtype):
        x = F.interpolate(x, size=size, mode="nearest-exact")
        x = rearrange(x, "b (n d) h w -> b h w n d", n=num_heads)
        return x.to(dtype)

    q_in = rearrange(q, "b (n d) h w -> b h w n d", n=num_heads)
    k_in = resize_for_natten(k, size=(query_h, query_w), dtype=q.dtype)
    v_in = resize_for_natten(v, size=(query_h, query_w), dtype=q.dtype)
    out = na2d(
        q_in,
        k_in,
        v_in,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=1,
        backend="cutlass-fna",
    )
    return rearrange(out, "b h w n d -> b (n d) h w")


class ImageTokenizer(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.img_encoder = ImageEncoderDecoupled(
            in_channels=3,
            out_channels=out_channels,
            img_layers=2,
            use_encoder=True,
        )

    def forward(self, x):
        return self.img_encoder(x)


def get_frequency(num_freqs, max_period, min_period):
    log_min_frequency = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max_frequency = torch.log(torch.tensor(2 * torch.pi / min_period))
    return torch.exp(torch.linspace(log_min_frequency, log_max_frequency, num_freqs))


def prepare_rope_coeffs(positions, num_freqs, freq_base, device):
    rope_angles = []
    max_periods = {
        "p0": 4.0,
        "pinf": 8.0,
    }

    for name, position in positions.items():
        max_period = max_periods.get(name)
        if max_period is None or num_freqs < 1:
            continue

        min_period = max_period / (freq_base ** max(1, num_freqs - 1))
        freqs = get_frequency(num_freqs, max_period, min_period).to(device)
        rope_angles.append(torch.einsum("f,...d->...fd", freqs, position))

    if not rope_angles:
        return None, None

    rope_angles = torch.cat(rope_angles, dim=-1)
    rope_angles = rope_angles.reshape(rope_angles.shape[:-2] + (-1,))
    return torch.cos(rope_angles), torch.sin(rope_angles)


def compute_rope_coeffs(ray_o, ray_d, head_dim, freq_base=3.0):
    device = ray_o.device
    batch_size, _, height, width = ray_o.shape
    positions = {
        "p0": ray_o.permute(0, 2, 3, 1),
        "pinf": ray_d.permute(0, 2, 3, 1),
    }

    coord_dim = 6
    rope_dim = head_dim // 2
    num_freqs = rope_dim // coord_dim

    if num_freqs == 0:
        ones = torch.ones(batch_size, height, width, rope_dim, device=device)
        zeros = torch.zeros(batch_size, height, width, rope_dim, device=device)
        return ones, zeros

    cos, sin = prepare_rope_coeffs(positions, num_freqs, freq_base, device)
    current_dim = cos.shape[-1]
    if current_dim < rope_dim:
        pad_dim = rope_dim - current_dim
        cos = torch.cat([cos, torch.ones(batch_size, height, width, pad_dim, device=device)], dim=-1)
        sin = torch.cat([sin, torch.zeros(batch_size, height, width, pad_dim, device=device)], dim=-1)

    return cos, sin


def apply_rope(feats, cos, sin, inverse=False):
    x1, x2 = torch.chunk(feats, 2, dim=-1)

    if cos.dim() == 4:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    if inverse:
        x_rotated_1 = x1 * cos + x2 * sin
        x_rotated_2 = -x1 * sin + x2 * cos
    else:
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos

    return torch.cat([x_rotated_1, x_rotated_2], dim=-1)


class RaysUp(nn.Module):
    def __init__(
        self,
        input_dim=3,
        qk_dim=256,
        v_dim=384,
        feature_dim=None,
        kernel_size=1,
        num_heads=4,
        use_spe=True,
        use_rope=True,
        rope_freq_base=3.0,
        **kwargs,
    ):
        super().__init__()
        del input_dim, v_dim, feature_dim, kernel_size, kwargs

        self.use_spe = use_spe
        self.use_rope = use_rope
        self.rope_freq_base = rope_freq_base
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.image_tokenizer = ImageTokenizer(qk_dim) if use_spe else nn.Conv2d(3, qk_dim, kernel_size=1, bias=False)

    def forward(self, input_image, features, poses, output_size):
        c2w, fxfycxcy = poses
        _, _, feat_h, feat_w = features.shape

        encoded_image = self.image_tokenizer(input_image)
        queries = F.adaptive_avg_pool2d(encoded_image, output_size=(output_size[0], output_size[1]))
        keys = F.adaptive_avg_pool2d(encoded_image, output_size=(feat_h, feat_w))

        if self.use_rope:
            ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, h=output_size[0], w=output_size[1], device=input_image.device)
            key_ray_o, key_ray_d = self.compute_rays(c2w, fxfycxcy, h=feat_h, w=feat_w, device=input_image.device)
            head_dim = self.qk_dim // self.num_heads
            cos_q, sin_q = compute_rope_coeffs(ray_o, ray_d, head_dim, self.rope_freq_base)
            cos_k, sin_k = compute_rope_coeffs(key_ray_o, key_ray_d, head_dim, self.rope_freq_base)

            queries = rearrange(queries, "b (h c) x y -> b h x y c", h=self.num_heads)
            queries = apply_rope(queries, cos_q, sin_q, inverse=True)
            queries = rearrange(queries, "b h x y c -> b (h c) x y")

            keys = rearrange(keys, "b (h c) x y -> b h x y c", h=self.num_heads)
            keys = apply_rope(keys, cos_k, sin_k, inverse=True)
            keys = rearrange(keys, "b h x y c -> b (h c) x y")

        return functional_cross_attention(
            queries,
            keys,
            features,
            num_heads=self.num_heads,
            kernel_size=(6, 6),
            use_global=False,
        )

    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        batch_size = c2w.size(0)
        fx, fy, cx, cy = fxfycxcy[:, 0], fxfycxcy[:, 1], fxfycxcy[:, 2], fxfycxcy[:, 3]
        original_h = int(2 * cy.max().item())
        original_w = int(2 * cx.max().item())

        if h is None or w is None:
            h, w = original_h, original_w

        if original_h != h or original_w != w:
            fx = fx * w / original_w
            fy = fy * h / original_h
            cx = cx * w / original_w
            cy = cy * h / original_h

        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        x = x[None, :, :].expand(batch_size, -1, -1).reshape(batch_size, -1)
        y = y[None, :, :].expand(batch_size, -1, -1).reshape(batch_size, -1)
        x = (x + 0.5 - cx.unsqueeze(1)) / fx.unsqueeze(1)
        y = (y + 0.5 - cy.unsqueeze(1)) / fy.unsqueeze(1)
        z = torch.ones_like(x)

        ray_d = torch.stack([x, y, z], dim=2)
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

        ray_o = rearrange(ray_o, "b (h w) c -> b c h w", b=batch_size, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "b (h w) c -> b c h w", b=batch_size, h=h, w=w, c=3)
        return ray_o, ray_d
