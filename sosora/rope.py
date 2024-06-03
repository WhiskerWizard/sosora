from functools import reduce
import math
import torch
from torch import nn, einsum
from einops import rearrange
from .helpers import compile_wrap


class AxialRoPE(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        log_min, log_max = math.log(math.pi), math.log(10.0 * math.pi)
        # TODO: check if frequency shift by 1 improves performance
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 2 + 1)[:-1].exp()
        self.freqs = nn.Parameter(freqs.view(dim // 2, n_heads).T)

    def forward(self, pos):
        out = einsum("...i,...j->...ij", pos[..., None, :], self.freqs.type(pos.dtype))
        return out.flatten(-2) # (..., dim, freqs) -> (..., dim * freqs)


@compile_wrap
def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    assert rot_dim * 2 <= x.shape[-1], f"few feats: {x.shape[-1]} for rot: {rot_dim}"
    dtype = reduce(torch.promote_types, (x.dtype, freqs.dtype, torch.float32))
    x1, x2, x3 = x[..., :rot_dim], x[..., rot_dim : rot_dim * 2], x[..., rot_dim * 2 :]
    x1, x2, freqs = x1.type(dtype), x2.type(dtype), freqs.type(dtype)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1.type(x.dtype), y2.type(x.dtype), x3), dim=-1)


@compile_wrap
def scale_for_cosine_sim(q: torch.Tensor, k: torch.Tensor, scale: torch.Tensor, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.type(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.type(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.type(q.dtype), k * scale_k.type(k.dtype)


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_axial_pos(l, h, w, dtype=None, device=None) -> torch.Tensor:
    ratio = w / h
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ratio > 1:
        y_min, y_max = -1 / ratio, 1 / ratio
    elif ratio < 1:
        x_min, x_max = -ratio, ratio
    l_pos = centers(-1.0, 1.0, l, dtype=dtype, device=device)
    h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
    w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return torch.stack(torch.meshgrid(l_pos, h_pos, w_pos, indexing="ij"), dim=-1)


def downscale_pos(pos, patch_size: list[int]):
    match patch_size:
        case (h, w):
            pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=h, nw=w)
        case (l, h, w):
            pos = rearrange(pos, "... (l nl) (h nh) (w nw) e -> ... l h w (nl nh nw) e", nl=l, nh=h, nw=w)
        case _:
            raise ValueError(f"Invalid patch size: {patch_size}")
    return torch.mean(pos, dim=-2)
