from functools import reduce
from dataclasses import dataclass
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import natten
from einops import rearrange
from .fake_quant import BitLinear
from .rope import AxialRoPE, apply_rotary_emb, downscale_pos, make_axial_pos, scale_for_cosine_sim
from .layers import AdaRMSNorm, FourierFeatures, LabelEmbedder, Level, RMSNorm, LinearGEGLU
from .helpers import apply_wd, filter_params, tag_module, zero_init, checkpoint


@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: GlobalAttentionSpec | NeighborhoodAttentionSpec
    dropout: float
    patch_size: list[int] | None = None


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(BitLinear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 4, self.n_heads) # 3 dims = rotate 3/4 of head
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(BitLinear(d_model, d_model, bias=False)))

    def forward(self, x: torch.Tensor, pos: torch.Tensor, cond: torch.Tensor):
        skip, (l, h, w) = x, x.shape[1:-1]
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... l h w e -> ... (l h w) e").type(qkv.dtype)
        theta = self.pos_emb(pos).movedim(-2, -3)
        q, k, v = rearrange(qkv, "n l h w (t nh e) -> t n nh (l h w) e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh (l h w) e -> n l h w (nh e)", l=l, h=h, w=w)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(BitLinear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 4, self.n_heads) # 3 dims = rotate 3/4 of head
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(BitLinear(d_model, d_model, bias=False)))

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n l h w (t nh e) -> t n l h w nh e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
        theta = self.pos_emb(pos)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)
        x = natten.functional.na3d(q, k, v, self.kernel_size, scale=1.0)
        x = rearrange(x, "n l h w nh e -> n l h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(BitLinear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x) # x.type(skip.dtype) required for bitblas
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.patch_size = patch_size
        mul = reduce(operator.mul, patch_size, 1)
        self.proj = apply_wd(nn.Linear(in_features * mul, out_features, bias=False))

    def forward(self, x):
        match self.patch_size:
            case (l, h, w):
                x = rearrange(x, "... (l nl) (h nh) (w nw) e -> ... l h w (nl nh nw e)", nl=l, nh=h, nw=w)
            case (h, w):
                x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=h, nw=w)
            case _:
                raise ValueError("Invalid patch size")
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.patch_size = patch_size
        mul = reduce(operator.mul, patch_size, 1)
        self.proj = apply_wd(nn.Linear(in_features, out_features * mul, bias=False))

    def forward(self, x):
        x = self.proj(x)
        match self.patch_size:
            case (l, h, w):
                return rearrange(x, "... l h w (nl nh nw e) -> ... (l nl) (h nh) (w nw) e", nl=l, nh=h, nw=w)
            case (h, w):
                return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=h, nw=w)
            case _:
                raise ValueError("Invalid patch size")


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.patch_size = patch_size
        mul = reduce(operator.mul, patch_size, 1)
        self.proj = apply_wd(nn.Linear(in_features, out_features * mul, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        match self.patch_size:
            case (l, h, w):
                x = rearrange(x, "... l h w (nl nh nw e) -> ... (l nl) (h nh) (w nw) e", nl=l, nh=h, nw=w)
            case (h, w):
                x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=h, nw=w)
            case _:
                raise ValueError("Invalid patch size")
        return torch.lerp(skip, x, self.fac.type(x.dtype))
    

class SoSora(nn.Module):
    def __init__(
        self,
        levels: list[LevelSpec],
        mapping: MappingSpec,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        num_classes: int = 0,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = LabelEmbedder(num_classes, mapping.width, class_dropout_prob)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout,)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width, spec_1.patch_size) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width, spec_1.patch_size) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0},
        ]
        return groups

    def _post_process_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, "post_process_weights"):
                print("Post processing weights for module", name)
                module.post_process_weights()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, class_cond: torch.Tensor):
        x = self.patch_in(x)

        pos = make_axial_pos(x.shape[-4], x.shape[-3], x.shape[-2], device=x.device)
        time_emb = self.time_in_proj(self.time_emb(sigma[..., None]))
        class_emb = self.class_emb(class_cond, self.training)
        cond = self.mapping(time_emb + class_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos, merge.patch_size)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)

        return x
