import torch
import torch.nn.functional as F


def rf_forward(model, x: torch.Tensor, cond: torch.Tensor):
    t = torch.sigmoid(torch.randn(x.size(0))).to(x.device)
    t_exp = t[(...,) + (None,) * (x.dim() - 1)]
    z1 = torch.randn_like(x)
    zt = (1 - t_exp) * x + t_exp * z1
    v_t = model(zt, t, cond)
    return F.mse_loss(z1, x + v_t)


@torch.inference_mode()
def rf_sample(model, z: torch.Tensor, cond: torch.Tensor, n_cond: torch.Tensor | None = None, steps: int = 50, cfg: float = 2.0):
    for t in torch.linspace(1.0, 0.0, steps):
        t = torch.full([z.size(0)], t).to(z.device)
        v_c = model(z, t, cond)
        if n_cond is not None:
            v_u = model(z, t, n_cond)
            v_c = v_u + cfg * (v_c - v_u)
        z = z - v_c * (1.0 / steps)
    return z
