import torch
import numpy as np

def edm_sampler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = t_steps.permute(3, 0, 1, 2)
    
    x_next = latents
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        t_cur = t_cur.unsqueeze(-1)
        t_next = t_next.unsqueeze(-1)

        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + 2 * (t_next - t_cur) * d_cur  + torch.sqrt(2 * (t_cur - t_next).abs() * t_cur) * torch.randn_like(x_cur)

    return x_next