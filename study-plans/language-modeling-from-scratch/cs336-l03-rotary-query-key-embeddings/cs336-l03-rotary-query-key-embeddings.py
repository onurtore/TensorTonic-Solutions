import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    B = q.shape[0]
    H = q.shape[1]
    S = q.shape[2]
    D = q.shape[3]
    half = D // 2

    positions_ = positions.reshape(-1, 1, S, 1).expand(B, H, S, half)
    inv_freq_ = inv_freq.reshape(1, half).expand(S, half)
    theta = torch.mul(positions_, inv_freq_)

    q = q.reshape(B, H, S, D // 2, 2)
    k = k.reshape(B, H, S, D // 2, 2)

    q_ = q.clone()
    k_ = k.clone()

    q_[:,:,:,:,0] = q[:,:,:,:,0] * torch.cos(theta) - q[:,:,:,:,1] * torch.sin(theta)
    q_[:,:,:,:,1] = q[:,:,:,:,0] * torch.sin(theta) + q[:,:,:,:,1] * torch.cos(theta)


    k_[:,:,:,:,0] = k[:,:,:,:,0] * torch.cos(theta) - k[:,:,:,:,1] * torch.sin(theta)
    k_[:,:,:,:,1] = k[:,:,:,:,0] * torch.sin(theta) + k[:,:,:,:,1] * torch.cos(theta)


    return {
        "q_rotated": q_.flatten(-2),
        "k_rotated": k_.flatten(-2)
    }