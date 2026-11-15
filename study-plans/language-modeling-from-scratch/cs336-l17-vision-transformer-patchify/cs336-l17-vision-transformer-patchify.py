import torch

def vision_transformer_patchify(
    images: torch.Tensor, patch_height: int, patch_width: int,
) -> dict:
    """
    Returns a dict: tokens (image dtype), coordinates (int64 tensor).
    """
    B, C, H, W = images.shape 

    patch_h_count = H // patch_height # Row
    patch_w_count = W // patch_width # Column

    new_image = images.reshape(B, C, patch_h_count, patch_height, patch_w_count, patch_width)

    new_image = new_image.permute(0, 2, 4, 1, 3, 5)
    new_image = new_image.reshape(B, patch_h_count * patch_w_count, C * patch_height * patch_width)

    a,b = torch.meshgrid(torch.arange(patch_h_count, device=images.device), torch.arange(patch_w_count, device=images.device), indexing="ij")
    b = torch.stack([a.flatten(), b.flatten()], dim=1)

    return {"tokens": new_image, "coordinates": b}
