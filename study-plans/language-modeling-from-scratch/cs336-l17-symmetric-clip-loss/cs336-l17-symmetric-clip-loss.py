import torch as t

def symmetric_clip_loss(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: int | float,
) -> dict:
    """
    Returns a dict: loss and two directional accuracies (scalars), similarities (tensor).
    """
    
    batch_size = image_embeddings.shape[0]

    # Each row is an embedding.
    def _normalize(x: t.Tensor) -> t.Tensor:
        return t.div(x, t.clamp(t.linalg.norm(x, dim=1, keepdim=True), min=1e-12))

    n_img_e = _normalize(image_embeddings)
    n_txt_e = _normalize(text_embeddings)

    similarity_matrix = t.div(t.matmul(n_img_e, n_txt_e.T), temperature)


    a = t.diag(t.log(t.div(t.exp(similarity_matrix) ,t.sum(t.exp(similarity_matrix), dim=1, keepdim=True))))
    b = t.diag(t.log(t.div(t.exp(similarity_matrix) ,t.sum(t.exp(similarity_matrix), dim=0, keepdim=True))))
    loss = -1 / (2 * batch_size) * t.sum(a + b)

    image_to_text_accuracy = (1 / batch_size) * t.sum(t.argmax(similarity_matrix, dim=1) == t.arange(batch_size,  device=similarity_matrix.device)).to(similarity_matrix.dtype)
    text_to_image_accuracy = (1 / batch_size) * t.sum(t.argmax(similarity_matrix, dim=0) == t.arange(batch_size,  device=similarity_matrix.device)).to(similarity_matrix.dtype)


    # Now its now turn to implement similarity
    return {"loss": loss, 
            "image_to_text_accuracy": image_to_text_accuracy,
            "text_to_image_accuracy": text_to_image_accuracy,
            "similarities": similarity_matrix}
