def color_to_grayscale(image: list) -> list:
    """
    Returns the luminance value of every RGB pixel.
    """
    # Write code here
    import numpy as np
    image = np.asarray(image, dtype=float)
    Y = 0.299 * image[:,:,0] + 0.587 * image[:,:, 1] + 0.114 * image[:, :, 2] 
    return Y.tolist()