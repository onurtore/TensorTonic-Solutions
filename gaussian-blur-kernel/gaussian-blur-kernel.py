import math
import numpy as np

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    gk = np.zeros((size, size), dtype=float)
    center = np.floor(size / 2)

    # Write code here
    for i in range(0, size):
        for j in range(0, size):
            offset_x = i - center
            offset_y = j - center
            a = np.square(offset_x) + np.square(offset_y)
            b = 2 * np.square(sigma)
            res = math.exp(-float(a) / b)
            gk[i, j] = res

    gk = gk / gk.sum()
            
    return gk.tolist()
