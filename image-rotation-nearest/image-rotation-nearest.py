import math

def rotate_image(image: list, angle_degrees: float) -> list:
    """
    Returns the counterclockwise nearest-neighbor rotation.
    """
    # Write code here
    H = len(image) 
    W = len(image[0])

    output = [[0] * W for _ in range(H)]

    # Write code here
    center_x = (W - 1) / 2
    center_y = (H - 1) / 2

    q = angle_degrees * math.pi / 180

    for i in range(H):
        for j in range(W):
            dx = j - center_x
            dy = i - center_y

            x = round(center_y + dy * math.cos(q) + dx * math.sin(q))
            y = round(center_x - dy * math.sin(q) + dx * math.cos(q))

            if 0 <= x < H and 0 <= y < W:
                output[i][j] = image[x][y]

    return output