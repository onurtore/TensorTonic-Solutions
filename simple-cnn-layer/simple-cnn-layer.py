import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)

    n, c_in, h, w = x.shape
    c_out, c_in, k_h, k_w = W.shape

    print("Batch size is: ", n)
    print("Number of input channel is: ", c_in)
    print("Input height is: ", h)
    print("Input width is: ", w)
    print(10*'-')
    print("Number of output channels is: ", c_out)
    print("Number of input channels is: ", c_in)
    print("Kernel height is: ", k_h)
    print("Kernel width is: ", k_w)

    # Out
    output_height = h - k_h + 1
    output_width  = w - k_w + 1
    output_batch = n
    output_channel = c_out

    output = np.zeros((output_batch, output_channel, output_height, output_width))

    for batch_idx in range(output.shape[0]):
        for c in range(output.shape[1]):
            for row in range(output.shape[2]):
                for col in range(output.shape[3]):
                    patch = x[batch_idx, :, row:row+k_h, col:col+k_w]
                    kernel = W[c, :, :,:]
                    output[batch_idx, c, row, col] = np.sum(patch * kernel) + b[c]
    return output
