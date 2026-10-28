import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    # Create a same matrix 
    A = np.asarray(A)
    row, col = A.shape
    B = np.zeros((A.shape[1], A.shape[0]))

    for i in range(0, row):
        for k in range(0, col):
            B[k,i] = A[i,k]
    return B
            