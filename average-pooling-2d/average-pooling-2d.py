def average_pooling_2d(X: list, pool_size: int) -> list:    
    """
    Returns non-overlapping average-pooled windows.
    """
    # Write code here
    curr_x = 0
    curr_y = 0
    values = []
    rows = len(X)
    cols = len(X[0])


    def find_avg(X, curr_x, curr_y):
        value = 0
        for i in range(curr_x, curr_x + pool_size):
            for j in range(curr_y, curr_y + pool_size):
                value += X[i][j]

        return value / (pool_size ** 2)


    while curr_x  <= rows - pool_size:
        tmp_list = []

        while curr_y <= cols - pool_size:

            value = find_avg(X, curr_x, curr_y)
            tmp_list.append(value)
            curr_y += pool_size

        values.append(tmp_list)
        curr_x += pool_size
        curr_y = 0

    return values

