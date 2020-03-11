import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    kernel_matrix = (X @ Y.T + c) ** p
    return kernel_matrix
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    n, m = X.shape[0], Y.shape[0]
    # kernel_matrix = np.zeros((n,m))
    # for i in range(n):
    #     for j in range(m):
    #         kernel_matrix[i,j] = np.exp(-gamma * sum((X[i]-Y[j])**2))

    # print("X=",X)
    # print("Y=",Y)
    
    # matrix1 = np.tile((X @ X.T).sum(axis=1),m).reshape(m,n).T # n,m
    # matrix2 = np.tile((Y @ Y.T).sum(axis=1),n).reshape(n,m)

    matrix1 = np.tile((X**2).sum(axis=1), m).reshape(m, n).T
    matrix2 = np.tile((Y**2).sum(axis=1), n).reshape(n, m)

    # print("matrix1=", matrix1)
    # print("matrix2=", matrix2)

    matrix3 = -2* X @ Y.T
    kernel_matrix = np.exp(-gamma * (matrix1 + matrix2 + matrix3))

    return kernel_matrix
    raise NotImplementedError
