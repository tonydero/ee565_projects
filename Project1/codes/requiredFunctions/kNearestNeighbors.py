import numpy as np


def knn(x_new, data, K):
    """
    Implementation of the K-Nearest Neighbors algorithm.

    Arguments:
              x_new - New point to classify.
               data - Array-like data set to be used to classify x_new.
                  K - Integer number of nearest points to use to classify x_new.
    """
    d = np.array([np.linalg.norm(x_new-point)**2 for point in data[:,:2]])
    I = np.argsort(d)[:K]
    counts = np.unique(data[I,-1], return_counts=True)
    y = int(counts[0][np.argmax(counts[1])])
    return y

