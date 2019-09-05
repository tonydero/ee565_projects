import numpy as np
import random


def circGauss(N, mean=(0,0), variance=1):
    """
    a) Write a function named "circGauss" which returns N samples from a
    circular symmetric multi-variate Gaussian distribution with a specified
    mean and variance.
    
    Inputs:
               N - desired number of samples
            mean - two tuple of desired mean value
        variance - desired variance value
    """
    # generate samples of circular symmetric multi-variate Gaussian for
    # each coordinate
    samples_x1 = np.sqrt(variance)*np.random.randn(N, 1) + mean[0]
    samples_x2 = np.sqrt(variance)*np.random.randn(N, 1) + mean[1]

    # combine into single Nx2 array
    samples = np.array((samples_x1,samples_x2))

    return samples

