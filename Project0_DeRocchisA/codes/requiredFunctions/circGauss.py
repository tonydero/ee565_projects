import numpy as np


def circGauss(N):
    """
    a) Write a function names "circGauss" which returns N samples from a
    circular symmetric multi-variate Gaussian distribution with a specified
    mean and variance.
    """
    samples = {}
    for i in range(mean, variance, N):
        new_sample = np.random.normal(mean, variance)

        samples += new_sample

        return samples

