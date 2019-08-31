import numpy as np


def circGauss(N, mean=(0,0), variance=1):
    """
    a) Write a function named "circGauss" which returns N samples from a
    circular symmetric multi-variate Gaussian distribution with a specified
    mean and variance.
    """
    samples_x1 = []
    samples_x2 = []
    np.random.seed(83704)

    for i in range(N):
        new_sample_x1 = np.random.normal(mean[0], variance)
        new_sample_x2 = np.random.normal(mean[1], variance)

        samples_x1.append(new_sample_x1)
        samples_x2.append(new_sample_x2)

    samples_x1 = np.array(samples_x1)
    samples_x2 = np.array(samples_x2)
    samples = np.array((samples_x1,samples_x2))

    return samples

