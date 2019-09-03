import numpy as np
import random


def circGauss(N, mean=(0,0), variance=1):
    """
    a) Write a function named "circGauss" which returns N samples from a
    circular symmetric multi-variate Gaussian distribution with a specified
    mean and variance.
    """
    samples_x1 = []
    samples_x2 = []

    for i in range(N):
        new_sample_x1 = random.gauss(0, np.sqrt(variance))
        new_sample_x2 = random.gauss(0, np.sqrt(variance))

        samples_x1.append(new_sample_x1)
        samples_x2.append(new_sample_x2)

    samples_x1 = np.array(samples_x1) + mean[0]
    samples_x2 = np.array(samples_x2) + mean[1]
    samples = np.array((samples_x1,samples_x2))

    return samples

