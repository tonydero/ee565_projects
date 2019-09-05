import numpy as np
import random


def noisySin(N, noise_var=1):
    """
    a) Write a function named "noisySin" that returns "N" samples drawn from the
    data set.

    Inputs:
                N - desired number of samples
        noise_var - variance for noise to be added to sinusoid
    """
    # generate x coordinate for N uniform samples from 0 to 1
    x_samples = np.random.rand(N, 1)

    # generate Gaussian noise for each of N samples
    noise = noise_var*np.random.randn(N, 1)

    # calculate the sinusoidal amplitude for each of N samples and add noise
    t_samples = np.sin(2*np.pi*x_samples) + noise

    # combine the x coordinate and amplitude for each of the N samples
    samples = np.array((x_samples, t_samples))

    return samples

