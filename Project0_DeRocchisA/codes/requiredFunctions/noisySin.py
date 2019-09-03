import numpy as np
import random


def noisySin(N, noise_var=1):
    """
    a) Write a function named "noisySin" that returns "N" samples drawn from the
    data set.
    """
    x_samples = []
    t_samples = []
    for i in range(N):
        x_n = random.uniform(0, 1)
        noise = random.gauss(0, noise_var)
        t_n = np.sin(2*np.pi*x_n) + noise
         
        x_samples.append(x_n)
        t_samples.append(t_n)

    x_samples = np.array(x_samples)
    t_samples = np.array(t_samples)

    samples = np.array((x_samples, t_samples))

    return samples

