import numpy as np
import random


def concentGauss(N, r=5, var_center=1, var_outer=1):
    """
    a) Write a function named “concentGauss” that returns samples drawn from the
    distribution along with the corresponding class labels.
    """
    samples_x1 = []
    samples_x2 = []
    classes = []

    for i in range(N):
        if i%2 == 0:
            new_sample_x1 = random.gauss(0, np.sqrt(var_center))
            new_sample_x2 = random.gauss(0, np.sqrt(var_center))

            classes.append(1)

        elif i%2 == 1:
            new_sample_r = random.gauss(r, np.sqrt(var_outer))
            new_sample_theta = random.uniform(0, 2*np.pi)
            new_sample_x1 = new_sample_r*np.cos(new_sample_theta)
            new_sample_x2 = new_sample_r*np.sin(new_sample_theta)

            classes.append(-1)

        samples_x1.append(new_sample_x1)
        samples_x2.append(new_sample_x2)

    samples_x1 = np.array(samples_x1)
    samples_x2 = np.array(samples_x2)
    classes = np.array(classes)
    samples = np.array((samples_x1,samples_x2,classes))

    return samples

