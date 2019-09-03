import numpy as np
import random


def gaussX(N, variance=1):
    """
    a) Write a function named “gaussX” that returns samples drawn from the
    distribution along with the corresponding class labels.
    """
    samples_x1 = []
    samples_x2 = []
    classes = []

    for i in range(N):
        new_sample_r = random.gauss(0, np.sqrt(variance))
        new_sample_theta = random.uniform(0, 2*np.pi)
        new_sample_x1 = new_sample_r*np.cos(new_sample_theta)
        new_sample_x2 = new_sample_r*np.sin(new_sample_theta)
        if (new_sample_x1 > 0 and new_sample_x2 < 0) or\
           (new_sample_x1 < 0 and new_sample_x2 > 0):
            classes.append(1)

        elif (new_sample_x1 > 0 and new_sample_x2 > 0) or\
             (new_sample_x1 < 0 and new_sample_x2 < 0):
            classes.append(-1)

        samples_x1.append(new_sample_x1)
        samples_x2.append(new_sample_x2)

    samples_x1 = np.array(samples_x1)
    samples_x2 = np.array(samples_x2)
    classes = np.array(classes)
    samples = np.array((samples_x1,samples_x2,classes))

    return samples

