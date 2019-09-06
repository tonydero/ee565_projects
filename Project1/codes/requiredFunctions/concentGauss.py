import numpy as np
import random


def concentGauss(N, r=5, var_center=1, var_outer=1):
    """
    a) Write a function named “concentGauss” that returns samples drawn from the
    distribution along with the corresponding class labels.

    Inputs:
                 N - desired number of samples
                 r - radius mean of Gaussian annulus
        var_center - variance of center circular symmetric Gaussian
         var_outer - variance of Gaussian annulus
    """
    # split N approximately equally
    first_half = int(N/2)
    second_half = N - first_half

    # identify labels for the classes
    label_outer = -1
    label_center = 1

    # generate samples of gaussian annulus
    samples_outer_r = r + var_outer*np.random.randn(first_half,1)
    samples_outer_theta = np.random.rand(first_half,1)*2*np.pi
    samples_outer_x1 = samples_outer_r*np.cos(samples_outer_theta)
    samples_outer_x2 = samples_outer_r*np.sin(samples_outer_theta)

    # generate samples of center circular symmetric gaussian
    samples_center_x1 = var_center*np.random.randn(second_half,1)
    samples_center_x2 = var_center*np.random.randn(second_half,1)

    # combine both classes into single vectors for each coordinate
    samples_x1 = np.append(samples_outer_x1, samples_center_x1)
    samples_x2 = np.append(samples_outer_x2, samples_center_x2)

    # generate label vector
    classes = np.append(label_outer*np.ones((first_half, 1)),
                        label_center*np.ones((second_half, 1)))

    # combine all into single Nx3 array
    samples = np.array((samples_x1,samples_x2,classes))

    return samples

