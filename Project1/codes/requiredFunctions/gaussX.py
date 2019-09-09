import numpy as np
import random


def gaussX(N, variance=1):
    """
    a) Write a function named “gaussX” that returns samples drawn from the
    distribution along with the corresponding class labels.

    Inputs:
               N - desired number of samples
        variance - variance of desired circular symmetric Gaussian
    """
    # split N approximately equally
    first_half = int(N/2)
    second_half = N - first_half

    # identify labels for the classes
    label_13 = -1
    label_24 = 1

    # generate samples of circular symmetric Gaussian for 2nd and 4th quadrants
    samples_24_r = np.sqrt(variance)*np.random.randn(first_half,1)
    samples_24_theta = -np.random.rand(first_half,1)*(np.pi/2)
    samples_24_x1 = samples_24_r*np.cos(samples_24_theta)
    samples_24_x2 = samples_24_r*np.sin(samples_24_theta)

    # generate samples of circular symmetric Gaussian for 1st and 3rd quadrants
    samples_13_r = np.sqrt(variance)*np.random.randn(first_half,1)
    samples_13_theta = np.random.rand(first_half,1)*(np.pi/2)
    samples_13_x1 = samples_13_r*np.cos(samples_13_theta)
    samples_13_x2 = samples_13_r*np.sin(samples_13_theta)

    # combine both classes into single vectors for each coordinate
    samples_x1 = np.append(samples_24_x1, samples_13_x1)
    samples_x2 = np.append(samples_24_x2, samples_13_x2)

    # generate label vector
    classes = np.append(label_24*np.ones((first_half, 1)),
                        label_13*np.ones((second_half, 1)))

    # combine all into single Nx3 array
    samples = np.concatenate((samples_x1,samples_x2,classes),axis=1)

    return samples

