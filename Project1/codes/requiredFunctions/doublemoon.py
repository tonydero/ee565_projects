import numpy as np
import random


def doublemoon(N, d=0, r=1, w=0.5):
    """
    a) Write a function named “doublemoon” that returns samples drawn from the
    distribution along with the corresponding class labels.

    Inputs:
        N - desired number of samples
        d - distance of bottom crescent from horizontal axis
        r - radius of both crescents and distance of bottom crescent from
            vertical axis
        w - width of each crescent
    """
    # split N approximately equally
    first_half = int(N/2)
    second_half = N - first_half

    # identify labels for the classes
    top_label = 1
    bot_label = -1

    # generate samples of top crescent
    samples_top_r = w*np.random.rand(first_half,1) + r - (w/2)
    samples_top_theta = np.random.rand(first_half,1)*np.pi
    samples_top_x1 = samples_top_r*np.cos(samples_top_theta)
    samples_top_x2 = samples_top_r*np.sin(samples_top_theta)

    # generate samples of bottom crescent
    samples_bot_r = w*np.random.rand(second_half,1) + r - (w/2)
    samples_bot_theta = np.random.rand(second_half,1)*np.pi
    samples_bot_x1 = samples_bot_r*np.cos(samples_bot_theta) + r
    samples_bot_x2 = -samples_bot_r*np.sin(samples_bot_theta)

    # combine both classes into single vectors for each coordinate
    samples_x1 = np.append(samples_top_x1, samples_bot_x1)
    samples_x2 = np.append(samples_top_x2, samples_bot_x2)

    # generate label vector
    classes = np.append(top_label*np.ones((first_half, 1)),
                        bot_label*np.ones((second_half, 1)))

    # combine all into single Nx3 array
    samples = np.stack((samples_x1,samples_x2,classes),axis=-1)

    return samples

