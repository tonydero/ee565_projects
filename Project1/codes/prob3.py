import numpy as np
from requiredFunctions.kMeans import kMeansBatch
import matplotlib.pyplot as plt
from matplotlib import rc
from distutils.spawn import find_executable
import imageio
from PIL import Image


"""
a-e) Use centroids of each image on itself, then "nature-1.png" on "machine-learning-1.png" and "Nature-Brain.png", then "machine-learning-1.png" on "nature-1.png" and "Nature-Brain.png" on "nature-1.png" each with K=4 and K=8.
"""
for K in [8,16]:
    img_n1 = imageio.imread('../data/nature-1.png')
    img_rgb_n1 = img_n1[:,:,:3]
    n1_x0 = int(img_rgb_n1.shape[0]/3)
    n1_x1 = int(img_rgb_n1.shape[1]/3)
    img_rgb_n1 = np.array(Image.fromarray(img_rgb_n1).resize((n1_x0,n1_x1)))
    plt.imshow(img_rgb_n1)
    plt.axis('off')
    plt.savefig('../p3_n1_third.pdf')
    plt.clf()
    num_points_n1 = n1_x0*n1_x1
    img_rgb_data_n1 = img_rgb_n1.reshape(num_points_n1, img_rgb_n1.shape[2])/255
    results_n1 = kMeansBatch(img_rgb_data_n1, K, iter_ind=True)
    means_n1 = results_n1[0]
    ind_vars_n1 = results_n1[1]

    img_ml1 = imageio.imread('../data/machine-learning-1.png')
    img_rgb_ml1 = img_ml1[:,:,:3]
    ml1_x0 = int(img_rgb_ml1.shape[0]/3)
    ml1_x1 = int(img_rgb_ml1.shape[1]/3)
    img_rgb_ml1 = np.array(Image.fromarray(img_rgb_ml1).resize((ml1_x0,ml1_x1)))
    plt.imshow(img_rgb_ml1)
    plt.axis('off')
    plt.savefig('../p3_ml1_third.pdf')
    plt.clf()
    num_points_ml1 = ml1_x0*ml1_x1
    img_rgb_data_ml1 = img_rgb_ml1.reshape(num_points_ml1, img_rgb_ml1.shape[2])/255
    results_ml1 = kMeansBatch(img_rgb_data_ml1, K, iter_ind=True)
    means_ml1 = results_ml1[0]
    ind_vars_ml1 = results_ml1[1]

    img_nb = imageio.imread('../data/Nature-Brain.png')
    img_rgb_nb = img_nb[:,:,:3]
    nb_x0 = int(img_rgb_nb.shape[0]/3)
    nb_x1 = int(img_rgb_nb.shape[1]/3)
    img_rgb_nb = np.array(Image.fromarray(img_rgb_nb).resize((nb_x0,nb_x1)))
    plt.imshow(img_rgb_nb)
    plt.axis('off')
    plt.savefig('../p3_nb_third.pdf')
    plt.clf()
    num_points_nb = nb_x0*nb_x1
    img_rgb_data_nb = img_rgb_nb.reshape(num_points_nb, img_rgb_nb.shape[2])/255
    results_nb = kMeansBatch(img_rgb_data_nb, K, iter_ind=True)
    means_nb = results_nb[0]
    ind_vars_nb = results_nb[1]

    # find the closest centroid from "nature-1.png" to "machine-learning-1.png" points
    ind_vars_n1_ml1 = np.zeros((num_points_ml1,K))
    distances_n1_ml1 = np.zeros((num_points_ml1,K))
    for j in range(K):
        distances_n1_ml1[:, j] = np.array([pow(np.linalg.norm(point - means_n1[j]),2)
                                    for point in img_rgb_data_ml1]).T
    I_n1_ml1 = np.argmin(distances_n1_ml1, axis=1)
    ind_vars_n1_ml1[range(num_points_ml1), I_n1_ml1[range(num_points_ml1)]] = 1

    ind_vars_n1_nb = np.zeros((num_points_nb,K))
    distances_n1_nb = np.zeros((num_points_nb,K))
    for j in range(K):
        distances_n1_nb[:, j] = np.array([pow(np.linalg.norm(point - means_n1[j]),2)
                                    for point in img_rgb_data_nb]).T
    I_n1_nb = np.argmin(distances_n1_nb, axis=1)
    ind_vars_n1_nb[range(num_points_nb), I_n1_nb[range(num_points_nb)]] = 1

    ind_vars_ml1_n1 = np.zeros((num_points_n1,K))
    distances_ml1_n1 = np.zeros((num_points_n1,K))
    for j in range(K):
        distances_ml1_n1[:, j] = np.array([pow(np.linalg.norm(point - means_ml1[j]),2)
                                    for point in img_rgb_data_n1]).T
    I_ml1_n1 = np.argmin(distances_ml1_n1, axis=1)
    ind_vars_ml1_n1[range(num_points_n1), I_ml1_n1[range(num_points_n1)]] = 1

    ind_vars_nb_n1 = np.zeros((num_points_n1,K))
    distances_nb_n1 = np.zeros((num_points_n1,K))
    for j in range(K):
        distances_nb_n1[:, j] = np.array([pow(np.linalg.norm(point - means_nb[j]),2)
                                    for point in img_rgb_data_n1]).T
    I_nb_n1 = np.argmin(distances_nb_n1, axis=1)
    ind_vars_nb_n1[range(num_points_n1), I_nb_n1[range(num_points_n1)]] = 1

    # create the data for the new plots and create the plots
    n1_data = []
    for ind_point in ind_vars_n1:
        n1_data.append(means_n1[np.nonzero(ind_point)[0]])
    n1_data_img = (np.array(n1_data).reshape(img_rgb_n1.shape)*255).astype('uint8')
    plt.imshow(n1_data_img)
    plt.axis('off')
    plt.savefig('../p3pc_{}bit.pdf'.format(K))
    plt.clf()

    nb_data = []
    for ind_point in ind_vars_nb:
        nb_data.append(means_nb[np.nonzero(ind_point)[0]])
    nb_data_img = (np.array(nb_data).reshape(img_rgb_nb.shape)*255).astype('uint8')
    plt.imshow(nb_data_img)
    plt.axis('off')
    plt.savefig('../p3pb_{}bit.pdf'.format(K))
    plt.clf()

    ml1_data = []
    for ind_point in ind_vars_ml1:
        ml1_data.append(means_ml1[np.nonzero(ind_point)[0]])
    ml1_data_img = (np.array(ml1_data).reshape(img_rgb_ml1.shape)*255).astype('uint8')
    plt.imshow(ml1_data_img)
    plt.axis('off')
    plt.savefig('../p3pa_{}bit.pdf'.format(K))
    plt.clf()

    n1_ml1_data = []
    for ind_point in ind_vars_n1_ml1:
        n1_ml1_data.append(means_n1[np.nonzero(ind_point)[0]])
    n1_ml1_data_img = (np.array(n1_ml1_data).reshape(img_rgb_ml1.shape)*255).astype('uint8')
    plt.imshow(n1_ml1_data_img)
    plt.axis('off')
    plt.savefig('../p3pd_{}bit_a.pdf'.format(K))
    plt.clf()

    n1_nb_data = []
    for ind_point in ind_vars_n1_nb:
        n1_nb_data.append(means_n1[np.nonzero(ind_point)[0]])
    n1_nb_data_img = (np.array(n1_nb_data).reshape(img_rgb_nb.shape)*255).astype('uint8')
    plt.imshow(n1_nb_data_img)
    plt.axis('off')
    plt.savefig('../p3pd_{}bit_b.pdf'.format(K))
    plt.clf()

    ml1_n1_data = []
    for ind_point in ind_vars_ml1_n1:
        ml1_n1_data.append(means_ml1[np.nonzero(ind_point)[0]])
    ml1_n1_data_img = (np.array(ml1_n1_data).reshape(img_rgb_n1.shape)*255).astype('uint8')
    plt.imshow(ml1_n1_data_img)
    plt.axis('off')
    plt.savefig('../p3pe_{}bit_a.pdf'.format(K))
    plt.clf()

    nb_n1_data = []
    for ind_point in ind_vars_nb_n1:
        nb_n1_data.append(means_nb[np.nonzero(ind_point)[0]])
    nb_n1_data_img = (np.array(nb_n1_data).reshape(img_rgb_n1.shape)*255).astype('uint8')
    plt.imshow(nb_n1_data_img)
    plt.axis('off')
    plt.savefig('../p3pe_{}bit_b.pdf'.format(K))
    plt.clf()

