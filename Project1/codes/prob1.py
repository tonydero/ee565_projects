import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kMeans import kMeansBatch
import matplotlib.pyplot as plt
from matplotlib import rc

num_avgng_iter = 100
"""
Generate a data set consisting of samples drawn from two circular symmetric multi-variate Gaussian distribu-tions in two dimensions. Center one distribution at the origin and one at the point(5, 5). Both distributions should use σ²=3.

a) Varying the number of training points from 10 to 500 to visualize how well we recover the true centers as the centroids.
"""
# initialization
np.random.seed(83704)
min_points = 10
max_points = 500
avgng_num_points = 10
avgng_step = int((max_points - min_points)/avgng_num_points)
avgng_points = range(min_points,max_points+min_points,avgng_step)
# set font attributes
font = {'family' : 'Serif',
        'size'   : 16}
rc('font', **font)

for num_points in avgng_points:
    means_errs = []
    for avgng_iter in range(num_avgng_iter):
        # generate the two Gaussian sets
        samples_0 = circGauss(int(num_points/2), (0, 0), 3)
        samples_5 = circGauss(int(num_points/2), (5, 5), 3)
        samples = np.concatenate((samples_0,samples_5),axis=0)
        means, cluster_ind, tot_iter, old_means = kMeansBatch(samples,2)
        cluster_1 = samples[np.where(cluster_ind[:,0]==1)]
        cluster_2 = samples[np.where(cluster_ind[:,1]==1)]


        means_err = min(np.linalg.norm((5,5)-means[0]) + np.linalg.norm((0,0)-means[1]),
                        np.linalg.norm((5,5)-means[1]) + np.linalg.norm((0,0)-means[0])) 
        means_errs.append(means_err)
    mean_means_err = np.mean(means_errs)
    plt.scatter(num_points,mean_means_err,color='g')
plt.xlabel('# Points')
plt.ylabel('Centroid Error')
plt.savefig('../p1pa.pdf')
plt.clf()

"""
c) Using a data set of 50 points, examining how many iterations it takes for the algorithm to converge, and what is the range.
"""
np.random.seed(83704)
num_points = 50
tot_iters = []
for avgng_iter in range(num_avgng_iter):
    samples_0 = circGauss(int(num_points/2), (0, 0), 3)
    samples_5 = circGauss(int(num_points/2), (5, 5), 3)
    samples = np.concatenate((samples_0,samples_5),axis=0)
    means, cluster_ind, tot_iter, old_means = kMeansBatch(samples,2)
    tot_iters.append(tot_iter)
mean_iters = np.mean(tot_iters)
min_iters = min(tot_iters)
max_iters = max(tot_iters)

print('Mean # Iters: ', mean_iters)
print(' Min # Iters: ', min_iters)
print(' Max # Iters: ', max_iters)

"""
d) Using a data set of 50 points, examining the effect of varying K from 2 to 20 using the cost function, J.
"""
np.random.seed(83704)
num_points = 50
avgng_step = int((max_points - min_points)/avgng_num_points)
avgng_points = range(min_points,max_points+min_points,avgng_step)
# set font attributes
font = {'family' : 'Serif',
        'size'   : 16}
rc('font', **font)

for K in range(2,20,1):
    costs = []
    for avgng_iter in range(num_avgng_iter):
        # generate the two Gaussian sets
        samples_0 = circGauss(int(num_points/2), (0, 0), 3)
        samples_5 = circGauss(int(num_points/2), (5, 5), 3)
        samples = np.concatenate((samples_0,samples_5),axis=0)
        means, cluster_ind, tot_iter, old_means, cost = kMeansBatch(samples,K, return_cost=True)
        costs.append(cost)

    mean_cost = np.mean(costs)
    plt.scatter(K,mean_cost,color='g')
plt.xlabel('K')
plt.ylabel('J')
plt.savefig('../p1pd.pdf')
plt.clf()

"""
d) Using a data set of 50 points with K=5 and clustering it 3 times to examine the effect of different initializations.
"""
np.random.seed(83704)
num_points = 50
# set font attributes
font = {'family' : 'Serif',
        'size'   : 16}
rc('font', **font)

np.random.seed(56527382)
samples_0 = circGauss(int(num_points/2), (0, 0), 3)
samples_5 = circGauss(int(num_points/2), (5, 5), 3)
samples = np.concatenate((samples_0,samples_5),axis=0)
means, cluster_ind, tot_iter, old_means = kMeansBatch(samples,5,rand_init=False,init_values=np.zeros((5,2)))
cluster_1 = samples[np.where(cluster_ind[:,0]==1)]
cluster_2 = samples[np.where(cluster_ind[:,1]==1)]
plt.axes(aspect=1)
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.xticks([-5, 0, 5, 10])
plt.yticks([-5, 0, 5, 10])
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(cluster_1[:,0], cluster_1[:,1], s=3, c='b')
plt.scatter(cluster_2[:,0], cluster_2[:,1], s=3, c='lime')
plt.scatter(means[:,0], means[:,1], s=60, c='r', marker='+')
plt.scatter(old_means[0,:,0], old_means[0,:,1], s=60, c='orange', marker='x')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.savefig('../p1pe_0.pdf')
plt.clf()

np.random.seed(56527382)
samples_0 = circGauss(int(num_points/2), (0, 0), 3)
samples_5 = circGauss(int(num_points/2), (5, 5), 3)
samples = np.concatenate((samples_0,samples_5),axis=0)
means, cluster_ind, tot_iter, old_means = kMeansBatch(samples,5,rand_init=False,init_values=np.array(((0,0),(100,0),(0,100),(100,100),(-100,-100))))
cluster_1 = samples[np.where(cluster_ind[:,0]==1)]
cluster_2 = samples[np.where(cluster_ind[:,1]==1)]
plt.axes(aspect=1)
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.xticks([-5, 0, 5, 10])
plt.yticks([-5, 0, 5, 10])
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(cluster_1[:,0], cluster_1[:,1], s=3, c='b')
plt.scatter(cluster_2[:,0], cluster_2[:,1], s=3, c='lime')
plt.scatter(means[:,0], means[:,1], s=60, c='r', marker='+')
plt.scatter(old_means[0,:,0], old_means[0,:,1], s=60, c='orange', marker='x')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.savefig('../p1pe_1.pdf')
plt.clf()

np.random.seed(56527382)
samples_0 = circGauss(int(num_points/2), (0, 0), 3)
samples_5 = circGauss(int(num_points/2), (5, 5), 3)
samples = np.concatenate((samples_0,samples_5),axis=0)
means, cluster_ind, tot_iter, old_means = kMeansBatch(samples,5,rand_init=False,init_values=np.array(((1,0),(4,3),(3,4),(3,1),(-3,-5))))
cluster_1 = samples[np.where(cluster_ind[:,0]==1)]
cluster_2 = samples[np.where(cluster_ind[:,1]==1)]
plt.axes(aspect=1)
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.xticks([-5, 0, 5, 10])
plt.yticks([-5, 0, 5, 10])
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(cluster_1[:,0], cluster_1[:,1], s=3, c='b')
plt.scatter(cluster_2[:,0], cluster_2[:,1], s=3, c='lime')
plt.scatter(means[:,0], means[:,1], s=60, c='r', marker='+')
plt.scatter(old_means[0,:,0], old_means[0,:,1], s=60, c='orange', marker='x')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.savefig('../p1pe_2.pdf')
plt.clf()

