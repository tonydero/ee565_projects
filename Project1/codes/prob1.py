import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kmeans import kMeansBatch
import matplotlib.pyplot as plt
from matplotlib import rc


"""
b) Use your function to generate a data set consisting of samples drawn from
two circular symmetric multi-variate Gaussian distribu-tions in two dimensions.
Center one distribution at the origin andone at the point(5, 5). Both
distributions should use σ²=3. Assume that samples drawn from the two circular
distributions in equal proportions.
"""
# generate the two Gaussian sets
np.random.seed(83704)
samples_0 = circGauss(250, (0, 0), 3)
samples_5 = circGauss(250, (5, 5), 3)
samples = np.concatenate((samples_0,samples_5),axis=0)
means, cluster_ind, total_iter = kMeansBatch(samples,2)
cluster_1 = samples[np.where(cluster_ind[:,0]==1)]
cluster_2 = samples[np.where(cluster_ind[:,1]==1)]

# set font attributes
font = {'family' : 'Serif',
        'size'   : 16}
rc('font', **font)
rc('text', usetex='True')

# plot
plt.axes(aspect=1)
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.xticks([-5, 0, 5, 10])
plt.yticks([-5, 0, 5, 10])
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(cluster_1[:,0], cluster_1[:,1], s=3, c='b')
plt.scatter(cluster_2[:,0], cluster_2[:,1], s=3, c='lime')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.show()

