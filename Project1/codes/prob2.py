import numpy as np
from requiredFunctions.circGauss import circGauss
from requiredFunctions.kMeans import kMeansOnline
import matplotlib.pyplot as plt
from matplotlib import rc


"""
a)
"""
# set font attributes
font = {'family' : 'Serif',
        'size'   : 16}
rc('font', **font)

plt.xlim(-10, 300)
plt.ylim(0, 0.0002)
plt.xticks(np.arange(0,350,50))
plt.yticks(np.arange(0,0.0003,0.0001))
plt.tick_params(direction='in', top=1, right=1)

# generate the two Gaussian sets
np.random.seed(83704)
samples_0 = circGauss(250, (0, 0), 3)
samples_5 = circGauss(250, (5, 5), 3)
samples = np.concatenate((samples_0,samples_5),axis=0)
for eta in np.arange(0.00001,0.0002,0.00003):
    epoch_list = []
    for rep in range(10):
        means, centroid_ind, epochs, old_means = kMeansOnline(samples,2,eta,0.0001)
        epoch_list.append(epochs)
    mean_epochs = np.mean(epoch_list)
    print(mean_epochs, eta)
    plt.scatter(mean_epochs, eta, color='g')
plt.xlabel('# Epoch Iterations')
plt.ylabel('$\eta$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.savefig('../p2pa.pdf')
plt.clf()

