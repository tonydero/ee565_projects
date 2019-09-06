import numpy as np
from requiredFunctions.circGauss import circGauss
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
plt.scatter(samples_0[0], samples_0[1], s=3, c='b')
plt.scatter(samples_5[0], samples_5[1], s=3, c='b')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.show()

