import numpy as np
from requiredFunctions.gaussX import gaussX
import matplotlib.pyplot as plt
from matplotlib import rc


"""
b) Generate a plot showing an exampleof the distribution N=500 samples, σ²=1,
where members of class C₁ are plotted as “blue +” and C₂ are plotted as
“green x”.
"""
# generate data set
np.random.seed(83704)
samples = gaussX(500, 1)

# split into classes
samples1 = []
samples2 = []
for sample in samples.T:
    if sample[2] == 1:
        samples1.append(sample)
    else:
        samples2.append(sample)

# convert back to arrays
samples1 = np.array(samples1)
samples2 = np.array(samples2)

# set font attributes
font = {'size'   : 16}
rc('font', **font)
rc('text', usetex='True')

# plot
plt.axes(aspect=1)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(samples1[:,0], samples1[:,1], s=33, c='b', marker='+')
plt.scatter(samples2[:,0], samples2[:,1], s=25, c='lime', marker='x')
plt.legend(['$target: +1$', '$target: -1$'])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.show()
