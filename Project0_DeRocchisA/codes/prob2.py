import numpy as np
from requiredFunctions.doublemoon import doublemoon
import matplotlib.pyplot as plt


"""
b) Generate a plot of an example double moon distribution N=500 samples,
d=0, r=1, and w=0.6 where members of class C1 are plotted as “blue +” and
C2 are plotted as “green x”.
"""
np.random.seed(83704)
samples = doublemoon(500, 0, 1, 0.6)
samples1 = []
samples2 = []
for sample in samples.T:
    if sample[2] == 1:
        samples1.append(sample)
    else:
        samples2.append(sample)

samples1 = np.array(samples1)
samples2 = np.array(samples2)

plt.axes(aspect=1)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xticks([-3, -2, -1, 0, 1, 2, 3])
plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(samples1[:,0], samples1[:,1], s=33, c='b', marker='+')
plt.scatter(samples2[:,0], samples2[:,1], s=25, c='g', marker='x')
plt.legend(['$target: +1$', '$target: -1$'])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.show()
