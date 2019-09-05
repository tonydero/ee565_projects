import numpy as np
from requiredFunctions.noisySin import noisySin
import matplotlib.pyplot as plt
from matplotlib import rc


"""
b) Generate a plot showing an example of the distribution N=50 samples, σ²=0.05
where data points are plotted as "blue o" and the clean noise free sinusoid is
plotted in green.
"""
# generate data set
np.random.seed(83704)
samples = noisySin(50, 0.05)

# generate clean sinusoid
x_values = np.arange(0, 1.01, 0.01)
sinusoid = np.sin(2*np.pi*x_values)

# set font attributes
font = {'size'   : 16}
rc('font', **font)
rc('text', usetex='True')

# plot
plt.xlim(-0.03, 1.03)
plt.ylim(-1.5, 1.5)
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(-1.4,1.6,0.2))
plt.tick_params(direction='in', top=1, right=1)
plt.grid(color='k', linestyle='--', alpha=0.2)
plt.scatter(samples[0], samples[1], s=33, facecolors='none', edgecolors='b')
plt.plot(x_values, sinusoid, c='lime')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.show()

"""
c) Load the file "curvefitting.txt" and exactly replicate the plot below
[fig 9].
"""
# load data set from file
with open('../data/curvefitting.txt','r') as f:
    x_data = []
    t_data = []
    for row in f:
        x_point, t_point = row.split()
        x_data.append(float(x_point))
        t_data.append(float(t_point))

# convert to array
curvefitting_data = np.array((x_data,t_data))

# plot
plt.xlim(-0.03, 1.03)
plt.ylim(-1.5, 1.5)
plt.xticks(np.arange(0,2,1))
plt.yticks(np.arange(-1,2,1))
plt.tick_params(direction='in', top=1, right=1)
plt.plot(x_values, sinusoid, c='lime')
# add axes labels to match given plot
plt.annotate('$t$', xy=(-0.03,0.5), xytext=(-0.07,0.5))
plt.annotate('$x$', xy=(0.9,-1.5), xytext=(0.9,-1.7))
plt.scatter(curvefitting_data[0], curvefitting_data[1], s=33,
            facecolors='none', edgecolors='b')
plt.show()

