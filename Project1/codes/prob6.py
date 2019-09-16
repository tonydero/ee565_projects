import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from requiredFunctions.concentGauss import concentGauss
from requiredFunctions.kNearestNeighbors import knn


"""
Generate a data set consisting of N = 500 samples drawn from a concentric
Gaussians dataset with \sigma^2_{center}=1, r=5, and \sigma^2_{outer}=1.
"""
# generate data set
data_set = concentGauss(500,5,1,1)

# set font attributes
font = {'size'   : 16}
rc('font', **font)

# plot
x1_min = -10
x1_max = 10
x2_min = -10
x2_max = 10
plt.axes(aspect=1)
plt.xlim(-11, 11)
plt.ylim(-11, 11)
plt.xticks(np.arange(x1_min,x1_max+1,2))
plt.yticks(np.arange(x2_min,x2_max+1,2))
plt.tick_params(direction='in', top=1, right=1)
step_int = 4
for x1,x2 in np.ndindex(((2*x1_max)*step_int,(2*x2_max)*step_int)):
    new_x = ((x1+x1_min*step_int)/step_int,(x2+x2_min*step_int)/step_int)
    y = knn(new_x,data_set,15)
    if y == -1:
        class_color = 'lime'
        class_marker = 'x'
    else:
        class_color = 'b'
        class_marker = '+'
    plt.scatter(new_x[0], new_x[1], s=40, c=class_color, marker=class_marker)
#plt.show()

