import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import csv


"""
Load the file "spikes.csv" and replicate the plot below [fig 13].
"""
# set font attributes
font = {'size'   : 16}
rc('font', **font)
rc('text', usetex='True')

# set up plot
ax = plt.subplot(111)
plt.xlim(-1, 30)
plt.ylim(-1, 2)
plt.xticks(np.arange(5,30,5))
plt.yticks(np.arange(-0.5,2,0.5))
ax.tick_params(direction='inout', length=10, width=2)
# turn off the borders on the top and right sides of the plot to match figure
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# add arrows for axes at left and bottom of plot to match figure
plt.annotate('', xy=(-1,2), xytext=(-1,-1),
             arrowprops=dict(
                             facecolor='black',
                             shrink=2,
                             width=2
                             ))
plt.annotate('', xy=(30,-1), xytext=(-1,-1),
             arrowprops=dict(
                             facecolor='black',
                             shrink=2,
                             width=2
                             ))
# add axes labels to match figure
plt.annotate('$t$', xy=(30,-1), xytext=(30.2,-1.03))
plt.annotate('$Amplitude (\mathrm{x}10^{-4}$)', xy=(-1,2), xytext=(-2.2,2.03))

# read data from file and stack plots of each spike curve
with open('../data/spikes.csv','r') as f:
    spikes_data = csv.reader(f)
    for row in spikes_data:
        t_vals = np.arange(len(row))
        ax.plot(t_vals+1, [float(i)*int(1E4) for i in row])
plt.show()

