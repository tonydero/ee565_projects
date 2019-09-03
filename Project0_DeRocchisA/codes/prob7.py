import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import csv


"""
Load the file "spikes.csv" and replicate the plot below [fig 13].
"""
ax = plt.subplot(111)
plt.xlim(-1, 30)
plt.ylim(-1, 2)
plt.xticks(np.arange(5,30,5))
plt.yticks(np.arange(-0.5,2,0.5))
ax.tick_params(direction='inout', length=10, width=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
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
plt.annotate('t', xy=(30,-1), xytext=(30.2,-1.03))
plt.annotate('Amplitude', xy=(-1,2), xytext=(-2,2.03))
with open('../data/spikes.csv','r') as f:
    spikes_data = csv.reader(f)
    for row in spikes_data:
        t_vals = np.arange(len(row))
        ax.plot(t_vals, [float(i)*int(1E4) for i in row])
plt.show()

