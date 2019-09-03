import numpy as np
import matplotlib.pyplot as plt


"""
Load the file "faithful.txt" and replicate the plot below [fig 11].
"""
with open('../data/faithful.txt','r') as f:
    x1_data = []
    x2_data = []
    for row in f:
        x1_point, x2_point = row.split()
        x1_data.append(float(x1_point))
        x2_data.append(float(x2_point))

faithful_data = np.array((x1_data,x2_data))

plt.axes(aspect=0.08)
plt.xlim(1, 6)
plt.ylim(40, 100)
plt.xticks(np.arange(1,7,1))
plt.yticks(np.arange(40,110,10))
plt.tick_params(direction='in', top=1, right=1)
plt.scatter(faithful_data[0], faithful_data[1], s=40, facecolors='none',
            edgecolors='lime')
plt.show()

