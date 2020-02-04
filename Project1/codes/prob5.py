import numpy as np
from requiredFunctions.polyfitLS import polyfitLS, polyfitRegLS
from requiredFunctions.noisySin import noisySin
import matplotlib.pyplot as plt
from matplotlib import rc
from distutils.spawn import find_executable


"""
a) Recreate Figure 1.7
"""
# generate data set
np.random.seed(83704)

# generate clean sinusoid
x_values = np.arange(0, 1.01, 0.01)
sinusoid = np.sin(2*np.pi*x_values)

# load data set from file
with open('../data/curvefitting.txt','r') as f:
    x_data = []
    t_data = []
    for row in f:
        x_point, t_point = row.split()
        x_data.append(float(x_point))
        t_data.append(float(t_point))

ws_star_reg = []
for lam in [0,pow(np.e,-18),1]:
    w_star_reg = polyfitRegLS(x_data,t_data,9,lam)
    ws_star_reg.append(w_star_reg)
    if lam != 0:
        y_lam = []
        for x in x_values:
            y_lam_n = 0
            for m in range(10):
                y_lam_n += w_star_reg[m]*pow(x,m)
            y_lam.append(y_lam_n)
        y_lam = np.array(y_lam).reshape(len(x_values),1)
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
        plt.annotate('$ln\ \lambda={}$'.format(int(np.log(lam))), xy=(0.7,0.9), xytext=(0.7,0.9))
        plt.scatter(x_data, t_data, s=33, facecolors='none', edgecolors='b')
        plt.plot(x_values, y_lam, c='r')
        plt.savefig('../p5pa_{}.pdf'.format(int(np.log(lam))))
        plt.clf()
    ws_star_reg_rd = ['{:.2f}'.format(elem) for elem in ws_star_reg]
    with open('../p5pa_table.csv','w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ws_star_reg_rd)

