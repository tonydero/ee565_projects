import numpy as np
from requiredFunctions.polyfitLS import polyfitLS, polyfitRegLS
from requiredFunctions.noisySin import noisySin
import matplotlib.pyplot as plt
from matplotlib import rc
from distutils.spawn import find_executable
import csv


"""
b) Generate a plot showing an example of the distribution N=50 samples, σ²=0.05
where data points are plotted as "blue o" and the clean noise free sinusoid is
plotted in green.
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
test_data = noisySin(10, noise_var=0.3)

M_vals = range(10)
ws_star = []
ERMS_vals = []
for M in M_vals:
    w_star, E = polyfitLS(x_data,t_data,M)
    w_star_t, E_t = polyfitLS(test_data[0],test_data[1],M)
    ws_star.append(w_star)
    ERMS = np.sqrt(E/len(x_data))
    ERMS_vals.append(ERMS)
    ERMS_t = np.sqrt(E_t/len(test_data[0]))
    ERMS_t_vals.append(ERMS_t)
    ws_star_rd = ['{:.2f}'.format(elem) for elem in ws_star]
    with open('../p4pa_table.csv','w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ws_star_rd)

for w_idx in [0,1,3,9]:
    y_m = []
    for x in x_values:
        y_m_n = 0
        for m in range(M_vals[w_idx]+1):
            y_m_n += ws_star[w_idx][m]*pow(x,m)
        y_m.append(y_m_n)
    y_m = np.array(y_m).reshape(len(x_values),1)
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
    plt.annotate('$M={}$'.format(M_vals[w_idx]), xy=(0.9,0.9), xytext=(0.9,0.9))
    plt.scatter(x_data, t_data, s=33, facecolors='none', edgecolors='b')
    plt.plot(x_values, y_m, c='r')
    plt.savefig('../p4pa_{}.pdf'.format(M_vals[w_idx]))
    plt.clf()

"""
b) Recreate Figure 1.5
"""
plt.xlim(-1, 10)
plt.ylim(0, 1)
plt.xticks(np.arange(0,10,3))
plt.yticks(np.arange(0,1.5,0.5))
plt.tick_params(direction='in', top=1, right=1)
plt.xlabel('$M$')
plt.ylabel('$E_{RMS}$')
plt.scatter(np.array(M_vals),np.array(ERMS_vals)[M],c='b',marker='o')
plt.plot(np.array(M_vals),np.array(ERMS_vals)[:,0,0],c='b')
plt.scatter(np.array(M_vals),np.array(ERMS_t_vals)[M],c='r',marker='o')
plt.plot(np.array(M_vals),np.array(ERMS_t_vals)[:,0,0],c='r')
plt.savefig('../p4pb.pdf')
plt.clf()

"""
c) Recreate Figure 1.6
"""
for N in [15, 100]:
    # generate data from noisySin
    data = noisySin(N, noise_var=0.3)
    x_data = data[:,0]
    t_data = data[:,1]

    M = 9
    w_star, E = polyfitLS(x_data,t_data,M)
    y_m = []
    for x in x_values:
        y_m_n = 0
        for m in range(10):
            y_m_n += w_star[m]*pow(x,m)
        y_m.append(y_m_n)
    y_m = np.array(y_m).reshape(len(x_values),1)
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
    plt.annotate('$N={}$'.format(N), xy=(0.9,0.9), xytext=(0.9,0.9))
    plt.scatter(x_data, t_data, s=33, facecolors='none', edgecolors='b')
    plt.plot(x_values, y_m, c='r')
    plt.savefig('../p4pc_{}.pdf'.format(N))
    plt.clf()

