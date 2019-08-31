# Created by Tony DeRocchis
# on 2019.08.24 at 23:33 UTC
import numpy as np
from bokeh.plotting import figure, output_file, show
from requiredFunctions.circGauss import circGauss
import matplotlib.pyplot as plt


"""
Main script for the solution to Project 0 for EE 565 Fall 2019
"""
samples_0 = circGauss(434, (0, 0), 1)
samples_5 = circGauss(434, (5, 2), 1)

plt.scatter(samples_0[0], samples_0[1])
plt.scatter(samples_5[0], samples_5[1])
plt.show()

