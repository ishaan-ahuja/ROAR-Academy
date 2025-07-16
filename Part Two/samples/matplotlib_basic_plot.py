## This is course material for Introduction to Python Scientific Programming
## Example code: matplotlib_basic_plot.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import matplotlib.pyplot as plt
import numpy as np

# generate a basic sample point array on x-axis
x = np.arange(0,2*np.pi,0.1)

# Create a sin function sample
y0 = np.sin(x)
plt.plot(x, y0, color = 'r', linewidth = 3)

# Create a dash cos function sample
y1 = np.cos(x)
plt.plot(x, y1, 'b--', linewidth = 1)
plt.ylim(-1, 1)
plt.xlim(0,2*np.pi)
plt.xticks(np.arange(0,2*np.pi,np.pi/4), ['0', 'pi/4', 'pi/2', '3pi/4', 'pi', '5pi/4', '3pi/2', '7pi/4'])
plt.show()

