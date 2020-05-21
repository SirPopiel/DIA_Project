from bidding_environment import *
import numpy as np
import matplotlib.pyplot as plt

xpts = np.linspace(0, 1, 100)

colors = {1: 'b', 2: 'r', 3: 'g'}


for c in [1,2,3]:
    plt.plot(xpts, [n_to_f[c](x) for x in xpts], color = colors[c])

plt.show()
