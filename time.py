import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import os


name = ["lr = (1, 0.1, 0.01)", "layers = (3, 4, 5)", "batch = (64, 10^5, 6x10^5)", "epochs = (5, 14, 20)", "nodes = (100, 256, 500)"]
data = np.array([[2171.90, 1156.88, 1512.76], [2441.10, 3579.93, 4238.53], [5116.83, 628.30, 446.72], [827.09, 2653.10, 3324.03], [2041.12, 2415.95, 2359.69]])


fig = plt.figure(figsize=(35,10))

x = np.linspace(1, len(data), num=len(data))

plt.bar(x, data[:, 0], width=0.15, label = "w/ branch removal", color = "darkgreen")
plt.bar(x+0.2, data[:, 1], width=0.15, label = "w/o branch removal", color = "darkblue")
plt.bar(x+0.4, data[:, 2], width=0.15, label = "w/o branch removal", color = "red")

plt.xticks(x+0.15, name, fontsize=28)
plt.xlabel('Parameters', fontsize=44, fontweight='bold')

plt.yscale("log")
# plt.yticks(fontsize=28)
plt.ylabel('Time(seconds)', fontsize=44, fontweight='bold')
#  plt.legend(loc = 'upper left', fontsize = 32)

#  fig.tight_layout()
# plt.show()
plt.savefig("test2.png", bbox_inches='tight')

