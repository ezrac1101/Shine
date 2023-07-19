import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import os


name = ["steps = (10^3, 10^4, 10^5, 10^6, 10^7)", "architecture = ([3, 256, 5], [3, 256, 256, 5], [3, 256, 256, 256, 5], [3, 256, 256, 256, 256, 5])"]
data = np.array([[-43.84, -293.09, -44.88, 836.91, 91.90], [-171.16, 836.91, 786.18, -270.94]])


fig = plt.figure(figsize=(35,10))

x = np.linspace(1, len(data), num=len(data))

plt.bar(x, data[:, 0], width=0.15, label = "w/ branch removal", color = "crimson")
plt.bar(x+0.2, data[:, 1], width=0.15, label = "w/o branch removal", color = "purple")
plt.bar(x+0.4, data[:, 2], width=0.15, label = "w/o branch removal", color = "dodgerblue")

plt.xticks(x+0.15, name, fontsize=28)
plt.xlabel('Parameters', fontsize=44, fontweight='bold')

plt.yscale("log")
# plt.yticks(fontsize=28)
plt.ylabel('Average Score over 10^2 Runs', fontsize=44, fontweight='bold')
#  plt.legend(loc = 'upper left', fontsize = 32)

#  fig.tight_layout()
# plt.show()
plt.savefig("test.png", bbox_inches='tight')

