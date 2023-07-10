import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import os

plt.rcParams['text.usetex'] = True

name = ["lr = (0.1, 0.01, 0.001)", "(784-256-256-10)"]
data = np.array([[0.5, 1.0, 2.0], [1.0, 10.0, 20.0]])

fig = plt.figure(figsize=(20,8))

x = np.linspace(1, len(data), num=len(data))

plt.bar(x, data[:, 0], width=0.15, label = "w/ branch removal", color = "skyblue")
plt.bar(x+0.2, data[:, 1], width=0.15, label = "w/o branch removal", color = "#0087BD")
plt.bar(x+0.4, data[:, 2], width=0.15, label = "w/o branch removal", color = "#00FFFF")

plt.xticks(x+0.15, name, fontsize=28)
plt.xlabel('X', fontsize=44)

plt.yticks(fontsize=28)
plt.ylabel('Y', fontsize=44, fontweight='bold')
#  plt.legend(loc = 'upper left', fontsize = 32)

#  fig.tight_layout()
plt.show()
#  plt.savefig("test.png")

