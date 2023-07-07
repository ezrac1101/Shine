import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import os


plt.rcParams['text.usetex'] = True

name = ["(0)", "(1)"]
data = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

fig = plt.figure(figsize=(20,8))

x = np.linspace(1, len(data), num=len(data))

plt.bar(x, data[:, 1], width=0.3, label = "w/ branch removal", color = "skyblue")
plt.bar(x+0.35, data[:, 2], width=0.3, label = "w/o branch removal", color = "#0087BD")

plt.xticks(x+0.15, name, fontsize=28)
plt.xlabel('X', fontsize=44)

plt.yticks(fontsize=28)
plt.ylabel('Y', fontsize=44, fontweight='bold', color = "#0087BD")
plt.legend(loc = 'upper left', fontsize = 32)

fig.tight_layout()
#  plt.savefig("../Figure_{}_transformation.png".format(args.env),bbox_inches='tight')
plt.show()

