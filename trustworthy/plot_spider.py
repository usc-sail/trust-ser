# Libraries
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi
from matplotlib.colors import ListedColormap

def make_spider(row, title, color):

    # number of variable
    categories = list(df.columns)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(2, 3, row+1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    # plt.yticks([25, 50, 75], ["25","50","75"], color="grey", size=7)
    plt.yticks([0, 25, 50, 75, 100], ["", "", "", "", ""], color="grey", size=7)
    plt.ylim(0, 100)

    # Ind1
    values = df.loc[df.index[row]].values.flatten().tolist()
    for cat_idx in range(len(categories)):
        max_num = df[categories[cat_idx]].max()
        min_num = df[categories[cat_idx]].min()
        if cat_idx == 0: values[cat_idx] = 100*(values[cat_idx] - 50) / (max_num - 50)
        else: values[cat_idx] = 100*(100 - values[cat_idx]) / (100 - min_num)
    
    values += values[:1]
    values = np.array(values)
    ax.plot(angles, values, linewidth=1.25, linestyle='solid', color=my_palette(row))
    ax.fill(angles, values, alpha=0.2, color=my_palette(row))

    # Add a title
    plt.title(title, size=11, color=color, y=1.1)

# Set data
df = pd.read_csv("summary.csv", index_col=0)

# Create a color palette:
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)

my_palette = plt.cm.get_cmap("tab10", len(df.index))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
my_palette = ListedColormap(colors)

for row in range(0, len(df.index)):
    make_spider(row=row, title=df.index[row], color=my_palette(row))
plt.tight_layout()
plt.savefig('trustworthy.png', dpi=300)
pdb.set_trace()
