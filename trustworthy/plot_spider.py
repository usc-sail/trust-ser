import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


from math import pi

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    # for d, (y1, y2) in zip(data[1:], ranges[1:]):
    for d, (y1, y2) in zip(data, ranges):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)

    x1, x2 = ranges[0]
    d = data[0]

    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1

    sdata = [d]

    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1

        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)

    return sdata

def set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kwargs):
    """
    Set the radial locations and labels of the *r* grids.
    The labels will appear at radial distances *radii* at the
    given *angle* in degrees.
    *labels*, if not None, is a ``len(radii)`` list of strings of the
    labels to use at each radius.
    If *labels* is None, the built-in formatter will be used.
    Return value is a list of tuples (*line*, *label*), where
    *line* is :class:`~matplotlib.lines.Line2D` instances and the
    *label* is :class:`~matplotlib.text.Text` instances.
    kwargs are optional text properties for the labels:
    %(Text)s
    ACCEPTS: sequence of floats
    """
    # Make sure we take into account unitized data
    radii = self.convert_xunits(radii)
    radii = np.asarray(radii)
    rmin = radii.min()
    # if rmin <= 0:
    #     raise ValueError('radial grids must be strictly positive')

    self.set_yticks(radii)
    if labels is not None:
        self.set_yticklabels(labels)
    elif fmt is not None:
        self.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    if angle is None:
        angle = self.get_rlabel_position()
    self.set_rlabel_position(angle)
    for t in self.yaxis.get_ticklabels():
        t.update(kwargs)
    return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()

class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ordinate_levels=5, color=''):
        angles = [90, 162, 234, 306, 18]

        # N = len(variables)
        # angles = [n / float(N) * 2 * pi for n in range(N)]
        axes = [fig.add_axes([0.2, -0.02, 0.6, 0.95], polar=True, label = "axes{}".format(i)) for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=[f'{variables[0]}(%)', f'{variables[1]}\n(%)', f'{variables[2]}\n(1e9)', f'{variables[3]}\n(%)', f'{variables[4]}\n(%)'])
        [txt.set_rotation(angle-90) for txt, angle in zip(text, angles)]

        label_positions = [(0, -0.05), (0, -0.2), (-100, -0.23), (0, -0.2), (0, -0.25)]
        [text[i].set_position(label_positions[i]) for i in range(len(text))]
        [text[i].set_fontsize(17.5) for i in range(len(text))]
        [text[i].set_fontweight("bold") for i in range(len(text))]
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")
            
        labels = [
            ["", "59", "62", "65", "68"],
            ["", "21", "18", "15", "12"],
            ["", "36", "24", "12", "0"],
            ["", "88", "76", "64", "52"],
            ["", "97", "94", "91", "88",]
        ]

        for i, ax in enumerate(axes):
            
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid, gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            # ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            set_rgrids(ax, grid, labels=gridlabel, angle=angles[i])
            # ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
            ax.set_yticklabels(labels[i])

            for label in (ax.get_yticklabels()):
                label.set_fontsize(13)
                label.set_fontweight('bold')
                
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.color = color
        
    def plot(self, data, color, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], color=color, *args, **kw)
    def fill(self, data, color, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], color=color, *args, **kw)

# Set data
df = pd.read_csv("summary.csv", index_col=0)

categories = list(df.columns)
model_names = list(df.index)

for row in range(len(model_names[:-1])):
    values = df.loc[df.index[row]].values.flatten().tolist()
    ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

    max_list = [68, 24, 48, 100, 100]
    min_list = [56, 12, 0, 40, 88]

    for cat_idx in range(len(categories)):
        max_num, min_num = max_list[cat_idx], min_list[cat_idx]
        if cat_idx == 0: values[cat_idx] = 100*(values[cat_idx] - min_num) / (max_num - min_num)
        else: values[cat_idx] = 100-100*(values[cat_idx] - min_num) / (max_num - min_num)

    # Plotting
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#008080']
    my_palette = ListedColormap(colors)

    fig1 = plt.figure(figsize=(6, 5.25))
    # fig.subplots_adjust(wspace=-0.5, hspace=-0.5)
    radar = ComplexRadar(fig1, categories, ranges, color=colors[row])
    radar.plot(values, color=colors[row])
    radar.fill(values, color=colors[row], alpha=0.2)
    
    plt.title(df.index[row], size=24, color=colors[row], y=1.15, weight="bold")
    plt.savefig(f'trust_profile/{model_names[row].replace(" ", "_")}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


df = pd.read_csv("summary.csv", index_col=0)
df = df.loc[["Whisper Tiny", "Whisper Tiny-FT"], :]

categories = list(df.columns)
model_names = list(df.index)
ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

fig1 = plt.figure(figsize=(6, 5.25), linewidth=8, edgecolor='black')
radar = ComplexRadar(fig1, categories, ranges, color=colors[row])
    
for row in range(len(model_names)):
    values = df.loc[df.index[row]].values.flatten().tolist()
    ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

    max_list = [68, 24, 48, 100, 100]
    min_list = [56, 12, 0, 40, 88]

    for cat_idx in range(len(categories)):
        max_num, min_num = max_list[cat_idx], min_list[cat_idx]
        if cat_idx == 0: values[cat_idx] = 100*(values[cat_idx] - min_num) / (max_num - min_num)
        else: values[cat_idx] = 100-100*(values[cat_idx] - min_num) / (max_num - min_num)

    # Plotting
    colors = ['#2ca02c', '#e74c3c']
    my_palette = ListedColormap(colors)

    # fig.subplots_adjust(wspace=-0.5, hspace=-0.5)
    radar.plot(values, color=colors[row])
    radar.fill(values, color=colors[row], alpha=0.2)

plt.title("Whisper Tiny\nFrozen vs. FT Last Layer", size=20, color='black', y=1.15, weight="bold")
legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0], label='Frozen', alpha=0.4),
                   Patch(facecolor=colors[1], edgecolor=colors[1], label='FT Last Layer', alpha=0.4)]

plt.legend(handles=legend_elements, prop={'weight':'bold', 'size': 10.75}, bbox_to_anchor=(1.3, 1.05))
plt.savefig(f'trust_profile/Compare.png', dpi=300, bbox_inches='tight', pad_inches=0.075)
plt.close()