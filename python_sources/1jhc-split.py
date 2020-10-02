#!/usr/bin/env python
# coding: utf-8

# # The story
# 
# In [this forum](https://www.kaggle.com/c/champs-scalar-coupling/discussion/104241#latest-605432) Scirpus suggested that 1JHC is not only one class and can be further divided. Lars Bratholm (organizer of the competition) said, that it is true, because carbon atoms can have different hybridization (i.e. they have different number of neighbours to which they are connected with bonds). Giba (Kaggle grandmaster) said that samples can be easily splitted by distance between hydrogen and carbon, setting the threshold to 1.065.
# 
# I haven't seen a kernel or a graph showing this, so I made one. I try to split samples by distance between hydrogen and carbon and then to split them by number of neighbours of carbon. I find that splits are the same, except one sample, which is just between two groups and according to different criteria belongs to different groups.

# In[ ]:


# Imports

# Basic imports
import numpy as np
import pandas as pd

# Graphs
# %matplotlib widget
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


# In[ ]:


# Seaborn advanced settings

sns.set(style='ticks',          # 'ticks', 'darkgrid'
        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'
        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'
        rc = {
           'figure.autolayout': True,
           'figure.figsize': (14, 8),
           'legend.frameon': True,
           'patch.linewidth': 2.0,
           'lines.markersize': 6,
           'lines.linewidth': 2.0,
           'font.size': 20,
           'legend.fontsize': 20,
           'axes.labelsize': 16,
           'axes.titlesize': 22,
           'axes.grid': True,
           'grid.color': '0.9',
           'grid.linestyle': '-',
           'grid.linewidth': 1.0,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'xtick.major.size': 8,
           'ytick.major.size': 8,
           'xtick.major.pad': 10.0,
           'ytick.major.pad': 10.0,
           }
       )

plt.rcParams['image.cmap'] = 'viridis'


# In[ ]:


df = pd.read_csv("../input/champsbasic1jhc/1JHC-Train.csv")
df = df[["molecule_name", "atom_index_0", "atom_index_1", "dist", "N_neighbours_Jatom", "scalar_coupling_constant"]]
test = pd.read_csv("../input/champsbasic1jhc/1JHC-Test.csv")
test = test[["id", "molecule_name", "atom_index_0", "atom_index_1", "dist", "N_neighbours_Jatom"]]


# # Histograms
# 
# Lets first plot histograms of scalar coupling constant, distance and number of neighbours of carbon atom.

# In[ ]:


plt.figure("Histogram - scalar coupling constant")
sns.distplot(df["scalar_coupling_constant"], bins=100, kde=False)
plt.xlabel("Scalar coupling constant")
plt.title("1JHC")
plt.savefig("HistogramSCC.png")
plt.show()


# In[ ]:


plt.figure("Histogram - Distance")
sns.distplot(df["dist"], bins=200, kde=False, rug=True)
plt.xlabel("Distance")
plt.title("1JHC")
plt.xlim(1.06, 1.145)
plt.savefig("HistogramDistances-Train.png")
plt.show()


# In[ ]:


plt.figure("Histogram - Neighbours")
sns.distplot(df["N_neighbours_Jatom"], bins=4, kde=False)
plt.xlabel("Number of neighbours")
plt.title("1JHC")
plt.show()


# All three graphs show a small group of very different samples:
# 
# - Scalar coupling constant > 180
# - Distance < 1.07
# - Number of neighbours == 2
# 
# Are they the same samples?

# # Divide by distance or number of neighbours?

# ## Divide by distance

# In[ ]:


distance_threshold = 1.065
plt.figure("Scatter by distance")
plt.plot(df[df["dist"]>distance_threshold]["dist"], df[df["dist"]>distance_threshold]["scalar_coupling_constant"], "go", label=f"Distance > {distance_threshold}")
plt.plot(df[df["dist"]<distance_threshold]["dist"], df[df["dist"]<distance_threshold]["scalar_coupling_constant"], "bo", label=f"Distance < {distance_threshold}")
plt.axvline(x=distance_threshold, linestyle="--", color="r")
plt.xlabel("Distance")
plt.ylabel("Scalar coupling constant")
plt.title("1JHC")
plt.xlim(1.06, 1.145)
plt.ylim(64, 208)
plt.legend()
plt.savefig("ScatterByDistance.png")
plt.show()


# The threshold 1.065 looks good, there is a gap big enough between the rightmost and leftmost samples from both groups.

# ## Divide by neighbours

# In[ ]:


plt.figure("Violin by neighbours")
sns.violinplot(x="N_neighbours_Jatom", y="scalar_coupling_constant", data=df)
plt.title("1JHC")
plt.xlabel("Number of neighbours")
plt.ylabel("Scalar coupling constant")
plt.savefig("ViolinsByNeighbours.png")
plt.show()


# When samples are divided by number of neighbours of a carbon atom, we see very different distributions. Note however, that the case with 5 neighbours is very rare and there are only a few samples.

# ## Comparison

# Lets check whether splits are the same:

# In[ ]:


first_condition = df[df["dist"]<distance_threshold].shape == df[df["N_neighbours_Jatom"]==2].shape
if first_condition:
    second_condition = np.allclose(df[df["dist"]<distance_threshold]["scalar_coupling_constant"], df[df["N_neighbours_Jatom"]==2]["scalar_coupling_constant"])
print(f"Comparisons by distance and by number of neighbours are {'' if first_condition else 'not '}same.")
if not first_condition:
    print(f"First comparison by distance contains {len(df[df['dist']<distance_threshold])} atoms, comparison by neighbours contains {len(df[df['N_neighbours_Jatom']==2])} atoms.")
else:
    print(f"All pairs are {'' if second_condition else 'not '}identical")


# In[ ]:


plt.figure("Scatter with differences")
plt.plot(df[(df["N_neighbours_Jatom"]>2) & (df["dist"]>distance_threshold)]["dist"], df[(df["N_neighbours_Jatom"]>2) & (df["dist"]>distance_threshold)]["scalar_coupling_constant"],
         "bo", label=f"3+ neighbours, distance > {distance_threshold}")
plt.plot(df[(df["N_neighbours_Jatom"]==2) & (df["dist"]<distance_threshold)]["dist"], df[(df["N_neighbours_Jatom"]==2) & (df["dist"]<distance_threshold)]["scalar_coupling_constant"],
         "go", label=f"2 neighbours, distance < {distance_threshold}")
plt.plot(df[(df["N_neighbours_Jatom"]==2) & (df["dist"]>distance_threshold)]["dist"], df[(df["N_neighbours_Jatom"]==2) & (df["dist"]>distance_threshold)]["scalar_coupling_constant"],
         "ro", markersize=12, alpha=0.8, label="Difference in comparisons")
plt.axvline(x=distance_threshold, linestyle="--", color="red")
plt.xlabel("Distance")
plt.ylabel("Scalar coupling constant")
plt.title("1JHC")
plt.xlim(1.06, 1.145)
plt.ylim(64, 208)
plt.legend()
plt.savefig("ScatterWithDifferences.png")
plt.show()


# Splits are the same except one sample, but its scalar coupling constant is closer to the group with higher distance, so it is probably indeed better to divide by distance. In test samples there is no such sample with distance right in between two groups (see histogram below), so it should be safe. The question is if this helps you somehow, because from total of cca. 709K samples the smaller group contains only cca. 9K samples and it can be too few for your training, however, this is up to you.
# 
# I save groups divided by distance as output.

# In[ ]:


plt.figure("Histogram - Distance")
sns.distplot(test["dist"], bins=200, kde=False, rug=True)
plt.xlabel("Distance")
plt.title("1JHC")
plt.xlim(1.06, 1.145)
plt.savefig("HistogramDistances-Test.png")
plt.show()


# In[ ]:


df[df["dist"]>distance_threshold].to_csv("1JHC-HighDistance-Train.csv", index=False)
df[df["dist"]<distance_threshold].to_csv("1JHC-LowDistance-Train.csv", index=False)
test[test["dist"]>distance_threshold].to_csv("1JHC-HighDistance-Test.csv", index=False)
test[test["dist"]<distance_threshold].to_csv("1JHC-LowDistance-Test.csv", index=False)


# # Bonus - Number of neighbours
# 
# As a bonus I made a graph of scalar coupling constant dependent on distance, but samples are divided by number of neighbours of carbon atom.

# In[ ]:


plt.figure("Scatter by neighbours")
plt.plot(df[df["N_neighbours_Jatom"]==2]["dist"], df[df["N_neighbours_Jatom"]==2]["scalar_coupling_constant"], "o", label=f"2 nbs", alpha=0.1)
plt.plot(df[df["N_neighbours_Jatom"]==3]["dist"], df[df["N_neighbours_Jatom"]==3]["scalar_coupling_constant"], "o", label=f"3 nbs", alpha=0.025)
plt.plot(df[df["N_neighbours_Jatom"]==4]["dist"], df[df["N_neighbours_Jatom"]==4]["scalar_coupling_constant"], "o", label=f"4 nbs", alpha=0.025)
plt.plot(df[df["N_neighbours_Jatom"]==5]["dist"], df[df["N_neighbours_Jatom"]==5]["scalar_coupling_constant"], "o", label=f"5 nbs", alpha=0.7)
legend_elements = [Line2D([0], [0], marker='o', markersize=12, color=sns.color_palette()[0], linestyle='none', label='2 nbs'),
                   Line2D([0], [0], marker='o', markersize=12, color=sns.color_palette()[1], linestyle='none', label='3 nbs'),
                   Line2D([0], [0], marker='o', markersize=12, color=sns.color_palette()[2], linestyle='none', label='4 nbs'),
                   Line2D([0], [0], marker='o', markersize=12, color=sns.color_palette()[3], linestyle='none', label='5 nbs')]
plt.xlabel("Distance")
plt.ylabel("Scalar coupling constant")
plt.title("1JHC")
plt.xlim(1.06, 1.13)
plt.ylim(64, 208)
plt.legend(handles=legend_elements)
plt.savefig("ScatterByNeighbours.png")
plt.show()


# P. S. There is one sample with very high distance between hydrogen and carbon (over 1.25), but I excluded it from graphs, so they are more readable.
