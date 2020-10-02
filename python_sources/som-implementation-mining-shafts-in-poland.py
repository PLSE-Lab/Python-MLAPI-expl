#!/usr/bin/env python
# coding: utf-8

# This kernel shows my implementation of the Self-Organizing Map on the Mining shafts in Poland dataset. The problem I was trying to solve was to find out classes of the shafts, based on their arrangement in the terrain. I was asked to look for similarities of the shaft classes, check if there are any differences between them etc. This information could be used as a guideline for historical origin of the shafts.

# Firstly, import part. I used PyPI MiniSom package for SOM.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
get_ipython().system('pip install MiniSom')
get_ipython().system('pip install --upgrade pip')
from minisom import MiniSom


# Data consists of six columns.
# 
# First three columns are connected with the measurement parameters (in sequence: section = search field ID, ID = measurement ID, section_area = search field area in hectares).
# Last three columns are connected with the shafts parameters (in sequence: object_area = area of the form (embankment + shaft) in square meters, rel_height = relative height of the embankment around the shaft in meters, object_dostance = distance between shafs (holes) in meters.
# 
# More information is in the data description in Public Dataset.

# In[ ]:


dataset = pd.read_csv("../input/mines-data/mines_data.csv", encoding='latin1')
dataset


# To create SOM, I used all data, referring to the measurements and, additionally, research field area, as it carries information about shafts density and arrangement.

# In[ ]:


data_som = dataset[['section_area', 'object_area', 'rel_height', 'object_distance']]
data_som


# I standardarized the data, to unify units (hectares and meters).

# In[ ]:


sc = StandardScaler()
data_som_sc = sc.fit_transform(data_som)


# Below, I subtracted section ID and measurement ID for further plotting purposes.

# In[ ]:


ids = dataset[['ID']]
I = ids['ID'].tolist()
section = dataset[['section']]
S = section['section'].tolist()


# Finally, I trained simple SOM on not much tuned parameters. Training is just a second.

# In[ ]:


som = MiniSom(x = 21, y = 21, input_len = 4, neighborhood_function = 'gaussian', sigma = 1.5, learning_rate = 0.5)
som.random_weights_init(data_som_sc)
som.train_random(data = data_som_sc, num_iteration = 1000)


# First plot shows distance map between neurons. Markers correspond to the section IDs. SOM clustered shafts rather according to their origin in the terrain, including some exeptions of course. Blank spaces between markers can be interpret as borders between clusters, where bright color means very strong border and darker means border, which is not so defined.
# 
# So, red point could be interpret as the outlier in the data, group of dark green squares and blue diamonds are also strongly different from the other data. Other clusters are mixed, so there are similarities between them. I will check this hypothesis later.

# In[ ]:


from pylab import bone, pcolor, colorbar, plot, show
bone()
plt.figure(figsize=(14, 10))
pcolor(som.distance_map().T, alpha=.9)
colorbar()
markers = ['*', 'o', 's', 'D', 'X', 'v', 'P', 'h', '^']
colors = ['grey', 'r', 'g', 'b', 'y', 'c', 'orange', 'fuchsia', 'lime']
for i, x in enumerate(data_som_sc):
    w = som.winner(x)
    plot(w[0] + 0.5,
        w[1] + 0.5,
        markers[S[i]],
        markeredgecolor=colors[S[i]],
        markerfacecolor='None',
        markersize = 10,
        markeredgewidth=2
        )
show()


# Second plot, with markers corresponding to section ID. I reversed colors of the distance map for beter readability of the plot. Legend shows markers, which are assigned to the section IDs.

# In[ ]:


from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='r', label='1',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='s', color='g', label='2',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='D', color='b', label='3',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='X', color='y', label='4',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='v', color='c', label='5',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='P', color='orange', label='6',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='h', color='fuchsia', label='7',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='^', color='lime', label='8',
                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2)]

from pylab import bone, pcolor, colorbar, plot, show
bone()
plt.figure(figsize=(14, 10))
pcolor(som.distance_map().T, cmap='gray_r', alpha=.7)
colorbar()
markers = ['*', 'o', 's', 'D', 'X', 'v', 'P', 'h', '^']
colors = ['grey', 'r', 'g', 'b', 'y', 'c', 'orange', 'fuchsia', 'lime']
for i, x in enumerate(data_som_sc):
    w = som.winner(x)
    plot(w[0] + 0.5,
        w[1] + 0.5,
        markers[S[i]],
        markeredgecolor=colors[S[i]],
        markerfacecolor='None',
        markersize = 10,
        markeredgewidth=2
        )
plt.legend(handles=legend_elements, bbox_to_anchor=(1.27, 1.01), prop={'size': 12}, ncol=1)
show()


# Third plot, the same, as previous, with addition of the following measurement IDs.

# In[ ]:


from pylab import bone, pcolor, colorbar, plot, show
bone()
plt.figure(figsize=(14, 10))
pcolor(som.distance_map().T, cmap='gray_r', alpha=.7)
colorbar()
markers = ['*', 'o', 's', 'D', 'X', 'v', 'P', 'h', '^']
colors = ['grey', 'r', 'g', 'b', 'y', 'c', 'orange', 'fuchsia', 'lime']
for i, x in enumerate(data_som_sc):
    w = som.winner(x)
    plot(w[0] + 0.5,
        w[1] + 0.5,
        markers[S[i]],
        markeredgecolor=colors[S[i]],
        markerfacecolor='None',
        markersize = 10,
        markeredgewidth=2
        )
wmap = {}
im = 0
for x, t in zip(data_som_sc, I):
    w = som.winner(x)
    wmap[w] = im
    plt. text(w[0]+.8,  w[1]+.3,  str(t),
              color='k', fontdict={'weight': 'normal', 'size': 14})
    im = im + 1
plt.legend(handles=legend_elements, bbox_to_anchor=(1.27, 1.01), prop={'size': 12}, ncol=1)
show()


# Let's test out hypothesis.
# 
# 1. Outlier red circle, which is measurement ID 0 from research field 1, has obviously different parameters from the other measurements.

# In[ ]:


ro1 = dataset[dataset['ID'] == 0]
ro1


# In[ ]:


x = dataset.object_area
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.distplot(x, label="item_id", kde=True, bins=20)


# In[ ]:


x = dataset.rel_height
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.distplot(x, label="item_id", kde=True, bins=20)


# In[ ]:


x = dataset.object_distance
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.distplot(x, label="item_id", kde=True, bins=20)


# 2. Dark green squares, which are measurements from research field 2, are the second group of the very high objects, right behind outlier object. All the parameters in this cluster have small variance, so only height determines their different character.

# In[ ]:


dgs2 = dataset[dataset['section'] == 2]
dgs2


# In[ ]:


x = dataset.rel_height
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.distplot(x, label="x", kde=False, bins=20)


# 3. Blue diamonds is the cluster, which is also readable, but it is more scattered, than previous ones. The objects inside the cluster have similar area and they are also rather high. That, what differs them, is variance in the object arrangement in the terrain. This group is arranged more irregular.

# In[ ]:


bd3 = dataset[dataset['section'] == 3]
bd3


# The data is dual - there are objects very close to each other and objects far away from each other.

# In[ ]:


x = irregular.object_distance
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.distplot(x, label="x", kde=False, bins=5)


# And that is the reason for the scattered character of the cluster. The right part of the cluster contains objects with the long distance between them, with the longest values in the upper right part.

# In[ ]:


bd3_upper_right = bd3[bd3['object_distance'] > 30]
bd3_upper_right


# In[ ]:


bd3_right = bd3[bd3['object_distance'] > 20]
bd3_right


# Lowest part of the cluster contains objets with middle distance between them.

# In[ ]:


bd3_lower = bd3[bd3['object_distance'] > 15]
bd3_lower


# And the left part of the cluster contains objects with short distance between them.

# In[ ]:


bd3_left = bd3[bd3['object_distance'] < 10]
bd3_left


# 4. Yellow X marked as research field 4 on the distance maps, are divided into 3 clusters.

# In[ ]:


yx4 = dataset[dataset['section'] == 4]
yx4


# Outlier part within section 4 is the cluster with objects located far away from each other. It is the right part of the section 4 on the distance map.

# In[ ]:


yx4_right = yx4[yx4['object_distance'] > 10]
yx4_right


# Cluster, which is lower part of the section 4, contains objects with medium distance between them.

# In[ ]:


yx4_lower = yx4_upper[yx4_upper['object_distance'] > 5]
yx4_lower


# Cluster, which is upper part of the section 4, consists of objects, distributed very closely in the terrain.

# In[ ]:


yx4_upper = yx4[yx4['object_distance'] < 10]
yx4_upper


# 5. Cyan triangle and orange cross sections are mixed and assigned by our SOM into one cluster. Those objects have similar parameters, with smaller objects located in the left part of the cluster and bigger in the right part of the cluster.

# In[ ]:


ct5 = dataset[dataset['section'] == 5]
ct5


# In[ ]:


oc6 = dataset[dataset['section'] == 6]
oc6


# 6. Fuchsia hexagon cluster, marked as research field 7, is one cluster with only one object, assigned by a SOM as a strong outlier. Measurement of the ID 120 is far away from other shafts and that was the determinant of the outlying.

# In[ ]:


fh7 = dataset[dataset['section'] == 7]
fh7


# 7. Lastly, bright green triangles, marked as research field 8, are assigned as one cluster with some outliers, near outlier from fuchsia hexagon point. And, indeed, those measurements with IDs 138 and 139, have long object distance, which makes them look similar to the measurement with ID 120 and in the same time, they are outliers within the group, where object distance is very low. Objects with small distance between each other, but larger than the rest of the cluster (except for the outliers), are located in the upper part of the bright green triangle cluster.

# In[ ]:


bgt8 = dataset[dataset['section'] == 8]
bgt8


# To summarize: Self-Organizing Map clusters with strong borders are the most different chunks of the data. That means section 0, 2, 3 differ from each other and from the rest of the dataset the most. Section 0 is highest object with only one shaft, section 2 contains high, regular spread shafts and section 3 contains high objects, randomly distributed across the terrain.
# 
# Other clusters are more similar to each other. Section 4 is divided into three parts, as there are shaft with long, medium and short distance between them. But, whole section 4 is similar to the section 3 in its character.
# 
# Section 5 looks almost identical to the section 6 for our SOM. Outliers from section 7 and 8 also share one cluster in the distance map.
# 
# The conclusion is - last measurements with small objects near to each other are very similar, where first measurements differ strongly in their height, area and arrangement in the terrain.

# That was all! I hope you enjoyed this simple approach to the task, I would be happy for any suggestions and thoughts.
