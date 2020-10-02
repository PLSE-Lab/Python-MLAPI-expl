#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.axes as ax
from math import sqrt


# In[ ]:


df = pd.read_csv('../input/World_vars.csv', index_col = 0)


# In[ ]:


df = df.fillna(value = 0)
df.columns


# In[ ]:


df.head()


# In[ ]:


plt.scatter(df['cropland_cover'], 
            df['tree_canopy_cover'], 
            c = df['Continent_color'],
            alpha = .5)
plt.xlabel('Cropland Cover (%)', fontsize = 12)
plt.ylabel('Tree Canopy Cover (%)', fontsize = 12)
plt.title('Crop and Tree Canopy Cover')
purple_patch = mpatches.Patch(color = 'purple', label = 'Africa', alpha = .5)
grey_patch = mpatches.Patch(color = 'grey', label = 'Antarctica', alpha = .5)
red_patch = mpatches.Patch(color = 'red', label = 'Asia', alpha = .5)
orange_patch = mpatches.Patch(color = 'orange', label = 'Australia and Oceania', alpha = .5)
blue_patch = mpatches.Patch(color = 'blue', label = 'Europe', alpha = .5)
green_patch = mpatches.Patch(color = 'green', label = 'Latin America', alpha = .5)
yellow_patch = mpatches.Patch(color = 'yellow', label = 'North America', alpha = .5)
plt.legend(handles=[purple_patch, grey_patch, red_patch, orange_patch,
                   blue_patch, green_patch, yellow_patch],
           loc = 'best',
           ncol = 2,
           borderaxespad = .1,
           columnspacing = 0)


# In[ ]:


plt.scatter(df['cropland_cover'], 
            df['tree_canopy_cover'], 
            s = df['Agr_land'] / 1000,
            c = df['Continent_color'],
            alpha = .25)
plt.xlabel('Cropland Cover (%)', fontsize = 12)
plt.ylabel('Tree Canopy Cover (%)', fontsize = 12)
plt.title('Country Land Cover', y = 1.05, fontsize = 14)

max_land = df['Agr_land'].max() / 1000
# Convert between plot markersize and scatter s parameters
max_land_ms = sqrt(max_land)
h = [plt.plot([], color = 'gray', marker = 'o', ms = i , ls = '', alpha = .5)[0]
    for i in np.linspace(0, max_land_ms, num = 10)[1:]]
leg = plt.legend(handles = h, 
           labels = [int(i) * 1000 for i in np.linspace(0, max_land, num = 10)[1:]],
           loc = (1.1, 0),
           labelspacing = 1.5,
           title = 'Agr Land (sq km)',
                borderpad = 3.5)
ax = plt.gca().add_artist(leg)
plt.legend(handles=[purple_patch, grey_patch, red_patch, orange_patch,
                   blue_patch, green_patch, yellow_patch],
           loc = 1,
           ncol = 1)
plt.savefig('Land_cover.png', bbox_inches = 'tight')
plt.show()

