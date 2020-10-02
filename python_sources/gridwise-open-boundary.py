#!/usr/bin/env python
# coding: utf-8

# # Open Boundary Condition
# The data are generated from a model with open boundary condition as inferred by Hamed.
# 
# 3D plot grouping the data into rounded off x and y 

# In[ ]:





# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16,5)


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='row_id')


# In[ ]:


train['grid_x'] = train.x.apply(lambda x: round(x))
train['grid_y'] = train.y.apply(lambda y: round(y))
grouped_grid = train.groupby(['grid_x', 'grid_y'])
grid_freq = grouped_grid.size().reset_index()


# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

xpos = grid_freq.grid_x
ypos = grid_freq.grid_y
zpos = np.zeros(len(grouped_grid))

dx = np.ones(len(grouped_grid))
dy = np.ones(len(grouped_grid))
dz = grid_freq[0]



ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

plt.show()


# In[ ]:




