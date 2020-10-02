#!/usr/bin/env python
# coding: utf-8

# # Quaterni... what?
# 
# ![](http://)A small snippet to show how one can visualize the 3d vector rotations in `matplotlib`.  Could this kind of visualizations give some insights?
# 
# > Note: This plot looks much better in Jupyter notebook. It will be interactive and you'll be able to rorate the axis and see the projections from the various sides.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from pyquaternion import Quaternion


# In[ ]:


x_trn = pd.read_csv('../input/X_train.csv')


# In[ ]:


ser0 = x_trn[x_trn.series_id == 0]


# In[ ]:


orient = ser0.columns.str.startswith('orientation')


# In[ ]:


qs = [Quaternion(list(row)) for _, row in ser0[ser0.columns[orient]].iterrows()]


# In[ ]:


vec = [1, 0, 1]
xyz = [q.rotate(vec) for q in qs]
xs, ys, zs = [list(seq) for seq in zip(*xyz)]
ax = plt.axes(projection='3d')
ax.scatter3D(xs, ys, zs, c=zs)


# In[ ]:




