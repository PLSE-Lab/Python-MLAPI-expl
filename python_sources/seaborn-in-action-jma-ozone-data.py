#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd


# In[ ]:


dat = pd.read_csv('../input/ozone-data/ozone.csv')
dat.head()


# In[ ]:


sns.lmplot(x="Jan", y="Feb", data=dat)


# In[ ]:


sns.lmplot(x="Jan", y="Mar", data=dat)


# In[ ]:


sns.lmplot(x="Mar", y="Apr", data=dat)


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(dat.Year, dat.Annual, ax=ax)
sns.rugplot(dat.Year, color="g", ax=ax)
sns.rugplot(dat.Annual, vertical=True, ax=ax);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(dat.Year, dat.Annual, cmap=cmap, n_levels=60, shade=True)


# In[ ]:


g = sns.jointplot(x="Year", y="Annual", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Year$", "$Annual$");

