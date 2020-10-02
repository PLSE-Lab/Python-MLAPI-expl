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

dat = pd.read_csv('../input/data-for-datavis/cancer_b.csv')
dat.head()


# In[ ]:


p = dat.hist(figsize = (20,20))


# In[ ]:


sns.regplot(x=dat['Radius (mean)'], y=dat['Area (mean)'])


# In[ ]:


sns.lmplot(x="Radius (mean)", y="Area (mean)", hue="Diagnosis", data=dat);


# In[ ]:


df = dat.rename({'Radius (mean)': 'Radius', 'Area (mean)': 'Area'}, axis=1)


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.Radius, df.Area, ax=ax)
sns.rugplot(df.Radius, color="g", ax=ax)
sns.rugplot(df.Area, vertical=True, ax=ax);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df.Radius, df.Area, cmap=cmap, n_levels=60, shade=True);


# In[ ]:


g = sns.jointplot(x="Radius (mean)", y="Area (mean)", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Radius (mean)", "$Area (mean)$");


# In[ ]:


datt = pd.read_csv('../input/data-for-datavis/cancer_m.csv')
datt.head()


# In[ ]:


q = datt.hist(figsize = (20,20))


# In[ ]:


sns.regplot(x=datt['Radius (mean)'], y=datt['Area (mean)'])


# In[ ]:


sns.lmplot(x="Radius (mean)", y="Area (mean)", hue="Diagnosis", data=datt);


# In[ ]:


df1 = datt.rename({'Radius (mean)': 'Radius', 'Area (mean)': 'Area'}, axis=1)


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df1.Radius, df1.Area, ax=ax)
sns.rugplot(df1.Radius, color="g", ax=ax)
sns.rugplot(df1.Area, vertical=True, ax=ax);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df1.Radius, df1.Area, cmap=cmap, n_levels=60, shade=True);


# In[ ]:


g = sns.jointplot(x="Radius (mean)", y="Area (mean)", data=datt, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Radius (mean)$", "$Area (mean)$");

