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

dat = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
dat.head()


# In[ ]:


p = dat.hist(figsize = (20,20))


# In[ ]:


sns.regplot(x=dat['Glucose'], y=dat['Insulin'])


# In[ ]:


sns.lmplot(x="Glucose", y="Insulin", hue="Outcome", data=dat)


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(dat.Glucose, dat.Insulin, ax=ax)
sns.rugplot(dat.Glucose, color="g", ax=ax)
sns.rugplot(dat.Insulin, vertical=True, ax=ax);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(dat.Glucose, dat.Insulin, cmap=cmap, n_levels=60, shade=True);


# In[ ]:


g = sns.jointplot(x="Glucose", y="Insulin", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Glucose$", "$Insulin$");

