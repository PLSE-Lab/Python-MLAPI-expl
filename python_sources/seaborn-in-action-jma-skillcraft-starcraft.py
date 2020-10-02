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

dat = pd.read_csv('../input/skillcraft/SkillCraft.csv')
dat.head()


# In[ ]:


p = dat.hist(figsize = (20,20))


# In[ ]:


sns.regplot(x=dat['MinimapAttacks'], y=dat['MinimapRightClicks'])


# In[ ]:


sns.lmplot(x="MinimapAttacks", y="MinimapRightClicks", data=dat)


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(dat.MinimapAttacks, dat.MinimapRightClicks, ax=ax)
sns.rugplot(dat.MinimapAttacks, color="g", ax=ax)
sns.rugplot(dat.MinimapRightClicks, vertical=True, ax=ax);


# In[ ]:


g = sns.jointplot(x="MinimapAttacks", y="MinimapRightClicks", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$MinimapAttacks$", "$MinimapRightClicks$");

