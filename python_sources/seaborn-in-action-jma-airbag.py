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

dat = pd.read_csv('../input/airbag-and-other-influences/nassCDS.csv')
dat.head()


# In[ ]:


sns.lmplot(x="weight", y="injSeverity", hue="dead", data=dat)


# In[ ]:


sns.lmplot(x="weight", y="injSeverity", hue="airbag", data=dat)


# In[ ]:


sns.lmplot(x="weight", y="injSeverity", hue="seatbelt", data=dat)


# In[ ]:


sns.lmplot(x="weight", y="injSeverity", hue="sex", data=dat)


# In[ ]:


sns.lmplot(x="weight", y="injSeverity", hue="deploy", data=dat)


# In[ ]:


sns.lmplot(x="weight", y="injSeverity", hue="occRole", data=dat)


# In[ ]:


g = sns.jointplot(x="weight", y="injSeverity", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$weight$", "$injSeverity$");


# In[ ]:


g = sns.jointplot(x="ageOFocc", y="injSeverity", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$ageOFocc$", "$injSeverity$");

