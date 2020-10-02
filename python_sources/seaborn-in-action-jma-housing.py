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

dat = pd.read_csv('../input/california-housing-prices/housing.csv')
dat.head()


# In[ ]:


sns.lmplot(x="total_rooms", y="total_bedrooms", hue="ocean_proximity",  data=dat);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(dat.median_income, dat.median_house_value, ax=ax)
sns.rugplot(dat.median_income, color="g", ax=ax)
sns.rugplot(dat.median_house_value, vertical=True, ax=ax);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(dat.median_income, dat.median_house_value, cmap=cmap, n_levels=60, shade=True);

