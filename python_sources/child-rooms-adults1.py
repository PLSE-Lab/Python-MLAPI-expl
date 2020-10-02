#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime


# In[ ]:


import datetime


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000000)


# In[ ]:


train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000)


# In[ ]:


mask = train.is_booking == True
mask1 = train1.is_booking == True


# In[ ]:


trainm = train[mask]
trainm1 = train1[mask1]


# In[ ]:


#understanding the numerical content 

b1 = trainm[['srch_adults_cnt','srch_children_cnt','srch_rm_cnt']]
b2 = trainm1[['srch_adults_cnt','srch_children_cnt','srch_rm_cnt']]


# In[ ]:


b1.describe()


# In[ ]:


b2.describe()


# In[ ]:


b2.info()


# In[ ]:


# putting the above separately
sns.countplot(y='srch_adults_cnt', data=trainm)


# In[ ]:


# putting the children
sns.countplot(y='srch_children_cnt', data=trainm)


# In[ ]:


# putting the children
sns.countplot(y='srch_rm_cnt', data=trainm)


# In[ ]:


# putting the above separately
sns.countplot(y='srch_adults_cnt', hue='srch_rm_cnt', data=trainm)


# In[ ]:


# putting the above separately
sns.countplot(y='srch_adults_cnt', hue='srch_children_cnt', data=trainm)


# In[ ]:


sns.set(style="ticks", context="talk")

# Make a custom sequential palette using the cubehelix system
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)

# Plot tip as a function of toal bill across days
g = sns.lmplot(x="srch_children_cnt", y="srch_adults_cnt", hue="srch_rm_cnt", data=trainm1,
               palette=pal, size=7)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Number of Children", "Number of Adults")


# In[ ]:




