#!/usr/bin/env python
# coding: utf-8

# **Statistics of cut straws**
# 
#     This kernel is a complementary study for a small project developed by Prof. FABIOLA EUGENIO ARRABACA MORAES during statistics class in University of Uberaba, Brazil. 
#     The project consist in cutting long straws in 11 cm's long pieces, however, once it is handmade, it obviously didn't worked out precisely, so the idea is to understand statistics from studying the imperfections on the straw's pieces.
#     It was computed 120 straw's pieces and those were divided in 3 groups of 40 pieces each randomly, so let's first import it.

# In[ ]:


import os, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('../input/canudos_data.csv')
dataset.head()


# Each column represent a group of 40 straws and each value is the straw's size in centimeters.
# So, now, let's split in 3 different datasets, one for each group.

# In[ ]:


dataset_1 = dataset['1']
dataset_2 = dataset['2']
dataset_3 = dataset['3']

dataset_1.head()


# Alright, now that we have each group separated, let's start to vizualize it.

# In[ ]:


plt.plot(dataset_1)
plt.plot(dataset_2)
plt.plot(dataset_3)
plt.show()


# Ok, this is something... But there is no valuable information that we could get out of this graph.

# In[ ]:


a = []
for i in list(dataset_1):
    a.append(float(i))

b = []
for i in list(dataset_2):
    b.append(float(i))

c = []
for i in list(dataset_3):
    c.append(float(i))

sns.set(style="darkgrid")

sns.jointplot(range(40), a, kind="reg", color="m")
sns.jointplot(range(40), b, kind="reg", color="m")
sns.jointplot(range(40), c, kind="reg", color="m")


# Nice! This is way better. We can clearly view a pattern between the 3 groups, but let's go further to see what we can get.

# In[ ]:


bp_a = sns.boxplot(data=dataset_1)


# In[ ]:


bp_b = sns.boxplot(data=dataset_2)


# In[ ]:


bp_c = sns.boxplot(data=dataset_3)


# We will work more on this kernel during the next fwe weeks, so stay tuned.
