#!/usr/bin/env python
# coding: utf-8

# ## I am going to attempt to find correlation between popular kernels and their attributes ##
# 
# 
# ----------
# 
# 
# Each of these goes per language, **py** for python or **R** for..R
# 
# 
# ----------
# 
# 
#  1. Pair grid
#  2. Notebook -> upvotes
#  3. Comments -> upvotes
#  4. Length of title -> upvotes
#  5. Comments -> upvotes 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# what makes datasets popular?
datakern = pd.read_csv('../input/top-kernels.csv')


# Always view info to make sure you have no nulls

# In[ ]:


datakern.info()


# In[ ]:


datakern.describe()


# In[ ]:


datakern.columns


# Lets start with a small pair grid
# ====

# In[ ]:


sns.set(style="whitegrid")
sns.pairplot(datakern, hue="Language",
             x_vars=["Upvotes", "Comments"],
             y_vars=["Upvotes", "visuals"],
             palette="husl",
             size=3)


# Time to get deep
# ===

#  Is there a correlation of having a notebook and upvotes per language?
# ====

# In[ ]:


sns.set(style="whitegrid")

# Draw a categorical scatterplot to show each observation
sns.swarmplot(palette="husl", x="isNotebook", y="Upvotes", hue="Language", data=datakern)


# Clearly notebooks are the favorible choice.
# ====

# Comments to upvotes:
# ====
# **(obviously, there is going to be more because of popularity stances)**

# In[ ]:


sns.set(style='whitegrid', palette='husl')
# Draw using seaborn
sns.jointplot('Comments','Upvotes',
          data=datakern, ylim=(0,400),
          xlim=(0,200), kind='reg')


# Is there any correlation between the length of the title and upvotes?
# ====

# In[ ]:


lenttl = []
for i in datakern['Name']:
    lenttl.append(len(i))
    
up_to_title = pd.DataFrame(columns=('Title length', 'Upvotes'))
up_to_title['Title length'] = lenttl
up_to_title['Upvotes'] = datakern['Upvotes']


# In[ ]:


sns.set(style='whitegrid', palette='husl')
sns.jointplot('Title length','Upvotes',
          data=up_to_title, kind='reg')


# **It seems as though you could draw a hyperbola in the very middle, suggesting that medium size titles are more favorable.** 
# 
# 
# ----------
# 
# 
# Not as I expected.
# ====

# How do the comments vary with upvotes per language?
# ====

# In[ ]:


sns.set(style='whitegrid')
# Draw using seaborn
sns.lmplot('Comments','Upvotes',
          data=datakern, palette='husl',
          hue='Language', fit_reg=True)


# **And those are the predictions**
# *Cheers*
# ====
