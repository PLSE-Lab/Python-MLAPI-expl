#!/usr/bin/env python
# coding: utf-8

# # Simple Visualization

# ## 1. Importing and taking a look at the data

# Import basic libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 2D plotting library
import seaborn as sns # Data visualization library based on matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Load the `.csv` file for the training data:

# In[2]:


train = pd.read_csv("../input/train.csv");


# We can print a summary of the data frame

# In[3]:


train.info(verbose=True, null_counts=True)
# verbose=False would not display data for each column


# ## 2. Visualizing survival vs sex using matplotlib

# In[7]:


survived_sex = train[train["Survived"] == 1]["Sex"].value_counts()
print(survived_sex)


# In[9]:


dead_sex = train[train["Survived"] == 0]["Sex"].value_counts()
print(dead_sex)


# In[10]:


df = pd.DataFrame([survived_sex, dead_sex])
print(df)


# In[11]:


df.index = ["Survived", "Dead"]
print(df)


# Before we plot `df`, we can define some useful fonts:

# In[ ]:


myfont = {"family": "serif",
        "color":  "darkred",
        "weight": "normal",
        "size": 16,
        } # For plot titles

myfont2 = {"size": 16} # For matplotlib axis labels. The default font is too small.


# In[32]:


plt.style.use("seaborn") # This line is optional. It sets matplotlib's style (colors) to a style called "seaborn", so it will match the chart we will do later with Seaborn

df.plot(kind="bar", stacked=True, figsize=(15, 8))
plt.title("Survival vs sex", fontdict=myfont) # fontdict is an optional parameter.
plt.ylabel("No. of ppl", fontdict=myfont2)
plt.show()


# ## 3. Using Seaborn to visualize survival rate vs passenger class, by sex

# In[50]:


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, aspect=2, kind="bar", palette="muted") # "aspect" is optional. Image size can be defined using "size" only.
# There are six variations of the default theme, called color palette: deep, muted, pastel, bright, dark, and colorblind.
g.fig.suptitle("Survival rate vs passenger class, by sex", fontdict=myfont)


# For a gallery of Seaborn visualizations with code visit: [Seaborn Example Gallery](https://seaborn.pydata.org/examples/index.html)
