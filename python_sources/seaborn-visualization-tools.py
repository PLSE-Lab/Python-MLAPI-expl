#!/usr/bin/env python
# coding: utf-8

# **Plot Types**
# 1. Bar Plot
# 1. Point Plot
# 1. Joint Plot
# 1. LM Plot
# 1. KDE Plot
# 1. Violin Plot
# 1. Heatmap
# 1. Box Plot
# 1. Swarm Plot
# 1. Pair Plot

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#read data
pokemon = pd.read_csv("../input/pokemon.csv")


# In[ ]:


pokemon.head()


# In[ ]:


pokemon.info()


# In[ ]:


#Bar Plot
plt.figure(figsize = (10,10))
sns.barplot(x = "Generation",y = "HP",data = pokemon)
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.xlabel("Generations")
plt.ylabel("HP")
plt.title("Generations per HP")
plt.show()


# In[ ]:


pokemon.head()


# In[ ]:


#Point Plot
f, ax1 = plt.subplots(figsize = (20,15))
sns.pointplot(x = "Type 1",y = "Attack", data = pokemon,color = "blue")
sns.pointplot(x = "Type 1",y = "Attack", data = pokemon,color = "red")
plt.grid()
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.xlabel("Type 1",color = "grey")
plt.ylabel("Attack",color = "pink")


# In[ ]:


#Joint Plot
jp = sns.jointplot(pokemon.Attack,pokemon.Generation)
plt.show()


# In[ ]:


#LM Plot
sns.lmplot(x = "Attack",y = "Defense",data = pokemon )
plt.xlabel("Attack",color = "r")
plt.xticks(rotation = 60)
plt.ylabel("Defense",color = "cyan")
plt.yticks(rotation = 60)
plt.title("Attack VS Defense",color ="yellow")
plt.show()


# In[ ]:


#KDE Plot
sns.kdeplot(pokemon.Attack,pokemon.Defense)


# In[ ]:


#Violin Plot
palle = sns.cubehelix_palette(1, rot=-.6, dark=.4)
sns.violinplot(data=pokemon,palette=palle, inner ="points")
plt.xticks(rotation =15)


# In[ ]:


pokemon.corr()


# In[ ]:


#Heatmap
f, ax =plt.subplots(figsize = (10,10))
sns.heatmap(pokemon.corr(),linewidths=0.4,linecolor="black",alpha=0.8)
plt.xticks(rotation=90,color="blue")
plt.yticks(color="blue")


# In[ ]:


#Box Plot
sns.boxplot(x = "Generation", y = "HP",data = pokemon)


# In[ ]:


#Swarm Plot
sns.swarmplot(x = "Generation",y = "HP", data = pokemon)


# In[ ]:


#Pair Plot
sns.pairplot(pokemon)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




