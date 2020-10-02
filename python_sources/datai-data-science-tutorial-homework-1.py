#!/usr/bin/env python
# coding: utf-8

# # Data Science - Analyze World Happiness 2015  with Python
# <a id="0"></a> <br>
# 
#     Firstly, DATAI (Kaan) i have to thank you very much about your tutorials. I follow your trainings through Udemy. I look forward to your new tutorial series and wish you success.
#      
#     This is my first kernel on Kaggle. I create this kernel for my homework and  i am going to show what i learned from Data ScienceTutorial for Beginners, in this kernel.
#     
# ### Note : I am still working on kernel.
#      
# **Whats inside this kernel **
# 
# 1. **Episode - 1:**
#     1. [Importing Libraries](#1)
#     1. [Reading Data and Taking Basic Information](#2)
#     1. [Cleaning Data ](#3)
#     1. [Basic Visualization](#4)
#        1. [Matplotlib](#5)
#        1. [Seaborn](#6)
#     1. [Filtering Data](#7)

# <a id="1"></a> <br>
# **Importing Libraries**

# In[ ]:


import numpy as np # Linear Algebra
import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Basic Visualization
import seaborn as sns # Visualition
import re # Regular Expression

import os
print(os.listdir("../input"))


# <a id="2"></a> <br>
# **Reading Data and Taking Basic Information**

# In[ ]:


#Reading Comma Separated Values (.csv) file.
data = pd.read_csv("../input/2015.csv")

#Taking basic informations.
data.info()
data.describe()


# In[ ]:


#Taking first 5 samples from data.
data.head()


# <a id="3"></a> <br>
# **Cleaning Data**
# 
# We need to clean data. Because a lot of datas not clean to use in Python. For example, in our data there are some gaps between columns names. It cause that; we cannot use it in our codes because it not appropriate for dot notation.

# In[ ]:


#Backing up the columns name to compare it with the modified columns name.
pre_names = [each for each in data.columns]

#Searching for gaps, other invalid characters and clearing them with Regular Expression library.
data.columns = [ re.sub("[ \(\)]", "", each).lower() for each in data.columns]

#Now let's look comparison with before and after.

lis = list(data.columns)
print("Before Names of Columns".center(94,"-"),end="\n\n")
for each in pre_names:
    print(each.ljust(29), end=" | ")
print("\n")
print("After Names of Columns".center(94,"-"),end="\n\n")
for each in lis:
    print(each.ljust(29), end=" | ")


# <a id="4"></a> <br>
# **Basic Visualization** 
# 
# In this section i will use some basic visualization technique for visualization of data informations.

# <a id="5"></a> <br>
# **Matplotlib**
# 
# Matplotlib is basic library for visualization of data information in Python.

# In[ ]:


#Line Plot - 1
data.healthlifeexpectancy.plot(kind = "Line", label = "healthlifeexpectancy", color = "r",
                              linewidth = 1, linestyle = "--", grid = True, alpha = 0.7,
                              figsize = (20,10))
data.economygdppercapita.plot(label = "economygdppercapita", color = "b",
                             linewidth = 1, linestyle = "-.", alpha = 0.7)
plt.legend(loc = "upper right", fontsize = "large")
plt.title("This is Line Plot - Relationsip Between Healt Life Expectancy and Economy GDP per Capita")
plt.xlabel("Happiness Rank", fontsize = "large")
plt.ylabel("Health Life Expectancy and Economy GDP per Capita", fontsize = "large")
plt.show()


# In[ ]:


#Line Plot - 2
data.generosity.plot(kind = "Line", label = "generosity", color = "g",
                    linewidth = 2, linestyle = "-.", grid = True,
                    figsize = (20,10))
plt.legend(loc = "upper right")
plt.title("This is a Line Plot - Generosity")
plt.xlabel("Happiness Rank")
plt.ylabel("Generosity")
plt.show()


# In[ ]:


#Scatter Plot - 1
data.plot(kind = "scatter", x = "happinessscore", y = "freedom", color = "g",
          alpha = 0.5, grid = True,s=80,
          figsize =(20,10))
plt.show()


# In[ ]:


#Scatter Plot - 2
ax = data.plot(kind = "scatter", x = "freedom", y = "generosity", color = "red",
          alpha = 0.5, grid = True,s=50,
          figsize =(20,10))
data.plot(kind = "scatter", x = "freedom", y = "healthlifeexpectancy", color = "blue",
          alpha = 0.5, grid = True,s=50,
          figsize =(20,10), ax=ax)
plt.show()


# In[ ]:


#Scatter Plot - 3
data.plot(kind = "scatter", x = "freedom", y = "healthlifeexpectancy", color = "blue",
          alpha = 0.5, grid = True,s=data['freedom']*650,
          figsize =(20,10))
plt.show()


# In[ ]:


#Box Plot -1

data.generosity.plot(kind = "box", grid = True, figsize = (10,10))
plt.title("This is a Box Plot")
plt.show()


# In[ ]:


#Area Plot - 1

data.happinessscore.plot(kind = "area", label = "happinessscore", color = "b",
                 linewidth = 1, linestyle = "--", grid = True,
                 alpha = 0.5, stacked=False, figsize = (20,10))
plt.show()


# In[ ]:


# Bar Plot

a = list(data['region'].unique())
b = []

for each in range(len(a)):
    x = data["region"] == a[each]
    k = list(data[x].happinessscore.describe())[1]
    b.append(k)
    if len(a[each])> 20:
        t = []
        for i in a[each].split():
            t.append(i[:3])
        a[each] = ".".join(t)

plt.figure(figsize=(20,10))
plt.bar(a,b)
plt.xlabel("Regions", fontsize = "large")
plt.ylabel("Average Happiness Score According to Regions", fontsize = "large")
plt.show()


# <a id="6"></a> <br>
# **Seaborn**
# 
# Seaborn is strong and popular library for visualization of data information in Python. There are lots of plots in seaborn but i will use only heatmap for just now.

# In[ ]:


#Data Correlation
data.corr()


# In[ ]:


#Heatmap Plot

f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = ".2f", ax=ax)
plt.show()


# <a id="7"></a> <br>
# **Filtering Data**

# In[ ]:


#Filtering with np.logical_and

x = data["happinessscore"]>7
y = data["happinessscore"]<7.2

data[np.logical_and(x, y)]


# In[ ]:


#This is just show a sample for using while loop.
i = 0
while data["happinessscore"][i]>6:
    i +=1

print("The happiness score value of {} countries is higher than 6.".format(i))


# 1. [Go to Top Page](#0)
