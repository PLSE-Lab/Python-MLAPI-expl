#!/usr/bin/env python
# coding: utf-8

# # Subplot tutorial in Python
# 
# Here's a quick code tutorial using the Iris dataset. This is a dataset that is introduced by the British statistician and biologist Ronald Fisher in 1936. 
# 
# The original data set (also known as Fishers' Iris Dataset) consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. He then used the data to create a linear discriminat for each of the species.
# 
# With so many features, it is always handy to do subplots. The ability to create subplots will enable you to present data in a better way - as you can fit more charts in one / do some comparisons with the data.

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt # main visualization library
import seaborn as sns # sits ontop of matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


iris = pd.read_csv('../input/iris-data/Iris.csv') # load in the data


# In[ ]:


iris.head(10) # show first 10 rows of the data


# In[ ]:


iris.columns


# In[ ]:


#create a figure "fig" with axis "ax1" with 3x2 configuration
fig, ax1 = plt.subplots(3,2, sharex='col', figsize=(22,18), gridspec_kw={'hspace': 0, 'wspace': 0.1}) 


# 1st plot
sns.set_style("whitegrid");
sns.scatterplot(data=iris, x="SepalLengthCm", y="SepalWidthCm", hue="Species", ax=ax1[0, 0], legend='brief') 

# 2nd plot
sns.scatterplot(data=iris, x="SepalWidthCm", y="SepalLengthCm", hue="Species", ax=ax1[0, 1], legend='brief') 

# 3rd plot
sns.scatterplot(data=iris, x="SepalLengthCm", y="PetalLengthCm", hue="Species", ax=ax1[1, 0], legend='brief') 

# 4th plot
sns.scatterplot(data=iris, x="SepalWidthCm", y="PetalLengthCm", hue="Species", ax=ax1[1, 1], legend='brief') 

# 5th
sns.scatterplot(data=iris, x="SepalLengthCm", y="PetalWidthCm", hue="Species", ax=ax1[2, 0], legend='brief') 

# 6th
sns.scatterplot(data=iris, x="SepalWidthCm", y="PetalWidthCm", hue="Species", ax=ax1[2, 1], legend='brief') 

fig.savefig("/kaggle/working/output.png")


# **6 unique pair plots in 3x2 grid plot**
# 
# To view the actual Python code, click on the [code] box above the chart.
# 
# Additional features : for less clutter
# - sharex - share same axis for SeptalLengthcm and septalWidthCm
# - gridspec_kw={'hspace': 0} which reduces height between each graph
# 
# Note: access the ax1 object like a 3x2 matrix
# where each subplot is denoted by the position in the matrix as below:
# 
# `[[0,0], [1,0]
#  [1,0], [1,1]
#  [2,0], [2,1]]`

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(iris, hue="Species", size=3);
plt.show()


# Seaborn has a quick feature called "pairplot" which allows one to quickly place all combinations of features. 
# This is a nice way to have a visual impact on the data.
# 
# As always, if you learnt something or enjoyed the read in general please do give this kernel an upvote on the top right! That would be greatly appreciated! 

# 
