#!/usr/bin/env python
# coding: utf-8

# This is my first kaggle kernel. This is more of knowing kaggle kernel usage and capabilities. 
# I have taken help from various public souces including some of the existing kernels. Thanks in advance to them..
# 
# **Dataset: **
# Iris dataset
# 
# 3 species of the flowers
# 
# 50 samples in each species
# 
# 6 columns/properties
# 
# One flower species is linearly seperable from other two, but those two are not lineearly seperable from each other.
# 
# 
# **Language:** Python 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Python 2D plotting library 
import seaborn as sns # python graphing library


sns.set(style="white", color_codes=True)

# Data in kaggle kerner lives in "../input/" directory
# Now I am going to import the dataset using pandas dataaframe
irisDataset = pd.read_csv("../input/Iris.csv") 

#Lets see how Iris dataset looks like

irisDataset.head(10)


# In[ ]:


# How many samples of each species exist
irisDataset["Species"].value_counts()


# In[ ]:


# I always thought Petal are usually longer than sepal
# Lets see what is the behavior in the various species.
df = irisDataset[irisDataset["SepalLengthCm"] > irisDataset["PetalLengthCm"]]
print("Original Dataset")
print(irisDataset.describe())
print("\n\nDataset where sepal is longer than petal")
print(df.describe())


# In[ ]:


# Now let's plot the samples using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Iris features.
irisDataset.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[ ]:


# Since scatter plot doesn't show the density of the samples.
# We'll make a hexbin plot of the Iris features.
irisDataset.plot(kind="hexbin", x="SepalLengthCm", y="SepalWidthCm")


# In[ ]:


# Now we want to see each flower species seperately
# Seaborn's FacetGrid can help us color the scatterplot by species type
sns.FacetGrid(irisDataset, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# > *In the above plot we can see that one species is linearly seperable from the other two, but two are overlapping.*

# In[ ]:


# Now we want to see the density of each flower species seperately based on the SepalLengthCm
# Seaborn's FacetGrid can help us plot/color the histogram by species type and frequency
sns.FacetGrid(irisDataset, col="Species", hue="SepalLengthCm", size=5, aspect=.7)    .map(plt.hist, "SepalLengthCm") 


# We can see a significant overlap in the representation on SepalLengthCm, so this doesn't adds much value to our analysis.

# In[ ]:


# Now we want to see the density of each flower species seperately based on the PetalLengthCm
# Seaborn's FacetGrid can help us plot/color the histogram by species type and frequency
sns.FacetGrid(irisDataset, col="Species", hue="PetalLengthCm", size=5, aspect=.7)    .map(plt.hist, "PetalLengthCm") 


# We can see PetalLengthCm clearly signifies a difference between 1st and other two species.
# Though we can see a partial overlap in the 2nd and 3rd species.

# In[ ]:


# We will look at the box plot keeping Species as the category through a Seaborn boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=irisDataset, palette="Set3")


# In[ ]:


# We will use swarmplot() to show the datapoints on top of the boxes:Seaborn
sns.boxplot(x="Species", y="PetalLengthCm", data=irisDataset, palette="Set3")
sns.swarmplot(x="Species", y="PetalLengthCm", data=irisDataset)


# In[ ]:


# Now we will represent the data via Violin plot which looks:
# fatter for more data
# thiner for sparser data
sns.violinplot(x="Species", y="PetalLengthCm", data=irisDataset, size=6)


# In[ ]:


# TO see the clear overlap on linear scale we will use Kdeplot
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(irisDataset, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# In[ ]:


# We will see relationship between each feature using sns.pairplot
sns.pairplot(irisDataset.drop("Id", axis=1), hue="Species", size=3)


# In[ ]:


# Now we will see few pandas plots
from pandas.plotting import andrews_curves
andrews_curves(irisDataset.drop("Id", axis =1), 'Species')


# In[ ]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(irisDataset.drop("Id", axis=1), "Species")


# In[ ]:


from pandas.plotting import radviz
radviz(irisDataset.drop("Id", axis=1), "Species")


# For reference : 
# 
#     1> https://pandas.pydata.org/pandas-docs/stable/visualization.html
#     
#     2> https://seaborn.pydata.org/tutorial.html
