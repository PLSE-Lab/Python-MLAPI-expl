#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# # Class distribution: 357 benign, 212 malignant

# In[ ]:


df["diagnosis"].value_counts()


# # Data Description
# 
# ID number
# diagnosis
# 
# The diagnosis of breast tissues (M = malignant, B = benign)
# radius_mean
# 
# mean of distances from center to points on the perimeter
# texture_mean
# 
# standard deviation of gray-scale values
# perimeter_mean
# 
# mean size of the core tumor
# area_mean
# smoothness_mean
# 
# mean of local variation in radius lengths
# compactness_mean
# 
# mean of perimeter^2 / area - 1.0
# concavity_mean
# 
# mean of severity of concave portions of the contour
# concave points_mean
# 
# mean for number of concave portions of the contour
# symmetry_mean
# fractal_dimension_mean
# 
# mean for "coastline approximation" - 1
# radius_se
# 
# standard error for the mean of distances from center to points on the perimeter
# texture_se
# 
# standard error for standard deviation of gray-scale values
# perimeter_se
# area_se
# smoothness_se
# 
# standard error for local variation in radius lengths
# compactness_se
# 
# standard error for perimeter^2 / area - 1.0
# concavity_se
# 
# standard error for severity of concave portions of the contour
# concave points_se
# 
# standard error for number of concave portions of the contour
# symmetry_se
# fractal_dimension_se
# 
# standard error for "coastline approximation" - 1
# radius_worst
# 
# "worst" or largest mean value for mean of distances from center to points on the perimeter
# texture_worst
# 
# "worst" or largest mean value for standard deviation of gray-scale values
# perimeter_worst
# area_worst
# smoothness_worst
# 
# "worst" or largest mean value for local variation in radius lengths
# compactness_worst
# 
# "worst" or largest mean value for perimeter^2 / area - 1.0
# concavity_worst
# 
# "worst" or largest mean value for severity of concave portions of the contour
# concave points_worst
# 
# "worst" or largest mean value for number of concave portions of the contour
# symmetry_worst
# fractal_dimension_worst
# 
# "worst" or largest mean value for "coastline approximation" - 1

# In[ ]:


#visualize the count
sns.countplot(df.diagnosis,label="count")
plt.show()


# # PAIR-PLOT

# In[ ]:


sns.pairplot(df.iloc[:,1:6],hue="diagnosis")
plt.show()


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(20,18))

corr = df.corr()
ax = sns.heatmap(corr,vmin=-1,vmax=1,center=0,annot=True)


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['radius_mean'], hist=True, bins=30, color='grey')
plt.xlabel('radius_mean')
plt.ylabel('Frequency')
plt.title('Distribution of radius_mean', fontsize=15)


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['concavity_mean'], hist=True, bins=30, color='grey')
plt.xlabel('concavity_mean')
plt.ylabel('Frequency')
plt.title('Distribution of concavity_mean', fontsize=15)


# In[ ]:


plt.figure(figsize=(20,10))
ax = sns.boxplot(data = df, orient = "h", palette = "Set1")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




