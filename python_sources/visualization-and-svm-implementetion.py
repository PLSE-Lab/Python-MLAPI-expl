#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. [Importing Libraries and Packages](#p1)
# 2. [Loading and Viewing Data Set](#p2)
# 3. [Clean and Normalization Data](#p3)
# 4. [Visualization](#p4)
# 5. [Initializing, Optimizing, and Predicting](#p5)

# <a id="p1"></a>
# # 1. Importing Libraries and Packages
# We will use these packages to help us manipulate the data and visualize the features/labels as well as measure how well our model performed. Numpy and Pandas are helpful for manipulating the dataframe and its columns and cells. We will use matplotlib along with Seaborn to visualize our data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="p2"></a>
# # 2. Loading and Viewing Data Set
# With Pandas, we can load both the training and testing set that we wil later use to train and test our model. Before we begin, we should take a look at our data table to see the values that we'll be working with. We can use the head and describe function to look at some sample data and statistics.

# In[ ]:


# Import dataset
data = pd.read_csv("../input/voice.csv")


# In[ ]:


# Showing five columns
data.head()


# In[ ]:


# Showing five column
data.tail()


# In[ ]:


# Describing data show us statics features
data.describe()


# <a id="p3"></a>
# # 3. Clean and Normalization Data
# We need to change categorical data to numeric data.

# In[ ]:


data.label = [1 if each == "female" else 0 for each in data.label ]
y = data.label.values
x_data = data.drop(["label"], axis = 1)


# In[ ]:


# Normalization
x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# <a id="p4"></a>
# # 4. Visualization
# 
# In order to visualizate the data, we are goingo to use matplotlib and seaborn. Before the visualization don't forget the normalize the data.

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


sns.set(style="white")
df = x.loc[:,['meandom','mindom','maxdom']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# In[ ]:


# Plotting
data.plot(kind='scatter', x='meanfreq', y='dfrange')
data.plot(kind='kde', y='meanfreq')


# In[ ]:


# Pairplotting
sns.pairplot(data[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']], 
                 hue='label', size=3)


# <a id="p5"></a>
# # 5. Initializing, Optimizing, and Predicting
# Now that our data has been processed and formmated properly, and that we understand the general data we're working with as well as the trends and associations, we can start to build our model. We can import different classifiers from sklearn. 

# # SVM

# In[ ]:


# Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


# Importing SVM from sklearn
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)


# In[ ]:


# Testing
print("Print accuracy of svm algorithm: ", svm.score(x_test,y_test))


# **If you liked the kernel, please upvote or make a comment. They motivate me :)**
