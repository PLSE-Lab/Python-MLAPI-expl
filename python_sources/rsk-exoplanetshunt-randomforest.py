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


# #### Load Datasets

# In[ ]:


train_df = pd.read_csv('/kaggle/input/kepler-labelled-time-series-data/exoTrain.csv')
test_df = pd.read_csv('/kaggle/input/kepler-labelled-time-series-data/exoTest.csv')


# #### View The First 5 Rows

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape


# ### Check For The Missing Values In Training Set

# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df.isna().sum()


# There are too many columns to insect the missing values manually. Create a function which checks for the missing values for you.

# In[ ]:


count_miss_values = 0
for column in train_df.columns:
    for item in train_df[column].isnull():
        if item == True:
            count_miss_values += 1
            
count_miss_values


# So, there are no missing values in the training set.
# 
# 
# ### Scatter & Line Plots
# Creating scatter plots and line plots for two stars labelled as 1 and two stars labelled as 2.

# In[ ]:


# Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# First Star In The Dataset
star0 = train_df.iloc[0, :]
star0.head()


# In[ ]:


# Scatter Plot For First Star
plt.figure(figsize=(15, 5))
plt.scatter(pd.Series([i for i in range(1, len(star0))]), star0[1:])
plt.ylabel('Flux')
plt.show()


# There is a period fluctuation in the Flux values for the first star. This shows that the star has at least one planet.

# In[ ]:


# Line Plot For First Star
plt.figure(figsize=(15, 5))
plt.plot(pd.Series([i for i in range(1, len(star0))]), star0[1:])
plt.ylabel('Flux')
plt.show()


# In[ ]:


# Second Star
star1 = train_df.iloc[1, :]
star1.head()


# In[ ]:


# Scatter Plot For Second Star
plt.figure(figsize=(15, 5))
plt.scatter(pd.Series([i for i in range(1, len(star1))]), star1[1:])
plt.ylabel('Flux')
plt.show()


# In[ ]:


# Line Plot For Second Star
plt.figure(figsize=(15, 5))
plt.plot(pd.Series([i for i in range(1, len(star1))]), star1[1:])
plt.ylabel('Flux')
plt.show()


# So, there is a clear period fluctiona in the flux values. This also confirms that the Star has at least 1 planet.

# In[ ]:


train_df.tail()


# In[ ]:


# Last Star
star5086 = train_df.iloc[5086, :]
star5086.head()


# In[ ]:


# Scatter Plot For Last Star
plt.figure(figsize=(15, 5))
plt.scatter(pd.Series([i for i in range(1, len(star5086))]), star5086[1:])
plt.ylabel('Flux')
plt.show()


# There is no clear periodic fluctuation. Hence, we can't say for sure whether the star has a planet or not. Let's reconfirm this with a line plot.

# In[ ]:


# Line Plot For Last Star
plt.figure(figsize=(15, 5))
plt.plot(pd.Series([i for i in range(1, len(star5086))]), star5086[1:])
plt.ylabel('Flux')
plt.show()


# The line plot confirms that there is no clear periodic fluctuation in the light intensity values.

# In[ ]:


# Second-Last Star
star5085 = train_df.iloc[5085, :]
star5085.head()


# In[ ]:


# Scatter Plot For Second-Last Star
plt.figure(figsize=(15, 5))
plt.scatter(pd.Series([i for i in range(1, len(star5085))]), star5085[1:])
plt.ylabel('Flux')
plt.show()


# In[ ]:


# Line Plot For Second-Last Star
plt.figure(figsize=(15, 5))
plt.plot(pd.Series([i for i in range(1, len(star5085))]), star5085[1:])
plt.ylabel('Flux')
plt.show()


# ### Applying Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


# Split the dataframe into feature variables and the target variable.
x_train = train_df.iloc[:, 1:]
x_train.head()


# In[ ]:


y_train = train_df.iloc[:, 0]
y_train.head()


# In[ ]:


rf_clf1 = RandomForestClassifier(n_jobs=-1)
rf_clf1.fit(x_train, y_train)
rf_clf1.score(x_train, y_train)


# The model has fit the dataset with a whopping accuracy of $99$%. Very likely, this might be a case of overfitting.

# In[ ]:


x_test = test_df.iloc[:, 1:]
x_test.head()


# In[ ]:


y_test = test_df.iloc[:, 0]
y_test.head()


# In[ ]:


y_test.shape


# In[ ]:


y_predicted = rf_clf1.predict(x_test)
y_predicted.shape


# In[ ]:


# Confusion Matrix
# In binary classification, the count of true negatives is C(0, 0), false negatives is C(1, 0), true positives is C(1, 1) and false positives is C(0, 1).
# [[TP, FN], 
#  [FP, TN]]
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predicted)


# In[ ]:


y_predicted = pd.Series(y_predicted)
y_predicted.value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:


print(classification_report(y_test, y_predicted))


# In[ ]:


accuracy_score(y_test, y_predicted)

