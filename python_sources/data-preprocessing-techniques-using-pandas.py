#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 24, 2018
# 
# This is step-by-step guide to Data Preprocessing using Pandas, which I created for reference. I added some useful notes along the way to clarify things. 
# This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of Data Preprocessing.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Plot charts


# In[ ]:


# In PyCharm, set the correct directory as the current working directory
# Using Data.csv

dataset = pd.read_csv('../input/Data.csv')
x = dataset.iloc[:, :-1].values #all the columns except the last one
y = dataset.iloc[:, 3].values #Using the Fourth column which contains dependent variables

# x[:,1:3] = x[:, 1:3].astype('float32')


# **Now, let's prepare the data**

# In[ ]:


#Missing Data
#Most common way to deal with missing data: take the mean of columns, 
#and replace each missing data with respective mean of the value of that column

#Due to sklearn update, different syntax is used that is different from videos
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean') # --> Use np.nan to look for blank values
#if axis = 0: along columns, if axis = 1: along rows
#verbose: how much I want to see regarding the running process of the program
imp = imp.fit(x[:, 1:3])
x[:, 1:3] = imp.transform(x[:, 1:3])

x = pd.DataFrame(x) #Converts np array to a pd DataFrame
x.head()


# **Categorical Data** contains categories, such as Country names, Yes or No,  which are different from numeric values. As machine learning models are based on mathematics, categorical data needs to be encoded into numbers

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 0] = labelencoder_x.fit_transform(x.iloc[:, 0])
x.head()


# However, this makes machine learning models assume 2 > 1 > 0, which may affect the result, while categories cannot be compared. therefore, **Dummy encoding** should be used. This is also called **One Hot Encoding**. The category should still be labeled as numbers first, then encoded as OneHot.

# In[ ]:


from keras.utils import to_categorical #for onehot encoding --> use keras as it is easier. Note that sklearn and keras are great to be used together

encoded = pd.DataFrame(to_categorical(x.iloc[:, 0]))
pd.concat([encoded, x], axis = 1) #along rows, as they have same index


# In[ ]:


#Yes and No should be converted just to numbers, not OneHotEncoding

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Now, let's divide the data set: **Training and Test set**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#random state enables random sampling. Can put any number, but 42 is recommended


# **Feature Scaling** When two variables are very different in size, the bigger one dominates the machine learning model. This is why variables should be at the same scale. Strategies:
# ![182](https://i.imgur.com/r3i7RXH.png)

# In[ ]:


#Feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
#As sc_X already fitted to train set, it should only transform the X_test by applying its fit

#For Y_test and Y_train, no feature scaling as it is simple categories

#OneHotCoded values may or may not be scaled, but it depends on context

