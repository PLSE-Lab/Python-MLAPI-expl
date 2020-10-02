#!/usr/bin/env python
# coding: utf-8

# **Predicion of Insurance Cost**  
# Steps of this excercise is going to be as following:
# * Import Data
# * Investigate Data
# * Clean Data
# * Divide into training and testing data
# * Traing LR model
# * Test the model

# **Import Necessary Libraries**  
# * numpy as from sklearn's linear model accepts numpy array type data and we are going to use it for analysis of the data
# * pandas as it will hgelp as working on the data and import CSV files.
# * seaborn for statistical visual data representation
# * matplotlib.pyplot so that I can create subplots
# * Sklearn.model_selection so that we can split that data into training and testing set

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score


# **Import Data**

# In[ ]:


#import data to using pandas function read_csv and storing it into personal_data
personal_data = pd.read_csv('../input/insurance.csv')
#show the first five entries in personal_data
personal_data.head()


# **Clean Data**
# * We need to convert in the sex column male = 0 and female = 1
# * smoke column we need to chage yes = 1 and no = 0
# * need to change region to southwest = 0, southeast = 1, northwest = 2, and northeast = 3

# In[ ]:


personal_data['sex'] = personal_data['sex'].map({'female': 1, 'male': 0})
personal_data['smoker'] = personal_data['smoker'].map({'yes':1,'no':0})
personal_data['region'] = personal_data['region'].map({'southwest':0,'southeast':1,'northwest':2,'northeast':3})
personal_data.head()


# **Data Analytics:**  
# Lets now examine that data and look at the following features:
# * In our data what is the range of age we have (histogram might be the most appropriate method) to see if we have enough data.
# * Do above for each field of the data

# In[ ]:


figure, ax = plt.subplots(2,4, figsize=(24,8))
sns.distplot(personal_data['age'],ax=ax[0,0])
sns.countplot(personal_data['sex'],ax=ax[0,1])
sns.distplot(personal_data['bmi'],ax= ax[0,2])
sns.distplot(personal_data['children'],ax= ax[0,3])
sns.countplot(personal_data['smoker'],ax= ax[1,0])
sns.countplot(personal_data['region'],ax= ax[1,1])
sns.distplot(personal_data['charges'],ax= ax[1,2])


# We can see from the above that all data is evenly divided.

# **Spliting Data into training and testing data**
# In order for us to be able to train the data and test the data we will be using sklearn's split method. The method randomly selects and divides the data into two sets. We will work on more on it afterwards. We will be using 80-20 rule for spliting data into training and testing datasets.

# In[ ]:


train_personal_data, test_personal_data = train_test_split(personal_data, test_size=0.2)
print("size of train data set:", train_personal_data.shape)
print("size of test data set:", test_personal_data.shape)


# **Some Co Relation Testing**  
# Before getting into multi feature linear regression, lets first look at individual feature and see the effect on it on charges.

# In[ ]:


# graph to see if there is any linearity and what is the co relation between independent features and charges
figure, ax = plt.subplots(2,3, figsize=(24,8))
sns.regplot(x=train_personal_data["age"], y=train_personal_data["charges"], ax=ax[0,0])
sns.regplot(x=train_personal_data["sex"], y=train_personal_data["charges"], ax=ax[0,1])
sns.regplot(x=train_personal_data["bmi"], y=train_personal_data["charges"], ax=ax[0,2])
sns.regplot(x=train_personal_data["children"], y=train_personal_data["charges"], ax=ax[1,0])
sns.regplot(x=train_personal_data["smoker"], y=train_personal_data["charges"], ax=ax[1,1])
sns.regplot(x=train_personal_data["region"], y=train_personal_data["charges"], ax=ax[1,2])


# Following are some of the observations from the above graphs:  
# * Age has some coorelation but there are alot of out-liers.
# * Sex has week coorelation and doesn't seem to effect insurance charges
# * bmi needs to be investigated more
# * number of childern does effect it seems more the childern overall charges are low but observation needs to be further verified
# * smoking directly effects overall charges. A smoker is charged more.
# * region doesn't seem to have any effect on overall charges

# **One feature Linear Regression**
# Let's practice one feature linear regression first to see how they work:

# 1. Age and Cost

# In[ ]:


age_lm = linear_model.LinearRegression()
age_lm.fit(train_personal_data.as_matrix(['age']),train_personal_data.as_matrix(['charges']))

# Have a look at R sq to give an idea of the fit 
print('R sq: ',age_lm.score(train_personal_data.as_matrix(['age']),train_personal_data.as_matrix(['charges'])))


# The above result shows that this way the model won't work

# **Training with multiple features**  
# Now that we know that signle variables are not good enough. Let's try training with multiple variables at the same time. We can do this by dropping the charges column (we will be using it as our Y) and using the rest of the data as X

# In[ ]:


lm = linear_model.LinearRegression()
X = train_personal_data.drop('charges', axis = 1)
lm.fit (X,train_personal_data.charges)
print('R sq: ',lm.score(X,train_personal_data.charges))


# It looks like that we have got a better model with better R Sq.

# **Predicting Values**  
# Now let's predict using the model we just created

# In[ ]:


X_test = test_personal_data.drop('charges', axis=1)
Y_Predicted = lm.predict(X_test)
print (r2_score(test_personal_data.charges,Y_Predicted))
sns.regplot(test_personal_data.charges, Y_Predicted)


# the above graphs shows that our predictions have been okay. There were some outliers which were far off from the our linear regression line. 
