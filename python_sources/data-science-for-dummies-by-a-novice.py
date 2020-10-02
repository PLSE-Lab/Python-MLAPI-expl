#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# *If you learning by failing then fastest way to learn is to fail often*
# 
# This kernel will help you start applying machine learning to datasets without any prior knowledge of how any algorithm works. My whole aim is to show that you can begin modelling data straight away without having to put much thought into it and treating the algorithms as black boxes.
# 
# Following will be the my approach
# 1.  Fill in the missing values (More often than not our data will contain missing values, we must chose between filling in the wholes or reject the data point entirely )
# 2. Convert the whole data into numerical form (Machine Learning models are based on mathematical computation, so we must convert all are fields to numerical form)
# 3. Chose an algorithm (here we use Random forest) and fit our data
# 4. Predict and Submit

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's load our data into dataframes

# In[ ]:


raw_data = pd.read_csv('../input/train.csv', low_memory=False)
raw_test_data = pd.read_csv('../input/test.csv', low_memory=False)


# We'll seperate our independent(features used to predict outcome) and dependent(outcome) variables from the training data.

# In[ ]:


train_data = raw_data.drop('Survived', axis = 1)
test_data = raw_test_data.copy()


# Now we will see what our training data looks like

# In[ ]:


raw_data.tail().T


# In[ ]:


raw_data.describe(include='all')


# Now onto **step (1)**,  we'll see which all features have null values in them and then devise a plan to elimanate them

# In[ ]:


raw_data.isna().sum()


# There are 3 different columns that we'll have to handle for missing values
# 1. Age
# 2. Embarked
# 3. Cabin
# 
# Let us start with age. 
# There are many ways to impute missing data. But let's just stick to replacing age with the **mean age** of the whole data.

# In[ ]:


train_data_age_mean = train_data.Age.mean()
train_data.Age.fillna(train_data_age_mean,inplace=True)
test_data.Age.fillna(train_data_age_mean, inplace=True)


# Note that, we will be applying the same transformations to the test set as we do on the train set. Now let's move onto 'Embarked'.

# In[ ]:


train_data.Embarked.value_counts()


# For embarked let's replace the missing value with the most common value of the lot, i.e., 'S'

# In[ ]:


train_data.Embarked.fillna('S', inplace=True)


# We'll be handling 'Cabin' differently. Let's say we'll segregate people on the basis of whether they had a cabin or not. After that we'll be dropping the 'Cabin' variable altogether.  

# In[ ]:


train_data['CabinBool'] = train_data.Cabin.isnull().astype('int64')
test_data['CabinBool'] = test_data.Cabin.isnull().astype('int64')

train_data.drop('Cabin', inplace=True, axis=1)
test_data.drop('Cabin', inplace=True, axis=1)


# Now that we have handled missing values, let us move onto **step 2**. We'll replace categorical variables with their coded representation.

# In[ ]:


sex_mapping = {'male':1, 'female':0}
train_data.Sex.replace(sex_mapping, inplace=True)
test_data.Sex.replace(sex_mapping, inplace=True)


# In[ ]:


train_data.Embarked.unique()


# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data.Embarked.replace(embarked_mapping, inplace=True)
test_data.Embarked.replace(embarked_mapping, inplace=True)


# In[ ]:


train_data.dtypes


# There are certain other columns that we'll be droping. Although, we can extract futher features from the data let us stick to making the simplest model with the most minimal changes

# In[ ]:


train_data.drop(labels = ['Name', 'PassengerId', 'Ticket'], axis=1,inplace=True)

test_data.drop(labels = ['Name', 'PassengerId', 'Ticket'], axis=1,inplace=True)


# Onto **strep 3**, let's train a Random Forest classifier over our dataset.

# In[ ]:


model = RandomForestRegressor(n_jobs=-1)


# In[ ]:


model.fit(train_data, raw_data.Survived)
model.score(train_data, raw_data.Survived)


# You can see that without applying much brain power we have gotten ourselves a decent model that has an r<sup>2</sup> score of **0.86**. Not bad ! Let's execute our final step and predict the results.

# In[ ]:


results = model.predict(test_data)


# Bummer !! We forgot to check for NA's in our test set. Let's handle this with our known friend **mean**

# In[ ]:


test_data.Fare.fillna(test_data.Fare.mean(), inplace=True)


# In[ ]:


results = model.predict(test_data)


# Let's put a threshold on the predicted probabilities of our result. Let's say if the probability of survival in greater than **0.5**, then we'll  predict the value as **1** 

# In[ ]:


results = [1 if x > 0.5 else 0 for x in results]


# In[ ]:


ids = raw_test_data.PassengerId
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': results })
output.to_csv('submission.csv', index=False)


# # Conclusion
# In this kernel we saw how easy it is to start building your model even without having any in depth knowledege of how an algorithim works. Now, getting your hands dirty is one thing but getting competent at data science is another. It requires a lot of practice and intution that will only come through practice. I will post another indepth kernel on the same dataset as I move onto become (Contributor/expert/master/grand master). Good luck !

# # References
# [Awesome kernel for beginners on Titanic dataset][1]
# 
# [1]: http://https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner/comments
# 
