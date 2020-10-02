#!/usr/bin/env python
# coding: utf-8

# # A simple tutorial for Beginners
# 
# This notebook used the simplest methods to help beginners understand.

# In order to successfully submit the results to Kaggle, the following code must be executed.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Read data
# First, import the data set using the pandas library.  
# With pandas we can read data directly from csv files.

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print('Shape of train dataset: {}'.format(train.shape))
print('Shape of test dataset: {}'.format(test.shape))


# Let's look at the data set.  
# The first step in the ML project is to understand the datas.  
# Some samples of the data set can be viewed in the following way.

# In[ ]:


train.head()


# In[ ]:


test.head()


# Each dataset has 12(train) and 11(test) features.  
# 'Survived' is the feature what we have to predict.  
# Other features consist of numeric or categorical data.

# # Process the Data
# Preprocessing of data is a very important part.  
# Good data preprocessing gives us good results.  
# There are many pre-processing methods, but this notebook will use the most basic methods for understanding.

# ## Dropping the categorical columns
# 
# The first step is to select the features to use.  
# In this notebook, we will only use numeric data.

# In[ ]:


train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)
train.head()


# In[ ]:


test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)
test.head()


# ## Check for missing values
# The second step is to check for missing data and fill it if it exists.  

# In[ ]:


train.isnull().values.any()


# In[ ]:


train.isnull().sum()


# There are missing values in the 'Age' feature.  
# I will fill the missing values with mode(value that appears most often).

# In[ ]:


train['Age'].value_counts()


# In[ ]:


train['Age'] = train['Age'].fillna(24)
train.isnull().sum()


# Do the same for test data.

# In[ ]:


test.isnull().sum()


# In[ ]:


test['Age'].value_counts()


# In[ ]:


test['Fare'].value_counts()


# In[ ]:


test['Age'] = test['Age'].fillna(24)
test['Fare'] = test['Fare'].fillna(7.75)
test.isnull().sum()


# # Training
# Next, we need to extract the target variable from the train data.

# In[ ]:


train_y = train['Survived']
train_x = train.drop('Survived', axis=1)
train_x.head()


# In[ ]:


train_y.head()


# The Titanic problem is a typical binary classification problem that predicts whether it is 0(dead) or 1(survived).  
# We will use LogisticRegression to train.  
# LogisticRegression is the simplest model for solving binary classification problem.  
# LogisticRegression is easy to use with the sklearn library.

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x, train_y)


# In[ ]:


from sklearn import metrics
pred = model.predict(train_x)
metrics.accuracy_score(pred, train_y)


# The resulting score is 0.7025813692480359.  
# Now let's predict the test data and save it as a csv file.  

# In[ ]:


import time

timestamp = int(round(time.time() * 1000))

pred = model.predict(test)
output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})
output.to_csv("submission_" + str(timestamp) + ".csv",index = False)


# Submit the results.  
# The public score(for test dataset) is 0.66028.  
# 
# Congratulations :)  
# You created your first machine learning project.  
# There are a lot of things to improve because we used the simplest methods.  
# Improve your model by referring to other notebooks.
# 
# Please refer to my [next notebooks](https://www.kaggle.com/hs1214lee/a-simple-tutorial-for-beginners-2-3)!
# 
