#!/usr/bin/env python
# coding: utf-8

# **1 Gather the Data**
# 
# Add the dataset of Titanic into the programe

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


# **2 Data Preparation**
# 
# * Split our data into test training data and test data.
# * Split both training data and test data into features data and label data
# * Handle with the missing data(Handle with the problems caused by NaN)

# **2.1 Prepare the traing data**

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train.head()


# In my opinion, 'Name' and 'Cabin' and 'Embarked' have nothing to do with the possibility of survival of the passengers. So I set up 'df1' with the rest of the features of the passengers.

# In[ ]:


df1 = train[['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]


# In[ ]:


df1.head()


# In[ ]:


df1.shape


# Count the number of 'NaN'

# In[ ]:


p = df1.isnull().sum().sum()
print(p)


# In my opinion, filling the 'NaN' with other data which are fake or calculated artificially will decrease the accuracy of our prediction model in this case. So I choose to delete the rows which contain the missing data.

# In[ ]:


df1_dl=df1.dropna(axis = 0)


# Make sure there is no missing data.

# In[ ]:


q = df1_dl.isnull().sum().sum()
print(q)


# Try to use '1' or '0' to describe 'Sex' instead of strings.
# * 1 -> male
# * 0 -> female

# * Set up a new column named 'Sex_value' 
# * Set the 'Sex_value' of every passenger according to 'Sex'

# In[ ]:


df1_dl['Sex_value'] = df1_dl['Sex']


# In[ ]:


df1_dl


# In[ ]:


df1_dl.loc[df1_dl['Sex'] == 'male','Sex_value'] = 1
df1_dl.loc[df1_dl['Sex'] == 'female','Sex_value'] = 0


# In[ ]:


df1_dl


# Set up 'df1_train' without column 'Sex'

# In[ ]:


df1_train = df1_dl[['Survived','Pclass','Age','SibSp','Parch','Fare','Sex_value']]


# In[ ]:


df1_train


# Split features and labels

# In[ ]:


x_train = df1_train.iloc[:,1:]
y_train = df1_train.iloc[:,0]


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# **2.2 Prepare the test data**
# * Almost the same as preparing the training data

# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test


# In[ ]:


df2 = test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]
G_S = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


df2


# In[ ]:


G_S


# Add the column 'Survived' of dataframe 'Survived' into dataframe 'df2' in order to delete the rows which contain missing data conveniently.

# In[ ]:


df2['Survived'] = G_S['Survived']


# In[ ]:


df2


# In[ ]:


m = df2.isnull().sum().sum()
print(m)


# In[ ]:


df2_dl=df2.dropna(axis = 0)


# In[ ]:


n = df2_dl.isnull().sum().sum()
print(n)


# In[ ]:


df2_dl


# In[ ]:


df2_dl['Sex_value'] = df2_dl['Sex']
df2_dl.loc[df2_dl['Sex'] == 'male','Sex_value'] = 1
df2_dl.loc[df2_dl['Sex'] == 'female','Sex_value'] = 0


# In[ ]:


df2_dl


# In[ ]:


df2_test = df2_dl[['Pclass','Age','SibSp','Parch','Fare','Sex_value','Survived']]


# In[ ]:


df2_test


# In[ ]:


x_test = df2_test.iloc[:,0:6]
y_test = df2_test.iloc[:,6]


# In[ ]:


x_test


# In[ ]:


y_test


# Check the shape of dataframe 

# In[ ]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# Make sure there is no NaN in our data frame

# In[ ]:


x1 = x_train.isnull().sum().sum()
y1 = y_train.isnull().sum().sum()
x2 = x_test.isnull().sum().sum()
y2 = y_test.isnull().sum().sum()
print(x1,y1,x2,y2)


# **3 Train and Evaluate the Model**
# 
# Let's see how LogisticRegression perform in this case 

# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)


# **4 Predict the Particular Data**

# In[ ]:


clf.predict(x_test[301:306])


# In[ ]:


y_test[301:306]


# **5 Other Models**
# 
# * 5.1 SVM
# * 5.2 RandomForestClassifier

# >>5.1 SVM

# In[ ]:


from sklearn import svm

clf = svm.SVC()
clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)


# >>5.2 RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)

