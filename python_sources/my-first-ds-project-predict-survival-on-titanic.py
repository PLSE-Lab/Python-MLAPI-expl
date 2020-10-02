#!/usr/bin/env python
# coding: utf-8

# Update Log:
# * 2019-02-12  Created kernel. Submitted first prediction with accuracy of 0.75598
# * 2019-02-13  Split ages into groups in hopes of having age correlate more with survival.
# * 2019-02-13  Removed split ages due to lower accuracy by ~5% and began including SibSp, Parch, and Fare in regression independent variables. Submitted with a lower accuracy of 0.74162
# * 2019-02-14  Tried replacing null age values with the median rather than settling for a poor linear regression. This resulted in an even lower accuracy of 0.73684
# * 2019-02-24  Onehot encode the categorical data as well as scaling our numerical data to create better training data for our logistic regression.
# 
# 
# This is my first data science project. I hope to learn a lot from this experience and improve in the future. I will try to do things in a simple way and expand by using more variables with future updates.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First I will import the training and test data about the passengers of the Titanic.

# In[ ]:


raw_train = pd.read_csv('../input/train.csv')
raw_test = pd.read_csv('../input/test.csv')
train = raw_train.copy()
test = raw_test.copy()
train.head()


# We see 12 columns in our data set:
# (I will add the explanations given with the data set)
# 1. **PassengerId** - Unique id used to identify each passenger in our data
# 1.  **Survived** - Indicates if passenger survived (1 = Yes, 0 = No)
# 1.  **Pclass** - Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# 1.  **Name** - Name of the passenger
# 1.  **Sex** - Sex of passenger
# 1.  **Age** - Age of passenger in years
# 1.  **SibSp** - Number of the passenger's siblings and spouses on the Titanic
# 1.  **Parch** - Number of the passenger's parents and children on the Titanic
# 1.  **Ticket** - Ticket number
# 1.  **Fare** - Amount paid for ticket
# 1. **Cabin** - Cabin number
# 1. **Embarked** - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southhampton)
# 
# I will group the columns as Numerical or Categorical.
# 
# **Numerical:** PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
# 
# **Categorical:** Name, Sex, Ticket, Cabin, Embarked
# 
# By first impression, I think it will be easiest to work without Name, Ticket Number, and Cabin Number as they are more complicated to pull meaning from. Later on, I can try to implement them to further increase accuracy.
# 
# First let's change our Sex and Embarked columns to integers so we can gather statistical data on it. I will set Male = 0 and Female = 1 as well as setting S = 0, C = 1, and Q = 2. These integers will have no meaning other than differentiating the classifcations for a given feature. I implement changes to both the training set and the test set because the data being tested needs to be in the same format as the data that was used to train.

# In[ ]:


train['Sex'] = train['Sex'].map({'male':0,'female':1})
test['Sex'] = test['Sex'].map({'male':0,'female':1})

train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})
test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


train.describe()


# By using the describe method on our data, we can see some possibly useful measures. It looks like the average age of our passengers is rounded up to 30 years old, with a minimum age of 0.42 years old and a maximum of age of 88 years old. There is also more men than women on the Titanic. Additionally, not many people have family aboard.

# In[ ]:


print(train.info(), test.info())


# Using the info method, I can see how much data is missing from our dataframe. The only columns missing data in both sets of data are Age and Embarked. Train is missing Cabin values and Test is missing a Fare value.I will just set the NaN values to the most common port of Embarked, Age, and Fare. I will ignore Cabin for now. 

# In[ ]:


train.loc[pd.isnull(train['Embarked']),'Embarked'] = train['Embarked'].median()
train.loc[pd.isnull(train['Age']),'Age'] = train['Age'].median()

test.loc[pd.isnull(test['Fare']), ['Fare']] = test['Fare'].median()
test.loc[pd.isnull(test['Age']), ['Age']] = train['Age'].median()


# Next I will scale the numerical data so that they are equally weighted when being used in training. I skip PassengerId because it is just used for identification and has no actual value.

# In[ ]:


train['Fare'] = preprocessing.scale(train['Fare'])
train['SibSp'] = preprocessing.scale(train['SibSp'])
train['Parch'] = preprocessing.scale(train['Parch'])
train['Age'] = preprocessing.scale(train['Age'])

test['Fare'] = preprocessing.scale(test['Fare'])
test['SibSp'] = preprocessing.scale(test['SibSp'])
test['Parch'] = preprocessing.scale(test['Parch'])
test['Age'] = preprocessing.scale(test['Age'])


# In order to better interpet the classification data without getting confused by the size of the integers that represent them, I will onehot encode the data.

# In[ ]:


# get dummy variables
pclass_dum = pd.get_dummies(train['Pclass'])
sex_dum = pd.get_dummies(train['Sex'])
embarked_dum = pd.get_dummies(train['Embarked'])

# rename the columns
pclass_dum.columns = ('Pclass 1', 'Pclass 2', 'Pclass 3')
sex_dum.columns = ('Male', 'Female')
embarked_dum.columns = ('S', 'C', 'Q')

# drop the original columns from training data
train = train.drop(columns=['Pclass','Sex','Embarked','Name','Ticket','Cabin'])

# add the new dummy variables into our training data
train = train.join(pclass_dum)
train = train.join(sex_dum)
train = train.join(embarked_dum)

# get dummy variables
pclass_dum = pd.get_dummies(test['Pclass'])
sex_dum = pd.get_dummies(test['Sex'])
embarked_dum = pd.get_dummies(test['Embarked'])

# rename the columns
pclass_dum.columns = ('Pclass 1', 'Pclass 2', 'Pclass 3')
sex_dum.columns = ('Male', 'Female')
embarked_dum.columns = ('S', 'C', 'Q')

# drop the original columns from testing data
test = test.drop(columns=['Pclass','Sex','Embarked','Name','Ticket','Cabin'])

# add the new dummy variables into our testing data
test = test.join(pclass_dum)
test = test.join(sex_dum)
test = test.join(embarked_dum)

train.head()


# Looking at the training data above, I have turned the Pclass column into three columns called Pclass 1, Pclass 2, and Pclass 3. A similar change has been made for the Sex column and the Embarked column as well.

# I am ready to run a logistic regression using our training data to build a model that can predict if a passenger survived. I removed PassengerId when training since it has no value in making predictions.

# In[ ]:


x_train = train.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13]]
y_train = train.iloc[:, [1]]

log_reg = LogisticRegression().fit(x_train, y_train)


# After training, I run the test data through the logistic regression model to find a prediction.

# In[ ]:


x_test = test.drop(columns=['PassengerId'])

predict_test =  log_reg.predict(x_test)


# All that is left to do is format my prediction into a dataframe that is ready to be submitted to the kaggle competition.

# In[ ]:


prediction = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": predict_test
})

prediction = prediction.astype(int)
prediction


# In[ ]:


prediction.to_csv(path_or_buf='prediction.csv',index=False)

