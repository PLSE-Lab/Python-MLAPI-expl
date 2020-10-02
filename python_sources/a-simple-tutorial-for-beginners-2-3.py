#!/usr/bin/env python
# coding: utf-8

# # A simple tutorial for Beginners
# 
# This notebook is a version that adds category data to the previous version.  
# Please refer to the [previous notebook](https://www.kaggle.com/hs1214lee/a-simple-tutorial-for-beginners-1-3).  
# 

# # Read data
# Same as previous version.

# In[ ]:


import os
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print('Shape of train dataset: {}'.format(train.shape))
print('Shape of test dataset: {}'.format(test.shape))


# # Process the Data
# ## Dropping the ambiguous columns
# 
# Previous notebooks only used numeric data.
# ```python
# train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)
# test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)
# ```
# 'Sex' and 'Embarked' data will also be used in this version.

# In[ ]:


train.drop(['Name', 'Ticket', 'Cabin'],axis = 1,inplace = True)
train.head()


# In[ ]:


test.drop(['Name', 'Ticket', 'Cabin'],axis = 1,inplace = True)
test.head()


# ## Check for missing values
# The second step is to check for missing data and fill it if it exists.  

# In[ ]:


print(train.isnull().values.any())
train.isnull().sum()


# There are missing values in the 'Age', 'Embarked' features.  
# 'Age' is filled in the same way as last time.(use mode value)

# In[ ]:


print(train['Age'].value_counts())
train['Age'] = train['Age'].fillna(24)
train.isnull().sum()


# Do the same for 'Embarked' data.

# In[ ]:


print(train['Embarked'].value_counts())
train['Embarked'] = train['Embarked'].fillna('S')
train.isnull().sum()


# The test data is the same as the previous version.

# In[ ]:


print(test['Age'].value_counts())
test['Age'] = test['Age'].fillna(24)
print(test['Fare'].value_counts())
test['Fare'] = test['Fare'].fillna(7.75)
test.isnull().sum()


# ## Categorical data to numerical data
# Categorical data cannot be used for training.  
# So we use it by changing to a numeric type.

# In[ ]:


# Categorical data to numerical data
train.loc[train['Sex'] == 'male', 'Sex'] =  1
train.loc[train['Sex'] == 'female', 'Sex'] = 0
train.loc[train['Embarked'] == 'S', 'Embarked'] =  0
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 1
train.loc[train['Embarked'] == 'C', 'Embarked'] =  2
print(train.head())

test.loc[test['Sex'] == 'male', 'Sex'] =  1
test.loc[test['Sex'] == 'female', 'Sex'] = 0
test.loc[test['Embarked'] == 'S', 'Embarked'] =  0
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 1
test.loc[test['Embarked'] == 'C', 'Embarked'] =  2
print(test.head())


# # Training
# Same as previous version.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train_y = train['Survived']
train_x = train.drop('Survived', axis=1)
model = LogisticRegression()
model.fit(train_x, train_y)
pred = model.predict(train_x)
metrics.accuracy_score(pred, train_y)


# The resulting score is 0.8002244668911336.  
# Performance improved by 13% compared to the previous results(0.7025813692480359).  
# Now let's predict the test data and save it as a csv file.

# In[ ]:


import time

timestamp = int(round(time.time() * 1000))

pred = model.predict(test)
output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})
output.to_csv("submission_" + str(timestamp) + ".csv",index = False)


# Submit the results.  
# The public score(for test dataset) is 0.7536.  
# The test data results also improved by 14% compared to the previous one(0.66028).  
# You can improve your results with just this simple task.  
#   
# There are still many factors that can be improved.  
# Improve your model by referring to other notebooks.
# 
# Please refer to my [next notebooks](https://www.kaggle.com/hs1214lee/a-simple-tutorial-for-beginners-3-3)!
