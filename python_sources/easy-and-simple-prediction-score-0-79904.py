#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# This simple kernel is a useful for the new users and beginners in Kaggle. It is very easy to understand how to start your journey in kaggle.
# I meant to make it very easy to encourage you and  make it clear without any complications. 
# I will try to optimize this kernal from time to time but will keep it simple as it is.
# 
# 

# In[ ]:



import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Read the Train file.

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()


# Simple query to find out the number of male and female in the train file.

# In[ ]:


train[['Sex','Survived' ]].groupby('Sex').count()


# Simple visualization to know who survived between male and female.

# In[ ]:


sns.countplot(x = 'Survived' ,data =train, hue = 'Sex')
plt.show()


# Other simple histogram to find out the distribution of passengers depend on their age.

# In[ ]:


plt.hist(train.Age)


# Also one more plot by Boxplot to find out the survivors ( male and female )

# In[ ]:


sns.catplot(x= 'Sex' , y ='Age', data = train, kind= 'box', col = 'Survived')


# Get more information about the train file to see the missing value and columns type.

# In[ ]:


train.info()


# Fill the missing values in the Age column by the mean value.

# In[ ]:


train.Age = train.Age.fillna(train.Age.mean())


# In[ ]:


train.head()


# Now read the test file.

# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


# Get more information about the test columns.

# In[ ]:


test.info()


# Fill the missing values in Age and Fare columns.

# In[ ]:


test.Age = test.Age.fillna(test.Age.mean())
test.Fare = test.Fare.fillna(test.Fare.mean())


# Check if the missing values were filled.

# In[ ]:


test.info()


# Before starting the training the the data, I should convert the string to integer.

# In[ ]:


train.Sex = train.Sex.map({'male':1, 'female': 2})
train.head()


# In[ ]:


train.Embarked = train.Embarked.map({'C':1, 'S':2, 'Q':3})
train.head()


# In[ ]:


train.Embarked = train.Embarked.fillna(train.Embarked.mean())
train.Embarked = train.Embarked.astype('int')
train.head()


# In[ ]:


test.Sex = test.Sex.map({'male':1, 'female':2})
test.Embarked = test.Embarked.map({'C':1, 'S':2, 'Q':3})


# Again have to check.

# In[ ]:


test.head()


# Assign the Survived column to y.

# In[ ]:


y = train.Survived


# Specify the columns to be included in the training process.

# In[ ]:


features = ['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare'] # 6 features


# Assign the featured columns of train file to X.

# In[ ]:


X = pd.get_dummies(train[features])


# In[ ]:


X.head()


# Assign the featured columns of the test file to X_test

# In[ ]:


X_test = pd.get_dummies(test[features])
X_test.head()


# # Training Data:
# * Define the model.

# In[ ]:


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

Training the train data.
# In[ ]:


model.fit(X,y)


# # Prediction:
# * Predict the X_test.

# In[ ]:


prediction = model.predict(X_test)
prediction


# # Exporting submission:
# Save your prediction to the sumission file.

# In[ ]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

