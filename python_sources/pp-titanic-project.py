#!/usr/bin/env python
# coding: utf-8

# **Initially, we will import all our data:**

# In[ ]:


import numpy as np
import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
train.head()


# We've got 12 columns, namely **PassengerId, Survived** (where Yes = 1, No = 0), **pclass** (1 = 1st, 2 = 2nd,3 = 3rd), **sex, name, age
# sibsp** (Number of siblings that were with them on the ship), **parch** (Number of parents that were with them on the ship), **tickets, fare, cabin, embarked** (C,S,Q being locations). Among these, the Age, Fare, SibSp and Parch are numeric, Sex, Embarked, Survived and Pclass are alphabetic, Ticket and Cabin are alphanumeric. Name is text feature.
# 

# In[ ]:


train.shape


# We can see that we have around 891 rows and 12 columns. Some of our rows are filled with an NaN, which is not a number and perhaps some undefined type. Let's see which columns have such values along with how many of in each of them.

# In[ ]:


train.isnull().sum()


# The Age and Cabin columns are short on a lot of values, which is undesireable for our work as it can lead to a poor prediction. To correct this, we must fill these in. But first, let's take a look at test data as well.

# In[ ]:


test.isnull().sum()


# An observation can be made here that the cabin column is missing out on a lot of rows here as well, so we might as well drop the column from both the datasets. For age and fare, we will deal with those by using mean of the values.

# In[ ]:


train.drop(['Cabin'],axis=1,inplace=True)
test.drop(['Cabin'],axis=1,inplace=True)


# The name and ticket columns do not contribute anything, so we might as well delete them:

# In[ ]:


train.drop(['Ticket'],axis=1,inplace=True)
test.drop(['Ticket'],axis=1,inplace=True)


# In[ ]:


train.drop(['Name'],axis=1,inplace=True)
test.drop(['Name'],axis=1,inplace=True)


# To fill in for NaN columns of age, we're calculating the mean age for train and test and filling it in:

# In[ ]:


print("Mean age:", np.mean(train['Age'].dropna()))
print("Mean age:", np.mean(test['Age'].dropna()))


# Rounding off 29.6 to 30 and 30.27 to 30 respectively, and replacing NaN with 30 in the age feature column for train and test. We'll also get an idea of the repeated values in the embarked column. This will rid us of all the null values in the train set.

# In[ ]:


from collections import Counter
from sklearn import preprocessing

train['Age'].fillna(30, inplace=True)
test['Age'].fillna(30, inplace=True)

ctr = Counter(train['Embarked'])
print("Embarked feature's most common 2 data points:", ctr.most_common(2))
train['Embarked'].fillna('S', inplace=True)
test.isnull().sum()


# In our test data, we still have to deal with the fare column. calculating the mean fare for test and filling it in:

# In[ ]:


test['Fare'].fillna(np.mean(test['Fare']), inplace=True)
test['Age'] = test.Age.astype(int)

test.isnull().sum()
test.info()


# Some of the columns such as sex and embarked are still not in int form, so we shall use label encoders that will to convert them into int: 

# In[ ]:


import copy

encoder = preprocessing.LabelEncoder()

embarkedEncoder = copy.copy(encoder.fit(train['Embarked']))
train['Embarked'] = embarkedEncoder.transform(train['Embarked'])

sexEncoder = copy.copy(encoder.fit(train['Sex']))
train['Sex'] = sexEncoder.transform(train['Sex'])

train['Fare'] = train['Fare'].astype(int)
train.loc[train.Fare<=7.91,'Fare']=0
train.loc[(train.Fare>7.91) &(train.Fare<=14.454),'Fare']=1
train.loc[(train.Fare>14.454)&(train.Fare<=31),'Fare']=2
train.loc[(train.Fare>31),'Fare']=3

train['Age']=train['Age'].astype(int)
train.loc[ train['Age'] <= 16, 'Age']= 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[train['Age'] > 64, 'Age'] = 4


# Repeating a similar task for test data:

# In[ ]:


import copy

encoder = preprocessing.LabelEncoder()

embarkedEncoder = copy.copy(encoder.fit(test['Embarked']))
test['Embarked'] = embarkedEncoder.transform(test['Embarked'])

sexEncoder = copy.copy(encoder.fit(test['Sex']))
test['Sex'] = sexEncoder.transform(test['Sex'])


# In[ ]:


train.head()

Our train data looks fairly good. Let's look at test data:
# In[ ]:


test.head()


# Fine, let's move on to training our model. Using the scikit train_test_split class, we'll split the training and testing data, and as we pass 0.3 as the value for test_size, the dataset will be split 30% as the test dataset. The train_size automatically gets assigned the remaining. Some of the values such as Fare, Age are float64, so we need to convert them to int in order to run them in prediction function.

# In[ ]:


X=train.drop('Survived',axis=1)
y=train['Survived'].astype(int)

train['Fare'] = train['Fare'].astype(int)
train['Age'] = train['Age'].astype(int)
test['Fare'] = test['Fare'].astype(int)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer


from sklearn.model_selection import train_test_split

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
acc_dict = {}

for train_index, test_index in sss.split(X, y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    

classifier=SVC()
xtrain=train.iloc[:,1:]
ytrain=train.iloc[:,:1]
ytrain=ytrain.values.ravel()
classifier.fit(xtrain,ytrain)

testIm=Imputer(missing_values='NaN',strategy='most_frequent',axis=1)
Age1=testIm.fit_transform(test.Age.values.reshape(1,-1))
Fare2=testIm.fit_transform(test.Fare.values.reshape(1,-1))
test['Age']=Age.T
test['Fare']=Fare.T
test.set_index('PassengerId',inplace=True)

test['Fare'] = test['Fare'].astype(int)
test.loc[test.Fare<=7.91,'Fare']=0
test.loc[(test.Fare>7.91) &(test.Fare<=14.454),'Fare']=1
test.loc[(test.Fare>14.454)&(test.Fare<=31),'Fare']=2
test.loc[(test.Fare>31),'Fare']=3

test['Age']=test['Age'].astype(int)
test.loc[ test['Age'] <= 16, 'Age']= 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[test['Age'] > 64, 'Age'] = 4

Result=classifier.predict(test)
print(Result)
print(len(Result))


# In[ ]:




