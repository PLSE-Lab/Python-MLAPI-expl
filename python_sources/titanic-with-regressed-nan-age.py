# In[1]:

import pandas as pd
import numpy as np


# In[2]:

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# In[3]:

test = test.set_index("PassengerId")
train = train.set_index("PassengerId")


# In[4]:

train = train.drop(["Ticket","Cabin","Name"],axis=1)
test = test.drop(["Ticket","Cabin","Name"],axis=1)


# In[5]:

train_Survived = train["Survived"]
train = train.drop("Survived",axis=1)


# In[6]:

SexD = {"male":0,"female":1}
EmbarkedD = {"S":0,"C":1,"Q":2}
train.Sex = train.Sex.apply(SexD.get).astype("category")
test.Sex = test.Sex.apply(SexD.get).astype("category")
train.Embarked = train.Embarked.apply(EmbarkedD.get).astype("category")
test.Embarked = test.Embarked.apply(EmbarkedD.get).astype("category")
test.Pclass = test.Pclass.astype("category")
train.Pclass = train.Pclass.astype("category")


# In[7]:

train.Embarked = train.Embarked.fillna(0)
test.Fare = test.Fare.fillna(test.Fare.mean())


# In[8]:

train_nan_age = train[train.Age.apply(np.isnan)]
train_age = train[~train.Age.apply(np.isnan)]
test_nan_age = test[test.Age.apply(np.isnan)]
test_age = test[~test.Age.apply(np.isnan)]


# In[9]:

from sklearn.ensemble import GradientBoostingRegressor


# In[10]:

est = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1,max_depth=1, loss='huber')


# In[11]:

est = est.fit(train_age.drop("Age",axis=1),train_age["Age"])


# In[12]:

train_nan_age["Age"] = est.predict(train_nan_age.drop("Age",axis=1))


# In[13]:

test_nan_age["Age"] = est.predict(test_nan_age.drop("Age",axis=1))


# In[14]:

train = train_age.append(train_nan_age)
test = test_age.append(test_nan_age)


# In[15]:

from sklearn.ensemble import GradientBoostingClassifier


# In[16]:

clf = GradientBoostingClassifier()
clf = clf.fit(train,train_Survived)
test["Survived"]=clf.predict(test)


# In[19]:

test.drop(["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"],axis=1).to_csv("titanic_rf.csv",index=True)