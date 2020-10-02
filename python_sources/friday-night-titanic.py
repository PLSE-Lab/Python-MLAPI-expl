#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import preprocessing
pd.set_option('display.max_columns', 200)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


gender_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


gender_sub.head()


# In[ ]:


test.head()


# In[ ]:


train.head()


# **Pclass**

# In[ ]:


train_pclass = pd.get_dummies(train.Pclass)
test_pclass = pd.get_dummies(test.Pclass)
train_pclass.head()


# **Sex**

# In[ ]:


train_sex = pd.get_dummies(train.Sex)
test_sex = pd.get_dummies(test.Sex)
test_sex.head()


# **Age**

# In[ ]:


allages = pd.concat((train.Age,test.Age),axis=0)
minage = min(allages)
maxage = max(allages)

train_age = (train.Age - minage)/(maxage-minage)
test_age = (test.Age - minage)/(maxage-minage)


# In[ ]:


train_age.head()


# In[ ]:


test_age.head()


# **SibSP & Parch**

# In[ ]:


train_sibsp = pd.get_dummies(train.SibSp)
test_sibsp = pd.get_dummies(test.SibSp)
train_parch = pd.get_dummies(train.Parch)
test_parch = pd.get_dummies(test.Parch)


# In[ ]:


train_sibsp.head()


# In[ ]:


test_sibsp.head()


# In[ ]:


train_parch.head()


# In[ ]:


test_parch.head()


# test_parch has an extra column for features not found in the training dataset, so we will drop that column.

# In[ ]:


#train_parch.head()
#test_parch.head()
test_parch = test_parch.drop(columns=9)
test_parch.head()


# **Ticket**

# In[ ]:


train_ticket = train.Ticket
test_ticket = test.Ticket
tridx = train_ticket[train_ticket.str.contains('[A-Za-z]')].index
teidx = test_ticket[test_ticket.str.contains('[A-Za-z]')].index

trt_nums = pd.DataFrame(np.zeros(train_ticket.shape))
tet_nums = pd.DataFrame(np.zeros(test_ticket.shape))
tr_above = pd.DataFrame(np.zeros(train_ticket.shape))
te_above = pd.DataFrame(np.zeros(test_ticket.shape))

trt_nums.iloc[tridx]=1
tet_nums.iloc[teidx]=1
trt_nums = trt_nums.astype(int)
tet_nums = tet_nums.astype(int)

test_ticket = test_ticket.str.extract('(\d+)', expand=False)
train_ticket = train_ticket.str.extract('(\d+)', expand=False)
test_ticket = test_ticket.fillna(0).astype(int)
train_ticket = train_ticket.fillna(0).astype(int)
temp_train = train_ticket>1000000
temp_test = test_ticket>1000000
print(temp_train[temp_train==True].size)
#print(temp)
train_ticket[temp_train==True]=-100
test_ticket[temp_test==True]= -100
train_ticket[tridx]=-50
test_ticket[teidx]=-50

minx = min(train_ticket)
maxx = max(train_ticket)

train_ticket = (train_ticket - minx)/(maxx-minx)
test_ticket = (test_ticket - minx)/(maxx-minx)

print(train_ticket[tridx])

tr_above[temp_train==True]=1
te_above[temp_test==True]=1


#test_ticket.loc[,:]=0
#train_ticket.loc[train_ticket>1000000,:]=0
plt.plot(test_ticket)
plt.plot(train_ticket)


# Fare

# In[ ]:


allfare = pd.concat((train.Fare,test.Fare),axis=0)
minfare = min(allfare)
maxfare = max(allfare)

train_fare = (train.Fare - minfare)/(maxfare - minfare)
test_fare = (test.Fare - minfare)/(maxfare - minfare)


# In[ ]:


train_fare.head()


# In[ ]:


print(train.Ticket)


# Cabin

# In[ ]:


temp = train['Cabin'].astype(str).str[0]
train_cabin = pd.get_dummies(temp)
temp = test['Cabin'].astype(str).str[0]
test_cabin = pd.get_dummies(temp)
train_cabin = train_cabin.drop(columns=['T'])
print(train_cabin.head())


# In[ ]:


print(test_cabin.head())


# In[ ]:


train['train']=1
test['train']=0
combined = pd.concat([train.train,test.train])
combined.head()
combined_cabin = pd.concat([train.Cabin,test.Cabin])
cabins = pd.get_dummies(combined_cabin)
train_cabins = cabins.loc[combined[:]==1,:]
test_cabins = cabins.loc[combined[:]==0,:]


# In[ ]:


train_cabin.head()


# In[ ]:


test_cabin.head()


# Embarked

# In[ ]:


train_embarked = pd.get_dummies(train.Embarked)
test_embarked = pd.get_dummies(test.Embarked)


# In[ ]:


train_embarked.head()


# In[ ]:


test_embarked.head()


# In[ ]:


train_all = pd.concat((train_embarked,train_fare,train_parch,train_sibsp,train_age,train_sex,train_pclass,train_cabin,trt_nums,tr_above),axis=1)
test_all = pd.concat((test_embarked,test_fare,test_parch,test_sibsp,test_age,test_sex,test_pclass,test_cabin,tet_nums,te_above),axis=1)


# In[ ]:


train_all.head()


# In[ ]:


param = {'max_depth':2, 'eta':2, 'objective':'binary:hinge'}
model = XGBClassifier()
y_train = train.Survived
model.fit(train_all.to_numpy(),y_train)
print(model)


# In[ ]:


print(train_all.shape)
print(test_all.shape)


# In[ ]:


y_pred = model.predict(test_all.to_numpy())


# In[ ]:


print(y_pred.shape)
Survived = pd.DataFrame(y_pred)
predictions = pd.concat((test.PassengerId,Survived),axis=1)
predictions = predictions.rename(columns={0:'Survived'})
print(predictions)


# In[ ]:


predictions.to_csv('FridayTitanic.csv',index=False)

