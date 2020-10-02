#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# load training data
ds = pd.read_csv("../input/train.csv")


# In[ ]:


ds.info()


# In[ ]:


ds.describe(include='all')


# In[ ]:


ds.Survived.hist()
sum(ds.Survived)


# In[ ]:


y = ds.Survived

my_vars = ['Pclass','Age','Parch','SibSp','Fare','Sex','Embarked']

x = ds[my_vars]
x.info()
x.describe(include='all')


# In[ ]:


# Imputation
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
xx = my_imputer.fit_transform(x[['Pclass','Age','Parch','SibSp','Fare']])
x2 = pd.DataFrame(xx, columns=['Pclass','Age','Parch','SibSp','Fare'])
x2.describe()

# add categorical features
x2['Sex'] = x['Sex']
x2['Embarked'] = x['Embarked']

# x2['Cabin'] = x['Cabin']
# x2['Ticket'] = x['Ticket']

x2.info()


# In[ ]:


# Random Forest Fit

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=50, max_features=4, random_state=1234)

# one hot encoding for categorical features
x3 = pd.get_dummies(x2)

clf.fit(x3,y)


# In[ ]:


# predict on training set
yhat = clf.predict(x3)

# check prediction on training set
plt.hist(yhat)
sum(yhat)


# In[ ]:


accuracy_score(y,yhat)


# In[ ]:


check_me = pd.DataFrame(y)
check_me['Pred'] = yhat
check_me['Diff'] = yhat - y
check_me['Diff_Factor'] = check_me['Diff'].astype(object)

print(check_me)

check_me.describe(include='all')


# In[ ]:


# show variable importance
varimp = clf.feature_importances_
print(varimp)
plt.bar([1,2,3,4,5,6,7,8,9,10],varimp)
plt.xticks([1,2,3,4,5,6,7,8,9,10],('Pclass','Age','Parch','SibSp','Fare','SexF','SexM','EmbC','EmbQ','EmbS'))


# In[ ]:


ds_test = pd.read_csv("../input/test.csv")
ds_test.describe(include='all')


# In[ ]:


test_x = ds_test[my_vars]

# Imputation
test_xx = my_imputer.fit_transform(test_x[['Pclass','Age','Parch','SibSp','Fare']])
test_x2 = pd.DataFrame(test_xx, columns=['Pclass','Age','Parch','SibSp','Fare'])
test_x2.describe()

# add categorical features
test_x2['Sex'] = x['Sex']
test_x2['Embarked'] = x['Embarked']

# test_x2['Cabin'] = x['Cabin']
# test_x2['Ticket'] = x['Ticket']

test_x2.info()


# In[ ]:


# one hot encoding
test_x3 = pd.get_dummies(test_x2)

# call and evaluate prediction
test_yhat = clf.predict(test_x3)
plt.hist(test_yhat)
sum(test_yhat)


# In[ ]:


result = pd.DataFrame(ds_test.PassengerId)
result['Survived'] = test_yhat
print(result)


# In[ ]:




