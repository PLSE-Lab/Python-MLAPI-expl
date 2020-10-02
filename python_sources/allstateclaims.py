#!/usr/bin/env python
# coding: utf-8

# Allstate Claims analysis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt # for plotting
# Any results you write to the current directory are saved as output.


# In[ ]:


#read data from CSV file
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Some info about the data
#train.head()


# In[ ]:


#Data housekeeping
#Category variables 116
train_cat = train.ix[:,'cat1':'cat116']
#Continuous variable 14
train_cont = train.ix[:,'cont1':'cont14']
#train_cat_dummy = pd.get_dummies(train_cat)
#print(train_cat.head(2))
#print(train_cat_dummy.head(2))
#Number of training samples
print("number of traning samples : {}".format(train.shape[0]))
print("number of test samples: {}".format(test.shape[0]))
#count the number of unique values under the categorical variables
print("number of unique categorical values : {}".format(len(pd.unique(train_cat[train_cat.columns[1:]].values.ravel()))))
#Check if there are any null values
#print(train.isnull().values.any())
#print(test.isnull().values.any())


# In[ ]:


#correlation between continuous predictors and traget
train_corr_loss = train.corr()["loss"]
ax = train_corr_loss.iloc[1:-1].plot(kind='bar',title="continuous features correlation with target", figsize=(5,5), fontsize=12)
ax.set_ylabel("correlation value")


# As we can see cont2 has the higher correlation with target followed by cont8,cont3,cont11 and cont12.
# cont13 has the lowest correlation.In general, all these features have low correlation values to the target, < 0.5

# In[ ]:


#classifcation just considering the continous features(why?)
y_train1 = np.asarray(train['loss'])
X_test1 = test.ix[:,'cont1':'cont14']
lreg = LinearRegression()
lreg.fit(train_cont,y_train1)
y_pred = lreg.predict(X_test1)
print("Training set score: {:.2f}".format(lreg.score(train_cont, y_train1)))


# The fact that a lot of features are missing implies too simple model(under fit)

# In[ ]:


#Categorical features analysis
from sklearn.preprocessing import LabelEncoder
catFeatures = []
for colName in train_cat.columns:
    le = LabelEncoder()
    le.fit(train_cat[colName].unique())
    train_cat[colName] = le.transform(train_cat[colName])
train_cat.head()


# In[ ]:


catFeatures = []
test_cat = test.ix[:,'cat1':'cat101']
for colName in test_cat.columns:
    le = LabelEncoder()
    le.fit(test_cat[colName].unique())
    test_cat[colName] = le.transform(test_cat[colName])
test_cat.head()


# In[ ]:


test_cont = test.ix[:,'cont1':'cont14']
X_train = train_cat.ix[:,'cat1':'cat101'].join(train_cont)
X_test = test_cat.join(test_cont)
#verify shapes
#Categorical features analysis using  dummy variables
print("(train)(test):{}{}".format(X_train.shape,X_test.shape))

lreg.fit(X_train,y_train1)
y_pred = lreg.predict(X_test)
print("Linear: Training set score: {:.2f}".format(lreg.score(X_train, y_train1)))


# In[ ]:


#Categorical features analysis using  dummy variables
train_cat_dummy = pd.get_dummies(train.ix[:,'cat1':'cat101'])
test_cat_dummy = pd.get_dummies(test_cat)

print("new dummy DF shapes(train)(test):{}{}".format(train_cat_dummy.shape,test_cat_dummy.shape))


# In[ ]:


X_train = train_cat_dummy.join(train_cont)
X_test = test_cat_dummy.join(test_cont)
lreg.fit(X_train,y_train1)
y_pred = lreg.predict(X_test)
print("Linear: Training set score: {:.2f}".format(lreg.score(X_train, y_train1)))


# "The kernel was killed for trying to exceed the memory limit of
#  8589934592; you can use the restart button in the toolbar to try.
# "
# It looks like I have to find a way to reduce the number of features so as to reduce the memory requirement for prediction. So I randomly truncated the number of categorical features, including the first 30 gives 32% score for the training data prediction.similarly, including 80 of the categorical features  gives score of 47%, 95|49%, 101|49%,

# In[ ]:


#Other SGD regressor
from sklearn.linear_model import SGDRegressor
reg= SGDRegressor()
reg.fit(X_train, y_train1)
y_pred = reg.predict(X_test)
print("SGD: Training set score: {:.2f}".format(reg.score(X_train, y_train1)))


# In[ ]:


#Using Lasso
'''from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.6)
reg.fit(X_train, y_train1)
y_pred = reg.predict(X_test)
print("Lasso: Training set score: {:.2f}".format(reg.score(X_train, y_train1)))'''


# In[ ]:


submission = pd.DataFrame({
        "id": test["id"],
        "loss": y_pred
    })
submission.to_csv('sample_submission.csv', index=False)

