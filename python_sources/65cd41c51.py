#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:





# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train.columns


# In[ ]:


train.target.value_counts()


# In[ ]:


train['target'].value_counts().plot(kind="bar")


# In[ ]:


train.isna().sum().sum()


# In[ ]:


test.isna().sum().sum()


# In[ ]:


import gc


# In[ ]:


gc.collect();
train.describe()


# In[ ]:





# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.corr()


# In[ ]:


test.corr()


# In[ ]:


sns.distplot(train.var_0) 
sns.distplot(train.var_10) 
sns.distplot(train.var_20) 
sns.distplot(train.var_30) 
sns.distplot(train.var_40) 
sns.distplot(train.var_50) 


# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(25, 5))
sns.distplot(train['var_0'], ax=ax[0], color='orange')
sns.distplot(train['var_10'], ax=ax[1], color='red')
sns.distplot(train['var_20'], ax=ax[2], color='yellow')
sns.distplot(train['var_30'], ax=ax[3], color='green' )
plt.show()


# In[ ]:


X = train.iloc[:,2:]


# In[ ]:


y = train.iloc[:,1]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 100, stratify = y)


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)


# In[ ]:


confusion_matrix


# In[ ]:


classification_report(y_test, y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print("accuracy: ", accuracy_score(y_test,y_pred))


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train, y_train)


# In[ ]:


y_predit_svc = gnb.predict(X_test)


# 

# In[ ]:


roc_auc_score(y_test, y_predit_svc)


# In[ ]:


test


# In[ ]:


xx = test[test.columns[1:]]


# In[ ]:


yy = test[test.columns[:1]]


# In[ ]:


yy_pred = gnb.predict(xx)


# In[ ]:





# In[ ]:


my_submission = pd.DataFrame({'ID_code': yy.ID_code,'target': yy_pred})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

accuracy_score(y_test, y_predit_svc)


# In[ ]:


confusion_matrix(y_test,y_predit_svc)


# In[ ]:


print(classification_report(y_test, y_predit_svc))


# In[ ]:


y_preds_res = gnb.predict(X)


# In[ ]:


accuracy_score(y, y_preds_res)


# In[ ]:


confusion_matrix(y, y_preds_res)


# In[ ]:


print(classification_report(y, y_preds_res))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X, y)


# In[ ]:


y_preds = clf.predict(X)


# In[ ]:


accuracy_score(y, y_preds)


# In[ ]:


X


# In[ ]:



from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_preds)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




