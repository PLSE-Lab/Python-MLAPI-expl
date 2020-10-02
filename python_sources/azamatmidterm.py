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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print('Train:', train.shape)
print('Test:', test.shape)


# In[ ]:


sns.countplot(train['target'])


# In[ ]:


train['target'].value_counts()


# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train.mean(axis=1),color="black", label='train')
sns.distplot(test.mean(axis=1),color="red",label='test')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per rows in the train and test set")
sns.distplot(train.std(axis=1),color="blue",label='train')
sns.distplot(test.std(axis=1),color="green",label='test')
plt.legend(); plt.show()


# **My Logistic Regression**

# In[ ]:


Target = train['target']
train_inp = train.drop(columns = ['target', 'ID_code'])
test_inp = test.drop(columns = ['ID_code'])


# In[ ]:


X_train, X_test, y_train,  y_test = train_test_split(train_inp, Target,test_size=0.5, random_state=0)


# In[ ]:


print('Train:',X_train.shape)
print('Test:',X_test.shape)
print('Train:',y_train.shape)
print('Test:',y_test.shape)


# In[ ]:


logist = LogisticRegression(class_weight='balanced')
logist.fit(X_train, y_train)


# In[ ]:


logist_pred = logist.predict_proba(X_test)[:,1]


# In[ ]:


logist_pred


# In[ ]:


def performance(Y_test, logist_pred):
    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]
    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)
    print('AUC:')
    print(auc(fpr, tpr))


# In[ ]:


performance(y_test, logist_pred)


# In[ ]:


logist_pred_test = logist.predict_proba(test_inp)[:,1]
submit = test[['ID_code']]
submit['target'] = logist_pred_test
submit.head()


# In[ ]:


submit.to_csv('log_reg_baseline.csv', index = False)


# **DECISION TREE  MODEL**

# In[ ]:


tree = DecisionTreeClassifier(class_weight='balanced',max_depth=4)


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


tree_pred = tree.predict_proba(X_test)[:, 1]
performance(y_test, tree_pred)


# In[ ]:


tree = DecisionTreeClassifier(class_weight='balanced',max_depth=10)


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


tree_pred = tree.predict_proba(X_test)[:, 1]
performance(y_test, tree_pred)


# In[ ]:





# In[ ]:





# **The next is Random Forest**

# In[ ]:


model = RandomForestClassifier(n_estimators=100, class_weight='balanced')


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


forest_pred= model.predict_proba(X_test)[:, 1]
performance(y_test, forest_pred)


# **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


nb = GaussianNB()


# In[ ]:


nb.fit(X_train,y_train)


# In[ ]:


nb_pred = nb.predict_proba(X_test)[:, 1]
performance(y_test, nb_pred)


# In[ ]:


nb_pred


# In[ ]:


nb_pred_test = nb.predict_proba(test_inp)[:,1]
submit = test[['ID_code']]
submit['target'] = nb_pred_test
submit.head()


# In[ ]:


submit.to_csv('NB_baseline.csv', index = False)


# In[ ]:





# In[ ]:


from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier(max_depth=8,random_state=0)


# In[ ]:


xgb.fit(X_train, y_train)


# In[ ]:





# In[ ]:


xgb_pred = xgb.predict_proba(X_test)[:, 1]
performance(y_test, xgb_pred)

