#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt


# In[ ]:


matplotlib inline


# In[ ]:


train = pd.read_csv("/kaggle/input/sf-crime/train.csv")
test = pd.read_csv("/kaggle/input/sf-crime/test.csv")


# ## Intro

# I'm just creating a quick notebook for this competition to try to keep my skills sharp. I've been using mostly SQL and Tableau at work lately, and I didn't want my python and data science skills to get too rusty. I'm not spending much time improving my score or going into much detail.  I'm just doing some very basic EDA, Feature Engineering, and getting a basic, functioning model up and running.

# ## Exploratory Data Analysis

# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


test.head()


# Columns in test not in train

# In[ ]:


np.setdiff1d(test.columns,train.columns)


# ### Columns in train not in test

# In[ ]:


np.setdiff1d(train.columns,test.columns).tolist()


# No Null Values

# In[ ]:


train.isnull().sum()


# Unique Value Counts

# In[ ]:


train.nunique()


# ### A closer look at a few select counts

# In[ ]:


train['Resolution'].value_counts().plot.barh();


# Resolution excluding NONE

# In[ ]:


train[train['Resolution']!='NONE']['Resolution'].value_counts().plot.barh();


# In[ ]:


train['PdDistrict'].value_counts().plot.barh();


# In[ ]:


train['DayOfWeek'].value_counts().plot.barh();


# In[ ]:


train['Category'].value_counts().plot.barh(figsize = (5,18));


# # Feature Engineering
dropping "Descript" and "Resolution" from train data since those features aren't in the test data.
# In[ ]:


train_feats = train.drop(labels = ['Descript','Resolution'],axis = 1)
train_feats.head()


# Will create dummy variables for PdDistrict and DayOfWeek

# In[ ]:


train_dummies = pd.get_dummies(train_feats[['PdDistrict','DayOfWeek']])
train_feats_dummies = pd.merge(train_feats.drop(['PdDistrict','DayOfWeek'],1),train_dummies,left_index = True, right_index = True)
train_feats_dummies.columns


# In[ ]:


test_dummies = pd.get_dummies(test[['PdDistrict','DayOfWeek']])
test_feats_dummies = pd.merge(test.drop(['PdDistrict','DayOfWeek'],1),test_dummies,left_index = True, right_index = True)
test_feats_dummies = test_feats_dummies.drop(['Dates','Address'],1)
test_feats_dummies.columns


# In[ ]:


X = train_feats_dummies.drop('Category',1)
y = train_feats_dummies[['Category']]


# Dropping "Dates" and "Address" for now...
# I might come up with something to do for those but not sure...

# In[ ]:


X = X.drop(['Dates','Address'],1)
X.head(2)


# In[ ]:


y.head(2)


# Set up to predict probability for all categories, so get dummies

# In[ ]:


pd.get_dummies(y,prefix = None,prefix_sep = '')


# ## Train/Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)


# ## Modeling

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# y_train.Category.tolist()
# pd.get_dummies(y_train)
# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(y_train.Category)
# X_train.pop('X')
# le = LabelEncoder()
# le.fit_transform(y_train)
# enc.transform(y_train).toarray()

# In[ ]:


#clf = GaussianNB().fit(X_train, y_train['Category']) #worked (maybe)
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train['Category']) #worked
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr').fit(X_train, y_train['Category']) #worked


# Looking for prediction probabilities for each category.
# Can see below that there are unique coefficients for each of the 39 categories.

# In[ ]:


clf.coef_.shape


# In[ ]:


clf.classes_


# ## Model Evaluation

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[ ]:


y_pred = clf.predict(X_test)

print(np.unique(y_pred,return_counts = True))
#print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# ## Predictions

# In[ ]:


X_train.head(2)


# In[ ]:


test_feats_dummies.head(2)


# In[ ]:


test_predictions = clf.predict(test_feats_dummies.drop('Id',1))
test_predictions


# In[ ]:


test_probability_predictions = clf.predict_proba(test_feats_dummies.drop('Id',1))


# In[ ]:


test_probability_predictions.shape


# In[ ]:


clf.classes_


# In[ ]:


submission = pd.merge(test['Id'],pd.DataFrame(data = test_probability_predictions,columns = clf.classes_),          left_index = True, right_index = True)


# In[ ]:


submission.to_csv('submission.csv', index = False)


# ## Closing Thoughts

# I could add a lot more to make this better. I really didn't try many models. I didn't even train on the full train data, which I should have done after deciding which model to use. I also would have found the 50 or so most common addresses and created dummy variables from them and added them as features. I expect this would have been a significant added value.

# I'm just leaving this as is, since my objective was just to keep my skills sharp. I've been using mostly SQL and Tableau these days, so I didn't want my python and data science skills to get too rusty.
