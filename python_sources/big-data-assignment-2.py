#!/usr/bin/env python
# coding: utf-8

# In[31]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[32]:


train = pd.read_csv('../input/train.csv', low_memory = False)


# In[44]:


train.head()


# ****1. What percentage of your training set loans are in default? ****

# In[33]:


print(np.mean(train.default == True))


# **** 2. Which ZIP code has the highest default rate? ****

# In[34]:


ZIP_labels = np.unique(train.ZIP)
print(ZIP_labels)


# In[35]:


train.groupby(by = 'ZIP').default.mean()


# **** 3. What is the default rate in the first year for which you have data? ****

# In[36]:


min_year_vector = (train.year == min(np.unique(train.year)))
print(sum((train.default == True) & min_year_vector)/sum(min_year_vector))


# **** 4. What is the correlation between age and income? ****

# In[37]:


np.corrcoef(train.age, train.income)


# **** 5. What is the in-sample accuracy? That is, find the accuracy score of the fitted model for 
# predicting the outcomes using the whole training dataset. ****

# In[76]:


y_train = train.default
X_train = pd.get_dummies(train[['loan_size', 'payment_timing', 'education', 'occupation', 'income','job_stability', 'ZIP', 'rent']])

X_train.head()


# In[77]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state = 42, oob_score = True)
model.fit(X_train, y_train)


# In[78]:


from sklearn.metrics import accuracy_score

in_sample_acc = accuracy_score(model.predict(X_train), y_train)
print(in_sample_acc)


# **** 6. What is the out of bag score for the model? The out of bag score is a fit and validation 
# method that is calculated while the model is being trained. For each observation in the dataset, 
# a prediction is found using all trees that do not contain that observation in their bootstrap 
# sample. The out of bag score is the resulting average error from this set of predictions. ****

# In[79]:


print(model.oob_score_)


# **** 7. What is the out of sample accuracy? That is, find the accuracy score of the model using the 
# test data without re-estimating the model parameters. ****

# In[80]:


test = pd.read_csv('../input/test.csv')
y_test = test.default
X_test = pd.get_dummies(test[['loan_size', 'payment_timing', 'education', 'occupation', 'income','job_stability', 'ZIP', 'rent']])
oos_accuracy = accuracy_score(model.predict(X_test), y_test)
print(oos_accuracy)


# **** 8. What is the predicted average default probability for all non-minority members in the test 
# set? ****

# In[82]:


predict_proba = model.predict_proba(X_test)
non_minority_vec = (test.minority == 0)*1
non_minority_default_probability = np.dot(predict_proba[:, 1], non_minority_vec)/sum(non_minority_vec)
print(non_minority_default_probability)


# **** 9. What is the predicted average default probability for all minority members in the test set? ****

# In[83]:


minority_vec = (test.minority == 1)*1
minority_default_probability = np.dot(predict_proba[:, 1], minority_vec)/sum(minority_vec)
print(minority_default_probability)


# **** 10. Is the loan granting scheme (the cutoff, not the model) group unaware? (This question does 
# not require calculation as the cutoff is given in the introduction to this assignment) ****

# In[ ]:


X_test.columns


# **** 11. Has the loan granting scheme achieved demographic parity? Compare the share of 
# approved female applicants to the share of rejected female applicants. Do the same for 
# minority applicants. Are the shares roughly similar between approved and rejected applicants? 
# What does this indicate about the demographic parity achieved by the model? ****

# In[ ]:


print(np.dot(model.predict(X_test), (X_test.sex == 0))/sum(X_test.sex == 0))


# In[ ]:


np.dot(model.predict(X_test), np.ones((160000, 1)))


# In[ ]:


import matplotlib.pyplot as plt

a = np.multiply (predict_proba[:, 1]),(X_test.sex == 0)
print(a)


# In[ ]:




