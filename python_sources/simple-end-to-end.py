#!/usr/bin/env python
# coding: utf-8

# ## End to End scripts for a beginner 
# * This notebook uses sklearn from standard anaconda distribution. No additional libaries are required
# * To get the feet wet, we make the following assumption to test end-to-end for initial submission
# * remove all columns with N/A
# * treat all survey data as ordinal (not 1-hot encoded)
# * Target: *is_female*
# * Enjoy 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
TARGET = "is_female"


# #### There are 1235 columns!
# * target is at column 10
# * Lot's of NaNs. 
#     * ignore all NaN for now

# In[ ]:


df = pd.read_csv("../input/train.csv",low_memory=False)
print(df.shape)
df.head()


# #### drop all columns with NaNs  (quick and dirty approach to establish end-to-end baseline)

# In[ ]:


df2 = df.dropna(axis=1)
print (df2.shape)


# In[ ]:


# to get all columns names
df2.columns


# ### let's create training data
# * drop train_id and is_female  for X
# * take "is_female" as Y

# In[ ]:



X= df2.drop(['train_id','is_female'], axis=1)
Y = df2.is_female


# ### Let's check Test data
# * Ignore NaNs for now
# * need to drop first column (test_id) for prediction
# * take all columns matched to trainning  data 
# 

# In[ ]:


test = pd.read_csv("../input/test.csv",low_memory=False)
print(test.shape)


# In[ ]:


# keep those columns from training data X
test3 = test.reindex(columns=X.columns, fill_value=0)
test3.shape


# In[ ]:



print (Y.shape, X.shape, test3.shape)


# ### start training
# * let's split the data to know how difficult this task is
# * let's use Random Forest model is number of tree = 500

# In[ ]:


X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


Ntree = 500
clf = RandomForestClassifier(n_estimators=Ntree,random_state=1234)
clf.fit(X_train, y_train)


# ### Run prediction and get accuracy 

# In[ ]:


y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)


# ### However, to get AUC, we need probablity score

# In[ ]:


y_prob = clf.predict_proba(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


# let check which score do we need,   we need column 1  not column 0
y_prob


# In[ ]:


metrics.roc_auc_score(y_test, y_prob[:,1])


# ### Great!! we got auc = 0.956  
# * Uses 2/3 of data as training
# * 1/3 as validation
# * now we can use all data as training to predict test set for submission

# In[ ]:


Ntree = 500
clf2 = RandomForestClassifier(n_estimators=Ntree,random_state=1234)
clf2.fit(X, Y)
y_submit = clf2.predict_proba(test3)[:,1]
test['is_female'] = y_submit
ans = test[['test_id','is_female']]
ans.to_csv('submit.csv', index=None)


# ### Now you are ready for your first submission.
# * If you are lucky, you probabaly are in the middle of leader board
# * next step, you can 
#     * Uses different classifier
#     * parameter tuning using cross validation 
#     * missing data imputation
#     * bagging and emsembing
#     * deep learning neural network 
#     * enjoy the task

# In[ ]:




