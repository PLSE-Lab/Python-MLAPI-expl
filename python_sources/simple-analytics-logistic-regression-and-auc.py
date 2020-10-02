#!/usr/bin/env python
# coding: utf-8

# The data is about the default in finance area. As far as I investigated, on this area, the data are basically imbalance and the analystics and the modeling is based on that fact.  
# 
# About this data, the explained variable is default.payment.next.month. So, we can imagine the degree of inbalance is not so much.  

# In[12]:


import pandas as pd

# load data
data_orig = pd.read_csv("../input/UCI_Credit_Card.csv")
data_orig.head(10)


# In[13]:


print("data size: " + str(data_orig.shape))
print("default size: " + str(data_orig.ix[data_orig['default.payment.next.month'] == 1,:].shape))


# ID information is sometimes important. We should not drop without caution. But here, I'll drop.  .  

# In[14]:


# omit-target columns
omit_target_label = ['ID']

# categorical columns
pay_label = ['PAY_'+str(i) for i in range(0,7) if i != 1]
categorical_label = ['SEX', 'EDUCATION', 'MARRIAGE']

categorical_label.extend(pay_label)


# In[15]:


dummied_columns = pd.get_dummies(data_orig[categorical_label].astype('category'))

# drop columns
data_orig = data_orig.drop(columns=categorical_label)
data_orig = data_orig.drop(columns=omit_target_label)

# merge one-hot-encoded columns
data = pd.concat([data_orig, dummied_columns], axis=1, join='outer')


# To make models, I'll do split the data into train and test ones.  

# In[16]:


from sklearn.model_selection import train_test_split

# explaining and explained
target = data['default.payment.next.month']
data = data.drop(columns=['default.payment.next.month'])

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.33)


# I'll do univariate analysis in rough manner.  
# At first, by using just one variable, I'll make a model and evaluate it by AUC. If this data had the information of TIME, I could split the test data into some groups based on TIME and check AUC per group to see the robustness.  

# In[17]:


from sklearn.linear_model import LogisticRegression

univar = x_train[['BILL_AMT4']]

lr = LogisticRegression()
lr.fit(univar, y_train)


# In[18]:


from sklearn.metrics import roc_auc_score
import numpy as np

predicted_score = np.array([score[1] for score in lr.predict_proba(x_test[['BILL_AMT4']])])

roc_auc_score(y_test.values, predicted_score)


# I'll do same thing to each explaining variables. To think about the conbination of variables, AIC and BIC work well. But here, I don't touch.  

# In[19]:


explaining_labels = x_train.columns
auc_outcomes = {}
for label in explaining_labels:
    univar = x_train[[label]]
    
    lr = LogisticRegression()
    lr.fit(univar, y_train)
    
    predicted_score = np.array([score[1] for score in lr.predict_proba(x_test[[label]])])
    
    auc_outcomes[label] = roc_auc_score(y_test.values, predicted_score)


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

label = []
score = []
for item in sorted(auc_outcomes.items(), key=lambda x: x[1], reverse=True):
    label.append(item[0])
    score.append(item[1])

# I wanted to show the bars with decreasing order. But it didn't work here.
plt.bar(label, score)


# 

# From the viewpoint of AUC, some models reach 0.6 and the others are around 0.5.  
# Next, with all variables, I'll make Logistic regression model. This time, I don't do standardization.  

# In[49]:


# using all the explaining variables
lr = LogisticRegression()
lr.fit(x_train, y_train)

predicted_score = np.array([score[1] for score in lr.predict_proba(x_test)])
    
roc_auc_score(y_test.values, predicted_score)


# From sklearn, I'll use different evaluation way.  

# In[47]:


from sklearn.metrics import brier_score_loss

brier_score_loss(y_test.values, predicted_score)


# To the train and test data, I'll do prediction and evaluate those with AUC and Brier.  

# In[48]:


predicted_score_train = np.array([score[1] for score in lr.predict_proba(x_train)])
predicted_score_test = np.array([score[1] for score in lr.predict_proba(x_test)])

auc_train = roc_auc_score(y_train.values, predicted_score_train)
auc_test = roc_auc_score(y_test.values, predicted_score_test)
brier_train = brier_score_loss(y_train.values, predicted_score_train)
brier_test = brier_score_loss(y_test.values, predicted_score_test)

auc = [auc_train, auc_test]
brier = [brier_train, brier_test]
pd.DataFrame({'auc': auc, 'brier': brier}, index=['train', 'test'])

