#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
train = pd.read_csv('../input/train.csv',index_col=0)


# In[ ]:


# Question One : What percentage of your training set loans are in default? 
train.default.mean()


# In[ ]:


# Question Two : Which ZIP code has the highest default rate?
train.groupby('ZIP')['default'].mean().idxmax()


# In[ ]:


# Question Three : What is the default rate in the first year for which you have data?
train[train.year == 0].default.mean()
# Note:Python starts indexing at 0


# In[ ]:


#Question Four : What is the correlation between age and income? 
train.age.corr(train.income)


# In[ ]:


# Question Five : What is the in-sample accuracy?Load random forest classifier package
#Load random forest classifier package
from sklearn.ensemble import RandomForestClassifier

# Define features and outcome variable
predictors = ['ZIP', 'rent', 'education', 'income','loan_size', 'payment_timing', 'job_stability','occupation']
target = ['default']
X = pd.get_dummies(train[predictors])
y = train[target]

# Define model hyperparameters
clf = RandomForestClassifier(n_jobs=-1,n_estimators=100,oob_score=True,random_state=42)
          
# Train model
clf.fit(X=X,y=y)
          
# Load accuracy score package and compute it
from sklearn.metrics import accuracy_score
accuracy_score(y_pred = clf.predict(X),y_true=y)


# In[ ]:


# Question Six:What is the out of bag score?
clf.oob_score_


# In[ ]:


# Question Seven : What is the test set accuracy?
# Load data and define features and outcome variable
test = pd.read_csv('../input/test.csv')
X_test = pd.get_dummies(test[predictors])
y_test = test[target].values
# Compute accuracy score
accuracy_score(y_pred = clf.predict(X_test),y_true=y_test)


# In[ ]:


# Question Eight : What is the predicted average default probability for all non-minority members in the test set?
# Save predictions for non-minority members
maj_dist = clf.predict(X_test[test.minority == 0])
# Compute average default probability
maj_dist.mean()


# In[ ]:


# Question Nine : What is the predicted average default probability for all minority members in the test set?
# Save predictions for minority members
min_dist = clf.predict(X_test[test.minority == 1])
# Compute average default probability
min_dist.mean()


# In[ ]:


# Question Ten: Is the loan granting scheme (the cut-off, not the model) group unaware?
print('Yes, because there is a fixed cut-off (no calculation needed)')


# In[ ]:


#Question Eleven : Has the loan granting scheme demographic parity? In one line, provide numbers that support your 
#answer. Examine both accepted and rejected applicant portfolios.

#Save model predictions
pred = clf.predict(X_test)

#Successful applicants
print(f'Share of succesful applicants amongst: minority members: {(len(test[(~pred) & (test.minority==1)]) / len(test[test.minority==1]))*100:.2f}%, non-minority members: {(len(test[(~pred) & (test.minority==0)]) / len(test[test.minority==0]))*100:.2f}%')
print(f'Share of succesful applicants amongst: females: {(len(test[(~pred) & (test.sex==1)]) / len(test[test.sex==1]))*100:.2f}%, males: {(len(test[(~pred) & (test.sex==0)]) / len(test[test.sex==0]))*100:.2f}%')

#Rejected applicants
print(f'Share of rejected applicants amongst: minority members: {(len(test[(pred) & (test.minority==1)]) / len(test[test.minority==1]))*100:.2f}%, non-minority members: {(len(test[(pred) & (test.minority==0)]) / len(test[test.minority==0]))*100:.2f}%')
print(f'Share of rejected applicants amongst: females: {(len(test[(pred) & (test.sex==1)]) / len(test[test.sex==1]))*100:.2f}%, males: {(len(test[(pred) & (test.sex==0)]) / len(test[test.sex==0]))*100:.2f}%')


# In[ ]:


#Question Twelve: Has the loan granting scheme equal opportunity? In one line, provide numbers that support your answer
print (f'Share defaulting on their loans : minority members : {y_test[(test.minority == 1) & (~pred)].mean()*100:.2f}%        non minority : {y_test[(test.minority == 0) &(~pred)].mean()*100:.2f}%        male : {y_test[(test.sex == 0) & (~pred)].mean()*100:.2f}%        female : {y_test[(test.sex == 1) & (~pred)].mean()*100:.2f}%')


# In[ ]:


#Additional code for the group presentation

# Look at age feature
min(test.age)
max(test.age)
test.age.median()

# Define age groups
test['young'] = (test.age <= 43)

# Calculate share of successful applicants
print(f'Share of successful applicants amongst: young: {(len(test[(~pred) & (test.young==1)]) / len(test[test.young==1]))*100:.2f}%, old: {(len(test[(~pred) & (test.young==0)]) / len(test[test.young==0]))*100:.2f}%')

# Calculate share of rejected applicants
print(f'Share of rejected applicants amongst: young: {(len(test[(pred) & (test.young==1)]) / len(test[test.young==1]))*100:.2f}%, old: {(len(test[(pred) & (test.young==0)]) / len(test[test.young==0]))*100:.2f}%')

# Calculate share of defaulting
print (f'Share of defaulting on their loans : young: {y_test[(test.young == 1) & (~pred)].mean()*100:.2f}%, old: {y_test[(test.young == 0) &(~pred)].mean()*100:.2f}%')


# In[ ]:


#heat map calculated using Pearson's correlation only
colormap = plt.cm.RdBu

plt.figure(figsize=(22,20))
train_feature = pd.get_dummies(train)
sns.heatmap(train_feature.astype(float).corr().round(2), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True,  fmt='.1g' )


# In[ ]:


#Modified correlation matrix :calculate pairwise correlation for different feature types using its corresponding correlations:
#i.e Kendall-Tau for both categorical, Bi-serial for categorical vs continues and Pearson for continous variables
import scipy.stats as stats
temp = 0;
var_corr =[[],[],[]]
train_feature = pd.get_dummies(train)
corr1 = [['sex',0],['age',1],['minority',0]]
corr2 = [['minority',0],['sex',0],['rent',0],['education',1],['age',1],['income',1],['loan_size',1],['payment_timing',1],['year',1],         ['job_stability',1],['default',0],['ZIP_MT01RA',0],['ZIP_MT04PA',0],['ZIP_MT12RA',0],['ZIP_MT15PA',0],['occupation_MZ01CD',0],         ['occupation_MZ10CD',0],['occupation_MZ11CD',0]]
for i in corr1:
    for j in corr2:
        if i[1] == j[1]:
            if i[1] == 0:
                test = stats.kendalltau(train_feature[i[0]], train_feature[j[0]])
                var_corr[temp].append(test[0])
            else:
                var_corr[temp].append(train_feature[i[0]].corr(train_feature[j[0]]))
        else:
            test = stats.pointbiserialr(train_feature[i[0]], train_feature[j[0]])
            var_corr[temp].append(test[0])
    temp += 1

var_corr

