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


#Question 1:What percentage of your training set loans are in default?
#Load the data
train= pd.read_csv('../input/train.csv', low_memory = False)

#Describe the data 
train.head()

#Percentage of default loans
print(f'{len(train[train["default"]==True])/len(train)*100:.4f}% of the training set loans are in default.')


# In[ ]:


#Question2: Which ZIP code has the highest default rate?
highest_default_rate = np.nanmax(train[train.default==True].groupby(['ZIP']).                                 default.count()/train.groupby(['ZIP']).default.count())*100
ZIP = (train[train.default==True].groupby(['ZIP']).       default.count()/train.groupby(['ZIP']).default.count()).idxmax()

print(f'The zip code with the highest default rate is {ZIP}, and the highest rate is {highest_default_rate:.2f}%.')


# In[ ]:


#Question 3: What is the default rate in the first year for which you have data?
train_year1 = train[train.year==min(train.year)]

print(f'The default rate in the first year is {100*len(train_year1[train_year1.default==True])/len(train_year1):.2f}% ')


# In[ ]:


#Question 4: What is the correlation between age and income?
print(f'The correlation between age the income is {train["age"].corr(train["income"]):.4f}.')


# In[ ]:


#Question 5: What is the in-sample accuracy?
#Model 1: Exclude both gender and minority
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#Construct the X_train set
X_train = train.filter(['loan_size','payment_timing','education','occupation','income',                        'job_stability','ZIP','rent'])
X_train = pd.get_dummies(X_train)
#Fit the model
clf = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42,oob_score=True)
trained_model = clf.fit(X_train, train.default)
#Calculate the in-sample accuracy
accuracy_score(train.default, trained_model.predict(X_train))
print(f'The in-sample accuracy is: {accuracy_score(train.default, trained_model.predict(X_train))*100:.4f}%')


# In[ ]:


#Question 6: What is the out of bag score?
print(f'The out-of-bag score is: {clf.oob_score_*100:.4f}%')


# In[ ]:


#Question 7: What is the out of sample accuracy?
test= pd.read_csv('../input/test.csv', low_memory = False)
X_test = test.filter(['loan_size','payment_timing','education','occupation','income',                        'job_stability','ZIP','rent'])
X_test = pd.get_dummies(X_test)
print(f'The out-of-sample accuracy is: {accuracy_score(test.default, trained_model.predict(X_test))*100:.4f}%')


# In[ ]:


#Question 8: What is the predicted average default probability for all non-minority members in the test set? 
print(f'The predicted average default probability for all non-minority members is {100*clf.predict_proba(X_test)[test.minority==0][:,1].mean():.4f}%.')


# In[ ]:


#Question 9: What is the predicted average default probability for all minority members in the test set?
print(f'The predicted average default probability for all minority members is {100*clf.predict_proba(X_test)[test.minority==1][:,1].mean():.4f}%.')


# In[ ]:


#Question 10: Is the loan granting scheme (the cutoff, not the model) group unaware? 
print('It is group unaware because all groups apply for the same thresholds,which is 50%.')


# In[ ]:


#Question 11: Compare the share of approved female applicants to the share of rejected female applicants. 
#Sex Group 0
prediction1=clf.predict_proba(X_test)[test.sex==0][:,0]
print(f'The share of approved sex group 0 applicants is {100*len(prediction1[prediction1>=0.5])/len(prediction1):.4f}%')
print(f'The share of rejected sex group 0 applicants is {100*len(prediction1[prediction1<=0.5])/len(prediction1):.4f}%')

#Sex Group 1
prediction2=clf.predict_proba(X_test)[test.sex==1][:,0]
print(f'The share of approved sex group 1 applicants is {100*len(prediction2[prediction2>=0.5])/len(prediction2):.4f}%')
print(f'The share of rejected sex group 1 applicants is {100*len(prediction2[prediction2<=0.5])/len(prediction2):.4f}%')

#Non-Minority Group
prediction3=clf.predict_proba(X_test)[test.minority==0][:,0]
print(f'The share of approved non-minority applicants is {100*len(prediction3[prediction3>=0.5])/len(prediction3):.4f}%')
print(f'The share of rejected non-minority applicants is {100*len(prediction3[prediction3<=0.5])/len(prediction3):.4f}%')

#Minority Group
prediction4=clf.predict_proba(X_test)[test.minority==1][:,0]
print(f'The share of approved minority applicants is {100*len(prediction4[prediction4>=0.5])/len(prediction4):.4f}%')
print(f'The share of rejected minority applicants is {100*len(prediction4[prediction4<=0.5])/len(prediction4):.4f}%')


# In[ ]:


#Question 12: 
# Minority Group
predict_minority_default = clf.predict_proba(X_test)[(test.minority==1) & (test.default == True)][:,0]
print(f'The share of successful minority applicants that defaulted is {100*len(predict_minority_default[predict_minority_default>=0.5])/len(prediction4[prediction4>=0.5]):.4f}%')

#Non-minority Group
predict_nonminority_default = clf.predict_proba(X_test)[(test.minority==0) & (test.default == True)][:,0]
print(f'The share of successful non-minority applicants that defaulted is {100*len(predict_nonminority_default[predict_nonminority_default>=0.5])/len(prediction3[prediction3>=0.5]):.4f}%')

#Sex Group 0
predict_sex0_default = clf.predict_proba(X_test)[(test.sex==0) & (test.default == True)][:,0]
print(f'The share of successful Sex group 0 applicants that defaulted is {100*len(predict_sex0_default[predict_sex0_default>=0.5])/len(prediction1[prediction1>=0.5]):.4f}%')

#Sex Group 1
predict_sex1_default = clf.predict_proba(X_test)[(test.sex==1) & (test.default == True)][:,0]
print(f'The share of successful Sex group 1 applicants that defaulted is {100*len(predict_sex1_default[predict_sex1_default>=0.5])/len(prediction2[prediction2>=0.5]):.4f}%')

