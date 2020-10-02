#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[31]:


#Load the data using panda's read_csv functiion as per the "assignment 1" instruction 
data = pd.read_csv("../input/loan.csv", low_memory = False)


# In[32]:


#Drop all obs that aren't either fully paid or in default
data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]


# In[33]:


#transfer loan status into a binary vatiable
data['target'] = (data.loan_status == 'Fully Paid')
data.head()


# In[34]:


#Question1
print('There are ' + str(len(data))+ ' records, and ' + str(data.shape[1]) + ' features')


# In[35]:


#Question 2: Plot the distribution of the loan amount
import matplotlib as plt
plt.pyplot.hist(data.loan_amnt,10)


# In[36]:


#Compute the mean, median, maximum and standard deviation
import statistics as stat
print(f'Mean: {data.loan_amnt.mean():.2f}, median: {data.loan_amnt.median():.2f}, max: {max(data.loan_amnt):.2f}, standard deviation: {data.loan_amnt.std():.2f}')


# In[37]:


#Question 3: show the unique term; another method: use the logic fucntion as before
data.term.unique()


# In[38]:


#Compute the mean and standard deviation of the short-term (36 months) and long-term (60 months)
print(f'The mean of the short-term interest rate is: rate is {data.groupby(by="term").mean().int_rate[0].item():.2f}%, \nand the mean of the long-term interest rate is: {data.groupby(by="term").mean().int_rate[1].item():.2f}%.')

print(f'The standard deviation of the short-term interest rate is: rate is {data.groupby(by="term").std().int_rate[0].item():.2f}, \nand the standard deviation of the long-term interest rate is: {data.groupby(by="term").std().int_rate[1].item():.2f}.')


# In[39]:


#Box plot of interest rate by term
data.boxplot('int_rate',by='term')


# In[40]:


#Question 4:average interest rate on the debt grade with the highest average interest rate
print(f'The average interest rate on the debt grade with the highest average interest rate is {max(data.groupby(by="grade").mean().int_rate):.2f}, and the grade is {data.groupby(by="grade").int_rate.mean().idxmax()}.')


# In[41]:


#Question 5:highest realized yield
print(f'The highest realized yield is {max(data.groupby(by="grade").sum().total_pymnt/data.groupby(by="grade").sum().loan_amnt-1):.2f}, and the grade is {(data.groupby(by="grade").sum().total_pymnt/data.groupby(by="grade").sum().loan_amnt).idxmax()}.')


# In[42]:


#Question 6: Number of records for each application type
#6a: How many records for each application type are there? 
print(f'There are {data.groupby(by="application_type").application_type.count()[0].item()} individual applications and {data.groupby(by="application_type").application_type.count()[1].item()} joint applications.')

#6b: Does it make sense to use this feature?
#Firstly, calculate the rate of fully paid loan for each application type
print(f'There are {float(data.groupby(by="application_type").target.sum()[0].item()*100/data.groupby(by="application_type").application_type.count()[0].item()):.3f}% of individual applications that are fully paid and \n{float(data.groupby(by="application_type").target.sum()[1].item()*100/data.groupby(by="application_type").application_type.count()[1].item()):.3f}% of joint application are fully paid.')

print('The percentage of fully paid loan does not differ signicantly across the twoapplication type.')

print('Therefore, this feature contains no useful information for our prediction.')


# In[43]:


#Question 7
df = pd.concat([data.loan_amnt,data.funded_amnt, data.funded_amnt_inv,data.int_rate, pd.get_dummies(data.emp_length),                pd.get_dummies(data.term),pd.get_dummies(data.addr_state),pd.get_dummies(data.verification_status),                 pd.get_dummies(data.purpose),data.policy_code], axis=1)


print("The total width is: " + str(df.shape[1]) + ".")


# In[44]:


#Question 8
from sklearn.model_selection import train_test_split
X_train,X_test = train_test_split(df,test_size=0.67, random_state=42)
print("The shape of X_train is: " + str(X_train.shape))


# In[49]:


#Question 9
from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test = train_test_split(df,data.target,test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'Accuracy: {clf.score(X_test,y_test)*100:.4f}%')


# In[50]:


#Question 10
from sklearn.ensemble import RandomForestRegressor
X_train2,X_test2,y_train2,y_test2 = train_test_split(df,data.total_pymnt,test_size=0.33, random_state=42)
rfr = RandomForestRegressor(n_estimators=100, max_depth=4,random_state=42)
rfr.fit(X_train2, y_train2)

print(f'Accuracy: {rfr.score(X_test2, y_test2)*100:.4f}%')


# In[51]:


#Question 11
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

print("All Class 0 are mispredicted as class 1.")


# In[52]:


# Class count: Fully repaid and Default
count_fully, count_default = data.target.value_counts()

# Divide by class
df_default = data[data['target'] == 0]
df_fully = data[data['target'] == 1]

#Under-sampling
df_fully_under = df_fully.sample(count_default)
df_test_under = pd.concat([df_fully_under, df_default], axis=0)

print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)');

#Over-sampling
df_default_over = df_default.sample(count_fully, replace=True)
df_test_over = pd.concat([df_fully, df_default_over], axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');

