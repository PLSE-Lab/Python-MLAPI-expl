#!/usr/bin/env python
# coding: utf-8

# **Decision Tree vs RandomForest** : Dataset used is Lending Club Loan Data

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


# **Import Libraries**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier


# **Reading the Data from CSV file**

# In[ ]:


dfloan=pd.read_csv("../input/loan_data.csv")
dfloan.head()


# In[ ]:


dfloan.info()


# In[ ]:


dfloan.describe()


# **Exploratory Data Analysis**

# In[ ]:


plt.figure(figsize=(15,10))
dfloan[dfloan['credit.policy']==1]['fico'].hist(bins=30,color='blue',rwidth=0.95,label='Credit Policy=1',alpha=0.6)
dfloan[dfloan['credit.policy']==0]['fico'].hist(bins=30,color='grey',rwidth=0.95,label='credit policy=0',alpha=0.6)
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(15,10))
dfloan[dfloan['not.fully.paid']==1]['fico'].hist(bins=30,color='blue',label='Not Fully Paid=1',alpha=0.6)
dfloan[dfloan['not.fully.paid']==0]['fico'].hist(bins=30,color='grey',label='Not Fully Paid=0',alpha=0.6)
plt.legend()
plt.xlabel('FICO')


# **Countplot using seaborn showing the counts of loan by purpose, with the color hue defined by not.fully.paid,**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='purpose',hue='not.fully.paid',data=dfloan,palette='Set1')


# Lets see the trend between FICO score and interest rate using Jointplot

# In[ ]:


sns.jointplot(x='fico',y='int.rate',data=dfloan)


# LMPlots to see if the trend differed between not.fully.paid and credit policy.

# In[ ]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=dfloan,hue='credit.policy',col='not.fully.paid',palette='Set1')


# **Setting up the Data** Since we have purpose which is a caregorical.
# create a list of 1 element containing the string 'purpose'.

# In[ ]:


cat_feats=['purpose']


# Now use pd.get_dummies(DF,columns=,drop_first=) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data

# In[ ]:


final_data=pd.get_dummies(dfloan,columns=cat_feats,drop_first=True)


# In[ ]:


final_data.info()


# In[ ]:


final_data.head()


# **Train Test Split** 
# Use sklearn to split your data into a training and testing set 

# In[ ]:


X=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# Trainign a DecisionTree Model
# Let's start by training single decision tree first
# Import DecisionTreeClassifier

# In[ ]:


dtree=DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# **Predictions and Evaluations of Decision Tree**
# Create predictions from test set and create a classification report and a confusion matrix.

# In[ ]:


dtreepredictions=dtree.predict(X_test)


# **Printing Classification Report**

# In[ ]:


print(classification_report(y_test,dtreepredictions))


# **Printing Confusion Matrix**

# In[ ]:


print(confusion_matrix(y_test,dtreepredictions))


# **Training the RandomForest Modeel**
# Create an instance of RFC class and fit it to our training data from the previous step

# In[ ]:


rfc=RandomForestClassifier(n_estimators=300)


# In[ ]:


rfc.fit(X_train,y_train)


# **Predictions and Evaluations** 
# Lets predict off the y_test values and evaluate our model.
# Predict the class of not.fully.paid for the X_test data
# 

# In[ ]:


rfcpredictions=rfc.predict(X_test)


# **Printing Classification Report******

# In[ ]:


print(classification_report(y_test,rfcpredictions))


# **Priting Confusion Matrix**

# In[ ]:


print(confusion_matrix(y_test,rfcpredictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




