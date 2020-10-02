#!/usr/bin/env python
# coding: utf-8

# # PREDICTING A PULSAR STAR
# 
# Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted, which treat the candidate data sets as binary classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class.
# 
# The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. These examples have all been checked by human annotators.
# 
# Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).
# 
# Machine Learning algorithms used to predict the target in the following dataset are
# 
# 1-Logistic Regression
# 
# 2-Random Forest Classifier 
# 
# 3-Support Vector Classification
# 
# the github repo to this same dataset can be found here https://github.com/sid26ranjan/pulsar-star
# 
# also check out my other kernels and github repos https://github.com/sid26ranjan

# In[ ]:


import pandas as pd


# In[ ]:


data=pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')


# ###### SOME BASIC INFORMATION ABOUT THE DATA SET

# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# ###### TARGET  CLASS

# In[ ]:


data['target_class'].unique()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.countplot(data['target_class'])


# In[ ]:


data[data['target_class']==1].count()


# In[ ]:


data[data['target_class']==0].count()


# In[ ]:


data.head()


# ###### Attribute Information:
# 
# Each candidate is described by 8 continuous variables, and a single class variable. 
# 
# The first four are simple statistics obtained from the integrated pulse profile (folded profile).
# 
# This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency .
# 
# The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:
# 
# Mean of the integrated profile.
# 
# Standard deviation of the integrated profile.
# 
# Excess kurtosis of the integrated profile.
# 
# Skewness of the integrated profile.
# 
# Mean of the DM-SNR curve.
# 
# Standard deviation of the DM-SNR curve.
# 
# Excess kurtosis of the DM-SNR curve.
# 
# Skewness of the DM-SNR curve.
# 
# Class

# ###### CORRELATION BETWEEN THE FEATURES

# In[ ]:


correlation=data.corr()
fig = plt.figure(figsize=(12, 10))

sns.heatmap(correlation, annot=True, center=1)


# In[ ]:


y=data['target_class']


# In[ ]:


X=data.drop(['target_class'],axis=1)


# In[ ]:


X.head()


# In[ ]:


X.describe()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ###### LOGISTIC REGRESSION:

# In[ ]:


model=LogisticRegression()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# ###### RANDOM FOREST CLASSIFIER :

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


print(classification_report(y_test, pred_rfc))


# In[ ]:


rfc.score(X_test,y_test)


# ###### SUPPORT VECTOR CLASSIFICATION :

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf = SVC() 


# In[ ]:


clf.fit(X_train, y_train) 


# In[ ]:


pred_svc=clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_svc))


# In[ ]:


clf.score(X_test,y_test)


# In[ ]:


print('Accuracy from Support Vector Classification is '+str(clf.score(X_test,y_test)*100)+"%")
print('Accuracy from Random forest classifier is '+str(rfc.score(X_test,y_test)*100)+'%')
print('Accuracy from Logistic regression is '+str(model.score(X_test,y_test)*100)+'%')


# **If you find this kernel useful please upvote.**

# In[ ]:




