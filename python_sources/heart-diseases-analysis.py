#!/usr/bin/env python
# coding: utf-8

# # Heart Diseases Analysis

# In this analysis we will check if the patient have heart disease or not based on the given features. We will train different models and will analyse them.

# ### Getting Started

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/heart.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dropna()
df.shape


# In[ ]:


df.describe()


# ### Visualising the data

# In[ ]:


plt.figure(figsize=(25, 10))
p = sns.heatmap(df.corr(), annot=True)
_ = plt.title('Correlation')


# The above plot can be used to determine the correlation between the different features of the dataset. From the above set we can also find out the features which have the most and the least effect on the target feature (whether the patient have heart diseases or not).

# In[ ]:


p = sns.countplot(x='sex', data=df)


# 0- Female ,1- Male.
# Above plot shows the number of males and female patients

# In[ ]:


p = sns.countplot(x='cp', data=df)


# The distribution above tells us the most common and least common type of chest pains

# ### Training the Models

# #### Preparing the dataset

# We will now split data into training and test set data.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


Y = df['target']
X = df.drop(columns=['target'], axis=1)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Splitting the dataset into training set and test set


# #### Using Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score # to find the accuracy for the trained model


# In[ ]:


# Setting Hyperparameters
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=1000)


# In[ ]:


_ = clf.fit(X_train, Y_train)


# In[ ]:


predictions = clf.predict(X_test) # Making Predictions
predictions


# In[ ]:


accuracy = accuracy_score(Y_test, predictions) # Finding Accuracy
accuracy


# #### Using Support Vector Machines

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


# Setting Hyperparameters
clf_svm = SVC(gamma='auto', kernel='poly') 


# In[ ]:


_ = clf_svm.fit(X_train, Y_train)


# In[ ]:


predictions_svm = clf_svm.predict(X_test)
predictions_svm


# In[ ]:


accuracy_svm = accuracy_score(Y_test,predictions_svm)
accuracy_svm

