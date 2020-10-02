#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# For Random Forest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Addolescent Health Data Analysis**

# In[4]:


data = pd.read_csv("../input/addhealth.csv")


# **Exploratory data analysis and preprocessing**

# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


clean_data = data.dropna()


# In[8]:


clean_data.describe()


# In[9]:


data.columns


# **Correlation of Variables**

# In[10]:


correlation = clean_data.corr()

plt.figure(figsize = (14,14))
heatmap = sns.heatmap(correlation, annot = True, linewidths = 0, vmin = -1, cmap = "RdBu_r")


# **Random Forest Classifier**
# 
# What features are the most important in determining whether you will be a smoker or not?

# In[11]:


predictor_list = clean_data[['BIO_SEX', 'HISPANIC', 'WHITE', 'BLACK', 'NAMERICAN', 'ASIAN', 'age',
       'ALCEVR1', 'ALCPROBS1', 'marever1', 'cocever1', 'inhever1',
       'cigavail', 'DEP1', 'ESTEEM1', 'VIOL1', 'PASSIST', 'DEVIANT1',
       'SCHCONN1', 'GPA1', 'EXPEL1', 'FAMCONCT', 'PARACTV', 'PARPRES']]

target = clean_data.TREG1


# In[12]:


pred_train, pred_test, tar_train, tar_test = train_test_split(predictor_list, target, test_size =0.4)


# In[13]:


print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)


# In[14]:


model = RandomForestClassifier(n_estimators = 10)


# In[15]:


classification = model.fit(pred_train, tar_train)


# In[16]:


prediction = model.predict(pred_test)


# In[17]:


print("Confusion Matrix")
print(sklearn.metrics.confusion_matrix(tar_test, prediction))
print("Accuracy")
print(sklearn.metrics.accuracy_score(tar_test, prediction))


# In[19]:


important_features = pd.Series(data = model.feature_importances_, index = predictor_list.columns)


# In[21]:


important_features.sort_values(ascending = True, inplace = True)
print(important_features)


# In[22]:


RND_STATE = 55324

classifier = ExtraTreesClassifier(random_state = RND_STATE)
classifier.fit(pred_train, tar_train)


# In[23]:


print(model.feature_importances_)


# In[24]:


trees = range(25)
accuracy = np.zeros(25)
for idx in range(len(trees)):
    model = RandomForestClassifier(n_estimators = idx + 1, random_state = RND_STATE)
    classifier = model.fit(pred_train, tar_train)
    predictions = model.predict(pred_test)
    accuracy[idx] = accuracy_score(tar_test, predictions)


# In[25]:


plt.cla()
plt.plot(trees, accuracy)
plt.show()

