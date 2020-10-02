#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# Reading the dataset using pandas built-in function 'read_csv'

ds = pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
ds


# In[ ]:


# GETTING INFO OF dataset
ds.info()


# In[ ]:


# Checking whether the dataset have 'nan' values or not

ds.isnull().sum()


# In[ ]:


# Encoding the Class/Target labels from object type to int..!

from sklearn.preprocessing import LabelEncoder

Class = LabelEncoder()
ds['Class_n'] = Class.fit_transform(ds['class'])

ds.drop('class', axis=1, inplace=True)
ds.head()


# # Balancing the DataSet

# In[ ]:


# checking the total values of target labels 
ds['Class_n'].value_counts()


# In[ ]:


galaxy_ds = ds[ds['Class_n'] == 0]                      # have all values that have class/target label as 'Galaxy'
qso_ds = ds[ds['Class_n'] == 1]                         # have all values that have class/target label as 'QSO'
star_ds = ds[ds['Class_n'] == 2]                        # have all values that have class/target label as 'Star'

galaxy_ds = galaxy_ds.sample(qso_ds.shape[0])           # getting any 850 random values from 'galaxy_ds' dataset
star_ds = star_ds.sample(qso_ds.shape[0])               # getting any 850 random values from 'star_ds' dataset

# now we have to append these three datasets
df = qso_ds.append(galaxy_ds, ignore_index=True)        
ds = star_ds.append(df, ignore_index=True)


# In[ ]:


ds.shape


# In[ ]:


ds['Class_n'].value_counts()

# now the dataset is balanced 


# In[ ]:


# spliting the dataset into train and test

from sklearn.model_selection import train_test_split

x = ds.drop('Class_n', axis=1)
y = ds['Class_n']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.24)

x_train.shape , x_test.shape


# # **Removing Constant & Quasi Constant**

# In[ ]:


from sklearn.feature_selection import VarianceThreshold

filter = VarianceThreshold()

x_train = filter.fit_transform(x_train)
x_test = filter.fit_transform(x_test)

x_train.shape , x_test.shape


# # Standardizing the DataSet

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


# converting the labels series into numpy array because it is more faster..!

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[ ]:


# Importing Machine Learning Algorithm 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# In[ ]:


# Decision tree training with accuracy result
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)


# In[ ]:


# Random Forest training with accuracy result
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_test,y_test)


# In[ ]:


# XGBClassifier training with accuracy result
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb.score(x_test,y_test)


# In[ ]:


# Naive byes training with accuracy result
nb = GaussianNB()
nb.fit(x_train,y_train)
nb.score(x_test,y_test)


# In[ ]:


# KNeighborsClassifier training with accuracy result
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.score(x_test,y_test)


# In[ ]:


# SVM training with accuracy result
svm = SVC()
svm.fit(x_train,y_train)
svm.score(x_test,y_test)


# # Confusion matrix

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = xgb.predict(x_test)

acs = accuracy_score(y_test, y_pred)
print('Accuracy Score of XGB Classifier: ', acs)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n',cm)


# In[ ]:


plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')

