#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading the data from CSV file

ds = pd.read_csv('/kaggle/input/job-classification-dataset/jobclassinfo2.csv')
ds.head()


# In[ ]:


# Getting info of data

ds.info()


# In[ ]:


# Checking the Null values in DataSet

ds.isnull().sum()


# In[ ]:


# Encoding the object type columns into integar 

from sklearn.preprocessing import LabelEncoder

JobFamilyDescription = LabelEncoder()
JobClassDescription  = LabelEncoder()
PG = LabelEncoder()

ds['JobFamilyDescription_n'] = JobFamilyDescription.fit_transform(ds['JobFamilyDescription'])
ds['JobClassDescription_n'] = JobClassDescription.fit_transform(ds['JobClassDescription'])
ds['PG_n']  = PG.fit_transform(ds['PG'])


# In[ ]:


# Droping the unnecessary columns 

ds.drop(['JobFamilyDescription','JobClassDescription','PG','ID'],axis=1, inplace=True)
ds.tail()


# In[ ]:


# Counting the label values of dataset

ds['PG_n'].value_counts()


# In[ ]:


# Showing the above counted values in Cool style

import seaborn as sns

sns.countplot(ds['PG_n'])
plt.xlabel('Classes')
plt.ylabel("Count")


# In[ ]:


# Habilitating the data

x = ds.drop(['PG_n'],axis=1)
y = ds['PG_n']


# In[ ]:


# Spliting the data into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)


# In[ ]:


# Removing the Constant & and Quasi Constant 

from sklearn.feature_selection import VarianceThreshold
filter = VarianceThreshold()

x_train = filter.fit_transform(x_train)
x_test = filter.transform(x_test)

x_train.shape , x_test.shape


# There are no constant & quasi constant in this dataset


# In[ ]:


# Standarizing or Regularizing the train & test data. So, it will help in training or testing the model.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train[0]


# In[ ]:


# Converting the training & testing labels into numpy arrays..!

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[ ]:


# Importing the Decision Tree Classifier & Naive Bayes Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Trainging the Decision Tree Classifier & print the Accuracy score

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)


# In[ ]:


# Trainging the Naive Bayes Classifier & print the Accuracy score

nb = GaussianNB()
nb.fit(x_train,y_train)
nb.score(x_test,y_test)


# In[ ]:


# Confusion Metrics

from sklearn.metrics import confusion_matrix

y_pred = dt.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix :\n',cm)


# In[ ]:


plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('turth')


# In[ ]:




