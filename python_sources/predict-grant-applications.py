#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/unimelb/unimelb_training.csv",low_memory=False)
data.head()


# In[ ]:


len(list(data))


# Total 252 variables in the data. 

# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# Remove all columns which have so many numbers of NA.

# In[ ]:


data1 = data.iloc[:,1:40] 
print(type(data1))


# In[ ]:


data1.isna().sum()


# In[ ]:


data2= data1.drop(['Country.of.Birth.1','Sponsor.Code','Grant.Category.Code','Start.date','Contract.Value.Band...see.note.A','Home.Language.1','With.PHD.1','No..of.Years.in.Uni.at.Time.of.Grant.1'],axis =1)


# In[ ]:


data2.shape


# In[ ]:


data2.isna().sum()


# Remove all rows having NA.

# In[ ]:


data3 = data2.dropna()


# In[ ]:


data3.isna().sum()


# In[ ]:


data3.shape


# In[ ]:


data3.head()


# In[ ]:


data3.dtypes


# In[ ]:


data3['Role.1'].unique()


# Convert object column to integer type.

# In[ ]:


data3['Role.1'] = data3['Role.1'].astype('category').cat.codes


# In[ ]:


data3['Grant.Status'].value_counts()


# We can see that classes are balanced.

# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


model = LogisticRegression()
X = data3.drop(['Grant.Status'],axis =1)
Y = data3['Grant.Status']


# In[ ]:


data3.shape


# In[ ]:


len(Y)


# In[ ]:


X.shape


# In[ ]:


X.head()


# In[ ]:


X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size =0.2, random_state=42)


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report 
print(classification_report(Y_test, pred))


# **k Nearest Neighbour**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


print(classification_report(Y_test, y_pred))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


random_forest1 = RandomForestClassifier(n_estimators=15, max_depth=15)


# In[ ]:


random_forest1.fit(X, Y)


# In[ ]:


pred2 = random_forest1.predict(X_test)


# In[ ]:


print(classification_report(Y_test, pred2))


# We can see the accuracy and recall of the model are good. Thus we can consider this model for real time deployment.

# In[ ]:





# In[ ]:




