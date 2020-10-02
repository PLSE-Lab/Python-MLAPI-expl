#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.shape


# In[ ]:



df = df.iloc[:,1:32]
df


# In[ ]:


df.isnull().sum().sum()


# In[ ]:


sns.countplot(df["diagnosis"])


# In[ ]:


df["diagnosis"].replace("M",1,inplace=True)
df["diagnosis"].replace("B",0,inplace=True)

df["diagnosis"].head()


# In[ ]:


df.info()


# In[ ]:


fig,ax = plt.subplots(figsize=(18,10))
sns.heatmap(df.corr())


# In[ ]:


y = df["diagnosis"]
X = df.drop("diagnosis", axis=1)


# In[ ]:


X.shape


# In[ ]:


from sklearn import preprocessing, model_selection, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

knn_clf = KNeighborsClassifier()


knn_clf.fit(X_train,y_train)
display(knn_clf.score(X_test,y_test))

prediction = knn_clf.predict(X_test)

#metrics using test set and train set
print(metrics.classification_report(y_test,prediction))


# In[ ]:


#metrics using cross validation 5 fold
print(model_selection.cross_val_score(knn_clf,X,y,cv=5).mean()*100)
print(model_selection.cross_val_score(knn_clf,X,y,cv=5).std()*100)


# In[ ]:


log_reg = LogisticRegression(max_iter = 10000,penalty="l1",solver="liblinear")
log_reg.fit(X_train,y_train)
display(log_reg.score(X_test,y_test))
prediction = log_reg.predict(X_test)

#metrics using test set and train set
print(metrics.classification_report(y_test,prediction))


# In[ ]:


#metrics using cross validation 5 fold
print(model_selection.cross_val_score(log_reg,X,y,cv=5).mean()*100)
print(model_selection.cross_val_score(log_reg,X,y,cv=5).std()*100)

