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


df=pd.read_csv('/kaggle/input/weight-height/weight-height.csv')


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


df.describe()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.hist(df.Height, bins=20, rwidth=0.8)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
plt.show()


# In[ ]:


plt.hist(df.Weight, bins=20, rwidth=0.8)
plt.xlabel('Weight (inches)')
plt.ylabel('Count')
plt.show()


# Removing outliers for male

# In[ ]:


df_male=df[(df['Gender']=='Male')]
df_male.shape


# In[ ]:


df_female=df[(df['Gender']=='Female')]
df_female.shape


# In[ ]:


Q3=df_male.Height.quantile(0.75)
Q1=df_male.Height.quantile(0.25)
max_threshold=Q3+1.5*(Q3-Q1)
min_threshold=Q1-1.5*(Q3-Q1)
df_male=df_male[(df_male['Height']<=max_threshold) &( df_male['Height']>=min_threshold)]
Q3=df_male.Weight.quantile(0.75)
Q1=df_male.Weight.quantile(0.25)
max_threshold=Q3+1.5*(Q3-Q1)
min_threshold=Q1-1.5*(Q3-Q1)
df_male=df_male[(df_male['Weight']<=max_threshold) &( df_male['Weight']>=min_threshold)]
df_male.shape


# In[ ]:


Q3=df_female.Height.quantile(0.75)
Q1=df_female.Height.quantile(0.25)
max_threshold=Q3+1.5*(Q3-Q1)
min_threshold=Q1-1.5*(Q3-Q1)
df_female=df_female[(df_female['Height']<=max_threshold) &( df_female['Height']>=min_threshold)]
Q3=df_female.Weight.quantile(0.75)
Q1=df_female.Weight.quantile(0.25)
max_threshold=Q3+1.5*(Q3-Q1)
min_threshold=Q1-1.5*(Q3-Q1)
df_female=df_female[(df_female['Weight']<=max_threshold) &( df_female['Weight']>=min_threshold)]
df_female.shape


# In[ ]:


df=df_male.append(df_female)
df.shape


# In[ ]:





# In[ ]:


from sklearn.utils import shuffle
df=shuffle(df)
df.head()


# In[ ]:


X=df.drop(labels=['Gender'], axis=1)
y=df['Gender']


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y=lb.fit_transform(y)
y


# In[ ]:


plt.scatter(X.Height, X.Weight,y+1, c=y)
plt.show()


# In[ ]:


'''from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)
X'''


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25)


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
pred=knn.predict(X_test)
score=accuracy_score(y_true=y_test, y_pred=pred)
print(score)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


pred=lr.predict(X_test)
score=accuracy_score(y_true=y_test, y_pred=pred)
print(score)


# In[ ]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
pred=svc.predict(X_test)
score=accuracy_score(y_true=y_test, y_pred=pred)
print(score)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
pred=tree.predict(X_test)
score=accuracy_score(y_true=y_test, y_pred=pred)
print(score)


# In[ ]:


matrix=confusion_matrix(y_true=y_test, y_pred=pred)
matrix


# In[ ]:


f1_score(y_true=y_test, y_pred=pred)


# In[ ]:


pred=lr.predict(X)


# In[ ]:


plt.scatter(X.Height, X.Weight,pred+1, c=pred)
plt.show()

