#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


#correlation of columns

f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(), annot = True,ax=ax)
plt.show()


# In[ ]:


sns.countplot("Outcome",data = data)
plt.xlabel("Patient- Normal")
plt.ylabel("mean of BMI")
plt.show()
print(data.BMI.mean())


# In[ ]:


f, ax = plt.subplots(figsize= (12,12))
sns.boxplot(x = "Outcome",y ="Glucose",data = data , ax=ax)
plt.show()


# In[ ]:


sns.countplot("Outcome",data = data)
plt.xlabel("Patient-Normal")
plt.ylabel("Glucose.mean")
plt.show()


# In[ ]:


print("Mean of Glucose:",data.Glucose.mean())


# mean Glucose in patients and normals
# 

# In[ ]:


f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(data[["Glucose","Insulin"]].corr(),annot = True, ax=ax)
plt.show()


# In[ ]:


f ,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data[["Glucose","Age"]].corr(),annot=True,ax=ax)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data[["Insulin","Age"]].corr(),annot=True,ax=ax)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data[["Insulin","Age","Glucose"]].corr(),annot = True,ax=ax)
plt.show()


# So 
# * Insulin hormone seems to decrease with age
# * directly proportional to insulin hormone and glucose

# In[ ]:


hist =data.hist(figsize=(10,10))


# In[ ]:


data_c = data.copy(deep=True)
#replace NaN To 0
data_c[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_c[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[ ]:


#Sum of NaNs
print(data_c.isnull().sum())


# In[ ]:


pairplot=sns.pairplot(data_c, hue = 'Outcome')


# In[ ]:


data.shape


# In[ ]:


X = data.drop('Outcome',axis = 1).values
y = data['Outcome'].values


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42, stratify=y)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

n = np.arange(1,10)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))


for i, k in enumerate(n):
    knn = KNeighborsClassifier(n_neighbors = k)

#Fitting
    knn.fit(X_train,y_train)
    
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)
    


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid_param = {'n_neighbors':np.arange(1,100)}


# In[ ]:


knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,grid_param,cv=5)
knn_cv.fit(X,y)


# In[ ]:


knn_cv.best_score_


# In[ ]:


knn_cv.best_params_

