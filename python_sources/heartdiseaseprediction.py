#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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


df=pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.head(5)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df.isnull().sum()/df.shape[0]*100


# In[ ]:


df=df.drop(['education','cigsPerDay','BPMeds'],axis=1)


# In[ ]:


df.info()


# In[ ]:


miss=[v for v in df.columns if df[v].isnull().sum()>0]

df[miss]


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,9))
sns.set()
for i,var in enumerate(miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var],bins=20,kde_kws={'linewidth':5,'color':'r'})
    


# In[ ]:


df2=df.fillna(df.mean())
plt.figure(figsize=(16,9))
sns.set()
for i,var in enumerate(miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var],bins=20,kde_kws={'linewidth':5,'color':'r'})
    sns.distplot(df2[var],bins=20,kde_kws={'linewidth':5,'color':'g'})
    


# In[ ]:


df2.isnull().sum().sum()


# In[ ]:


df2


# In[ ]:


sns.heatmap(df2.isnull())


# In[ ]:


d3=df.fillna(df.median())
plt.figure(figsize=(16,9))
sns.set()
for i,var in enumerate(miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var],bins=20,kde_kws={'linewidth':5,'color':'r'})
    #sns.distplot(df2[var],bins=20,kde_kws={'linewidth':5,'color':'g'})
    sns.distplot(df2[var],bins=20,kde_kws={'linewidth':5,'color':'b'})


# In[ ]:


plt.figure(figsize=(16,9))
sns.set()
for i,var in enumerate(miss):
    plt.figure(figsize=(16,9))
    plt.subplot(3,2,1)
    sns.boxplot(df[var])
    plt.subplot(3,2,2)
    sns.boxplot(df2[var])
    plt.subplot(3,2,3)
    sns.boxplot(d3[var])
    


# In[ ]:


x=d3.drop('TenYearCHD',axis=1)
y=d3['TenYearCHD']


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x)


# In[ ]:


models=[]
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVC',SVC()))


# In[ ]:


results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=10,random_state=7)
    cv_result=cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg=("%s: %f (%f)" % (name,cv_result.mean(),cv_result.std()))
    print(msg)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[ ]:


my_model=LogisticRegression()
my_model.fit(x_train,y_train)
y_pred=my_model.predict(x_test)


# In[ ]:


print('Training Accuracy:- ',my_model.score(x_train,y_train))


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Confusion_matrix:\n',confusion_matrix(y_test,y_pred))
print('Accuracy:- ',accuracy_score(y_test,y_pred)*100,'%')
print('Classification Report:-\n',classification_report(y_test,y_pred))

