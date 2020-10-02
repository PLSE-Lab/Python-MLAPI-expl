#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sea 
from sklearn import svm
from sklearn.datasets import load_iris


# In[ ]:


iris=load_iris()


# In[ ]:


dir(iris)


# In[ ]:


iris.feature_names


# In[ ]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[ ]:


iris.target_names


# In[ ]:


df['target']=iris.target
df.head()


# In[ ]:


df[df.target==1].head()


# In[ ]:


df[df.target==2].head()


# In[ ]:


df.tail()


# In[ ]:


df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df.head()


# In[ ]:


df.tail()


# In[ ]:


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]


# In[ ]:


plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='red',marker='*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='*')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='green',marker='*')
#we can't apply svm between df1 and df2 therefore we'll apply svm between df0 and df1.


# In[ ]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='red',marker='*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='*')


# In[ ]:


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='red',marker='*')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='*')


# In[ ]:


from sklearn.model_selection import train_test_split #we're splitting the data set  into training and test set


# In[ ]:


X=df.drop(['target','flower_name'],axis='columns')
X.head()


# In[ ]:


Y=df.target


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


svc=svm.SVC(gamma='auto')


# In[ ]:


svc.fit(X_train,Y_train)


# In[ ]:


svc.score(X_test,Y_test)

