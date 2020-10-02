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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/companydata/companydata.txt",index_col=0)


# In[ ]:


df.head()


# In[ ]:


x=df.drop('TARGET CLASS',axis=1)


# # **Standardize the variables**

# **Standardisation is important in KNN because the prediction of model is based on the distance between the neighbour points so all the features values should be on the standard scale,otherwise it will lead to wrong prediction**

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scal=StandardScaler()


# In[ ]:


scal.fit(x)


# In[ ]:


scale=scal.transform(x)


# In[ ]:


x=pd.DataFrame(scale,columns=df.columns[:-1])


# In[ ]:


x.head()


# In[ ]:


sns.pairplot(df,hue='TARGET CLASS')


# **From the above diagram,the target classes are overlapped and it can be classified using K-nearest neighbour algorithm.**

# # Train test split****

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,df['TARGET CLASS'],test_size=0.30)


# # K-Nearest Neighbour  ****
# **We use this algorithm to classify the two target class**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(xtrain,ytrain)


# In[ ]:


pred=knn.predict(xtest)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
print(confusion_matrix(ytest,pred))


# In[ ]:


print('Classification report')
print(classification_report(ytest,pred))


# # Choosing the K-neighbour****

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


acc=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,df['TARGET CLASS'],cv=10)
    acc.append(score.mean())
    


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,linestyle='dashed',marker='o',markersize=10,markerfacecolor='red')
plt.xlabel('K-neighbor')
plt.ylabel('Accuracy rate')


# Here we can see that that after arouns K>23 the accuracy rate just tends to hover around 0.06-0.05 Let's retrain the model with that and check the classification report!****

# In[ ]:


err=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,df['TARGET CLASS'],cv=10)
    err.append(1-score.mean())
    


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),err,linestyle='dashed',marker='o',markersize=10,markerfacecolor='red')
plt.xlabel('K-neighbor')
plt.ylabel('Error rate')


# From the above figure we can say that after k=23 there is no increase in error.So we can take the value as k=23

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(xtrain,ytrain)
pred = knn.predict(xtest)

print('WITH K=1')
print('\n')
print(confusion_matrix(ytest,pred))
print('\n')
print(classification_report(ytest,pred))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(xtrain,ytrain)
pred = knn.predict(xtest)

print('WITH K=23')
print('\n')
print(confusion_matrix(ytest,pred))
print('\n')
print(classification_report(ytest,pred))


# # # Comparing k=1 and k=23 there will be increase in accuracy****

# In[ ]:




