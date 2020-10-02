#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors with Python
# 
# We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.
# 
# 
# In wikipedia description:
# 
# In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
# 
# In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
# k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.
# 
# Both for classification and regression, a useful technique can be used to assign weight to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.
# 
# The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit training step is required.
# 
# 
# Let's grab it and use it!

# ### Let's import libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/breastCancer.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.drop(columns=['Unnamed: 32'],inplace=True)


# In[ ]:


df.drop(columns=['id'],inplace=True)


# In[ ]:


set(df['diagnosis'])


# In[ ]:


df['diagnosis'] =[ 1 if i =='M' else 0 for i in df['diagnosis']]


# # Data Visualization

# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x= 'diagnosis', data=df)


# In[ ]:


df.plot(figsize=(18,8))


# In[ ]:


plt.figure(figsize=(12,7))
df['smoothness_mean'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:


df['perimeter_worst'].hist(color='blue',bins=40,figsize=(8,4))


# We can figure it out better using data visualization.

# ## Standardize the Variables
# 
# The KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(df.drop('diagnosis',axis=1))


# In[ ]:


scaled_features = scaler.transform(df.drop('diagnosis',axis=1))


# In[ ]:


df.columns


# In[ ]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[1:])
df_feat.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['diagnosis'],test_size=0.20,random_state=101)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)
pred


# In[ ]:


df['diagnosis'].head(5)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# ## Choosing a K Value
# 
# Let's go ahead and use the elbow method to pick a good K Value:

# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Here we can see that that after arouns K>11 the error rate just tends to hover around 0.03-0.02 Let's retrain the model with that and check the classification report!

# In[ ]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


# NOW WITH K=11
knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=11')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# As we can see, when we use K=11 the result is better.
# 
# So if it is usefull kernel for you, please vote it :)
