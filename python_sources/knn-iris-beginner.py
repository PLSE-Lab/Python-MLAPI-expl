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


# # **KNN Algorithm**
# KNN is a non-parametric and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution. In other words, the model structure determined from the dataset. This will be very helpful in practice where most of the real world datasets do not follow mathematical theoretical assumptions. Lazy algorithm means it does not need any training data points for model generation. All training data used in the testing phase. This makes training faster and testing phase slower and costlier. Costly testing phase means time and memory. In the worst case, KNN needs more time to scan all data points and scanning all data points will require more memory for storing training data.

# ## Steps for KNN
# 1. Calculate distance
# 2. Find closest neighbors
# 3. Vote for labels

# In[ ]:


df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()


# In[ ]:


df.isnull().sum()# NO missing value


# In[ ]:


df.Species.value_counts()


# In[ ]:


# Converting labels into numerical values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.Species=le.fit_transform(df.Species)


# In[ ]:


#Dividing into features and target
X=df.drop(columns=['Species','Id'])
Y=df.Species


# In[ ]:


#Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=1)


# In[ ]:


X.head()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn=KNeighborsClassifier(n_neighbors=5)

#Fitting the model on train data
knn.fit(x_train,y_train)

#Calculating accuracy score
acc_train=accuracy_score(y_train,knn.predict(x_train))
acc_test=accuracy_score(y_test,knn.predict(x_test))
print(f'accuracy score on train is {acc_train}')
print(f'accuracy score on test is {acc_test}')


# In[ ]:


#Lets see results when n_neighbours is different
acc_train=[]
acc_test=[]
for i in range(2,10):
    print(f'For n_neighbours = {i}')
    knn1=KNeighborsClassifier(n_neighbors=i)
    
    #Fitting the model on train data
    knn1.fit(x_train,y_train)
    
   #Calculating accuracy score
    acc_train.append(accuracy_score(y_train,knn1.predict(x_train)))
    acc_test.append(accuracy_score(y_test,knn1.predict(x_test)))
    print(f'accuracy score on train is {accuracy_score(y_train,knn1.predict(x_train))}')
    print(f'accuracy score on test is {accuracy_score(y_test,knn1.predict(x_test))}')
    print()


# In[ ]:


import matplotlib.pyplot as plt
#plotting test and train score
plt.plot(range(2,10),acc_train,color='g')
plt.plot(range(2,10),acc_test,color='orange')


# ## Pros of KNN
# The training phase of K-nearest neighbor classification is much faster compared to other classification algorithms. There is no need to train a model for generalization, That is why KNN is known as the simple and instance-based learning algorithm. KNN can be useful in case of nonlinear data. It can be used with the regression problem. Output value for the object is computed by the average of k closest neighbors value.

# ## Cons of KNN
# The testing phase of K-nearest neighbor classification is slower and costlier in terms of time and memory. It requires large memory for storing the entire training dataset for prediction. KNN requires scaling of data because KNN uses the Euclidean distance between two data points to find nearest neighbors. Euclidean distance is sensitive to magnitudes. The features with high magnitudes will weight more than features with low magnitudes. KNN also not suitable for large dimensional data.

# In[ ]:




