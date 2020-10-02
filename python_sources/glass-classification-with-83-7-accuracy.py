#!/usr/bin/env python
# coding: utf-8

# # Context
# This is a Glass Identification Data Set from UCI. It contains 10 attributes including id. The response is glass type(discrete 7 values)

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# # Load and Prepare Data

# In[ ]:


data=pd.read_csv('../input/glass/glass.csv')
data.head()


# Check if the data has a null value

# In[ ]:


data.isnull().sum()


# # describe the dataset with some methods 

# In[ ]:


data.info


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.describe()


# # Data Visualization

# In[ ]:


fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data.hist(ax = ax)


# In[ ]:


X=data.drop(['Type'],axis=1)
X.head()


# In[ ]:


Y=data['Type']


# # Spliting Data

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=4,test_size=0.20)


# # Evaluate Algorithms

# # LinearRegression Model

# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
predict=lr.predict(x_test)
print(mean_squared_error(predict,y_test))
print(lr.score(x_test,y_test))


# # KNeighborsClassifier Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)


# In[ ]:


knn_predict=knn.predict(x_test)
print(knn.score(x_test,y_test))
print(mean_squared_error(knn_predict,y_test))


# # DecisionTreeClassifier Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
knn_predict=dt.predict(x_test)
print(dt.score(x_test,y_test))
print(mean_squared_error(knn_predict,y_test))


# # LinearDiscriminantAnalysis Model

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
knn_predict=lda.predict(x_test)
print(lda.score(x_test,y_test))
print(mean_squared_error(knn_predict,y_test))


# # Finaly The most model was LinearDiscriminantAnalysis that has 83.7% accuracy
# 
# we can increase the accuracy be geting more dataset

# In[ ]:




