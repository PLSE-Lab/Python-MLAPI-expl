#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Load Packages
get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#Load Data
iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


iris.head()


# In[ ]:


iris['Id'] = iris['Id'].astype(str)
iris.describe()


# In[ ]:


#Data cleaning
a =  LabelEncoder()
a1 = a.fit(iris['Species'])
iris_label = a1.transform(iris['Species'])
iris['iris_label'] = iris_label
print(pd.Series(iris_label).unique())
print(iris)


# In[ ]:


label_mapping = dict(zip(pd.Series(iris_label).unique() , iris['Species']))
label_mapping


# In[ ]:


#Train Test Split
X = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['iris_label']

X_train ,X_test , y_train ,  y_test = train_test_split(X,y,random_state=0)


# In[ ]:


from matplotlib import cm

c_map_iris = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c=y_train , marker = 'o' , s = 40 , hist_kwds = {'bins':15} , figsize = (9,9) , cmap=c_map_iris)


# In[ ]:


#training Classifier model
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(X_train , y_train)


# In[ ]:


#Evaluate Accuracy of the Classifier
kNN.score(X_test , y_test)


# In[ ]:


# Predict unknown data using classifier 

predict_iris = kNN.predict([[3,3,3,3]])
label_mapping[predict_iris[0]]

