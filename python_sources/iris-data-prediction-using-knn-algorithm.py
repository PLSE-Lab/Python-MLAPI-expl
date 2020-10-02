#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import Series,DataFrame
import scipy
from pylab import rcParams
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
print ('Setup Complete')


# In[ ]:


df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")


# In[ ]:


df.shape


# In[ ]:


df.head(5)


# This dataset consists of 150 samples of species of iris flower. There are 50 samples for each, Iris setosa, Iris virginica and Iris versicolor.For each such sample there are four features,  the length and the width of the sepals and petals, in centimeters.

# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# There are no missing values.

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(df.drop('species',axis=1).corr(),annot=True)


# From above heatmap we can see that there is good correlation between petal_width and petal_lenth, sepal_lenth and petal_length and also between sepal_length and petal_width (written in decreasing order of correlation).

# We can also analyze the correlation using pairplots given below. 

# In[ ]:


sns.set(style="ticks")
sns.pairplot(df, hue="species")


# Now, we will use K-Nearest Neighbor Algorithm.
# 

# In **X_prime** variable, we will store independent columns values and in **y** variable, we will store dependent column values.
# 
# We will split the data into train and test data. Here test data is 30% and remaining 70% is train data.

# In[ ]:


X_prime=df.ix[:,(0,1,2,3)].values
y=df.ix[:,4].values
X= preprocessing.scale(X_prime)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3, random_state  = 5)


# In[ ]:


#K-Nearest Neighbours
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy is:',accuracy_score(y_pred,y_test))


# Here, **y_pred** variable is used to store all the predicted values of test data.

# In[ ]:


pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])


# crosstab is used to get information about where our model predicted wrong values.

# Above, we have set our model for n=3 neighbors, now we will check that if we can increase the accuracy of our model by giving different values of **n**.

# In[ ]:


scores=[]
for n in range(1,15):
    model=KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    scores.append(accuracy_score(y_pred,y_test))
    
plt.plot(range(1,15),scores)
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.show()


# Accuracy is much greater at n=6 neighbors as compared to n=3 neighbors, therefore we will build our model at n=6 neighbors.

# In[ ]:


model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy is:',accuracy_score(y_pred,y_test))


# In[ ]:


pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])


# Accuracy of our model is much better than the previous model.
# 
# 
# 
# 
# 
# 
# Thank You.
