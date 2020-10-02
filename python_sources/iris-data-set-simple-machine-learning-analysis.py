#!/usr/bin/env python
# coding: utf-8

# In[82]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[83]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[84]:



from sklearn.datasets import load_iris
iris=load_iris()


# #  Iris dataset is in build in sklearn datasets so we imported the data and named it as 'iris'.
# 
# 

# ![](http://)![](http://)

# 

# ### From the above data we can see that the data isn't recorded in tabularized manner i.e it doesn't have rows and columns. Such kinds of data are called non relational databases.
# ### We can also see 'data' and 'target' as keys and numerical digits as a value.In key 'data' we see 4 numerical value in each single list .In a sequential order it denotes sepal length (cm),sepal width ,petal length,petal width which is also known as feature . In 'target' key  we see digits as 0,1,2 which denotes ('setosa', 'versicolor', 'virginica') type of flower.
# 
# ### The goal is to be able to predit the type of flower given the information regarding it's feature i.e spal length(cm),sepal width.....
# 

# In[88]:


print(iris.feature_names)
print(iris.target_names)


# ### Importing machine learning algorithms like KNeighborsClassifier ,LogisticRegression and also train_test_split

# In[89]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[90]:


print(iris.data.shape)
iris.target.shape


# ### Basically it shows the shape of data as well as of target variable .It shows that we have 150 observation and 4 features in iris.data(x variable) .
# 

# In[91]:


x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.4,random_state=1)


# ### Now we split the data into training and testing so that we can train our model using training data and see how accurately it can predict in the test data which isn't used while building the model

# In[94]:


knn=KNeighborsClassifier(n_neighbors=2)
logreg=LogisticRegression()


# #### In this question we will use two classification model i.e KNeighborsClassifier and LogisticRegression() and see which model performs the best

# In[95]:


logreg.fit(xTrain,yTrain)


# In[96]:


yPred =logreg.predict(xTest)



# In[97]:


print(metrics.accuracy_score(yPred,yTest))


# #### We see that logistic regression give the accuracy of 90% .Now we will see KNighbourClassification to see how it performs.

# ## Kneighborsclassif

# In[98]:


k_range=range(1,26)
a=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(xTrain,yTrain)
    y_pred=knn.predict(xTest)
    a.append(metrics.accuracy_score(y_pred,yTest))


# In[99]:


import matplotlib.pyplot as plt
plt.plot(k_range,a)


# ### From the given graph we can see that the accuracy of the model is high at k=(2 t0 4) and k >5 but we will select k=10 as more it is the standard value used .

# In[100]:


knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(xTrain,yTrain)
y_pred=knn.predict(xTest)
metrics.accuracy_score(y_pred,yTest)


# 

# In[101]:


### 


# ## So we have 98.3 percent of accuracy in the model. It's the first and very basic algorithm .I hope it would have some help for the beginner and I would love to receive some feedback .

# 

# In[ ]:





# In[ ]:





# In[ ]:




