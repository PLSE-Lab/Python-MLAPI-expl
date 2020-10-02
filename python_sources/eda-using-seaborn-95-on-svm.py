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


# ## Importing Required Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm


# In[ ]:


iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()


# The column Id is not needed so lets drop it

# In[ ]:


iris=iris.drop('Id',axis=1)
iris.head()


# ##  Visualization using seaborn

# lets findout how many null values are there in the data

# In[ ]:


iris.isnull().count()


# Luckily,this dataset is complete and doesn't have any null values

# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)


# In[ ]:


sns.boxplot(x='Species',y='PetalWidthCm',data=iris)


# Some observations from the above two graphs--
#  - Virginica has the largest flower amongst all,followed by Versicolor,then Setosa come in the end.
#  - Setosa is the only species to have __Outliers__ .

# In[ ]:


sns.catplot(x='Species',y='SepalLengthCm',data=iris ,kind='bar')


# The __Sepal length__ also follows the same trend as petal length and petal width

# In[ ]:


sns.catplot(x='Species',y='SepalWidthCm',data=iris ,kind='bar')


# The sepal width however disrupts the trend by following the reverse trend(not exactly) but it manages to disrupt the continuity of trend 

# Personally I prefer to use the boxplot instead of barplots,since it gives us an idea about outliers in the former case.

# In[ ]:


iris.corr()


# ## Key Observations
# 
# - The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated
# 
# - We will use all the features for training the algorithm and check the accuracy.

# ## Splitting the Data into Test and Train set

# In[ ]:


X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
y


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=3)


# In[ ]:


model = svm.SVC() #select the algorithm
model.fit(X_train,y_train) # we train the algorithm with the training data and the training output
prediction=model.predict(X_val) #now we pass the testing data to the trained algorithm
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_val))#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output


# ### Thank you For Reading!!!

# In[ ]:




