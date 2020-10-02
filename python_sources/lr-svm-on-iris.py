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


iris = pd.read_csv(os.path.join(dirname, filename))


# In[ ]:


iris.head()


# In[ ]:


# Various Species Classes are:

iris['species'].value_counts()


# In[ ]:


iris.describe()


# In[ ]:


iris.shape


# ### Removing the third class as we are doing Logistic Regression as a binary classifier

# In[ ]:


iris_df = iris[iris['species'] != 'Iris-virginica']


# In[ ]:


iris_df['species'].value_counts()


# In[ ]:


# Converting categorical value to numeric value

from sklearn import preprocessing


# In[ ]:


flower_class = preprocessing.LabelEncoder()
flower_class.fit(['Iris-setosa', 'Iris-versicolor'])
iris_df['species'] = flower_class.transform(iris_df['species'])


# In[ ]:


iris_df.head()


# In[ ]:


iris_df['species'].unique()


# * <b> We dont neeed any more preprocessing in this case </b>

# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# ### Train-Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

x = iris_df[['sepal_length','sepal_width','petal_length','petal_width']]
y = iris_df['species']


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=3)


# In[ ]:


x_train.shape , x_test.shape


# In[ ]:


y_train.shape , y_test.shape


# In[ ]:


LR = LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)


# In[ ]:


yhat = LR.predict(x_test)
yhat


# In[ ]:


y_test


# In[ ]:


yhat_prob = LR.predict_proba(x_test)
yhat_prob


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,yhat,labels=[0,1])


# In[ ]:


from sklearn.metrics import accuracy_score

score =accuracy_score(y_test,yhat)
score


# # Support Vector Machine

# In[ ]:


iris.head()


# In[ ]:


# Convert categorical values to numeric values


# In[ ]:


iris['species'].value_counts()


# In[ ]:


iris['species'].unique()


# In[ ]:


flowers = preprocessing.LabelEncoder()
flowers.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
iris['species'] = flowers.transform(iris['species'])


# In[ ]:


iris['species'].unique()


# In[ ]:


iris.dtypes


# In[ ]:


# Convert the label types to int

iris[['sepal_length','sepal_width','petal_length','petal_width']] = iris[['sepal_length','sepal_width','petal_length','petal_width']].astype('int')


# In[ ]:


iris.dtypes


# In[ ]:


x = iris[['sepal_length','sepal_width','petal_length','petal_width']]
y = iris['species']


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)


# In[ ]:


x_train.shape , x_test.shape


# In[ ]:


y_train.shape,y_test.shape


# In[ ]:


from sklearn import svm


# In[ ]:


clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
yhat = clf.predict(x_test)


# In[ ]:


yhat


# In[ ]:


confusion_matrix(y_test,yhat)


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[ ]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

