#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set()


# In[ ]:


iris = pd.read_csv('../input/iris.csv')


# In[ ]:


iris.shape


# In[ ]:


iris.head()


# In[ ]:


iris['sepal_length'].plot.hist()


# In[ ]:


iris['sepal_width'].plot.hist()


# In[ ]:


iris['petal_length'].plot.hist()


# In[ ]:


iris['petal_width'].plot.hist()


# In[ ]:


iris['species'].value_counts()


# In[ ]:


iris['species'].value_counts().plot.bar()


# In[ ]:


iris['sepal_width'].plot.box()


# In[ ]:


iris['sepal_length'].plot.box()


# In[ ]:


iris['petal_length'].plot.box()


# In[ ]:


iris['petal_width'].plot.box()


# In[ ]:


iris.plot.scatter('sepal_length','sepal_width')


# In[ ]:


iris.plot.scatter('petal_length','petal_width')


# In[ ]:


iris.corr()


# In[ ]:


iris.isna().sum()


# In[ ]:


##Treatment of outlier


# In[ ]:


iris['sepal_width'].loc[iris['sepal_width']>4] = np.mean(iris['sepal_width'])


# In[ ]:


iris['sepal_width'].plot.box()


# In[ ]:


c ={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[ ]:


iris['species'] =iris['species'].map(c)


# In[ ]:


iris.head()


# In[ ]:


iris['species'].value_counts()


# In[ ]:


# Collecting X and y 
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y= iris['species']


# In[ ]:


iris.columns


# In[ ]:


X.shape,y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


model = LogisticRegression()

model.fit(X_train,y_train)


# In[ ]:


predicted=model.predict(X_test)


# In[ ]:


pd.DataFrame(predicted,columns=['Predictedvalues'])


# In[ ]:


print(accuracy_score(y_test,predicted))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predicted)

