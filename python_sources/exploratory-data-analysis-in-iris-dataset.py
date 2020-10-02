#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load data sets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


iris = pd.read_csv('../input/iris.csv')


# In[ ]:


iris.head()


# In[ ]:


iris.shape


# In[ ]:


iris.columns


# In[ ]:


iris.dtypes


# In[ ]:


iris['Species'].describe


# In[ ]:


iris.describe()


# In[ ]:


iris.columns=iris.columns.str.lower()


# In[ ]:


iris.columns


# In[ ]:


iris['sepallengthcm'].plot.box()


# In[ ]:


iris['sepalwidthcm'].plot.box()


# In[ ]:


iris['petallengthcm'].plot.box()


# In[ ]:


iris['petalwidthcm'].plot.box()


# In[ ]:


iris['sepallengthcm'].plot.hist()


# In[ ]:


iris['sepalwidthcm'].plot.hist()


# In[ ]:


# Univariate === Single analysis ==== Categorical ====
iris['species'].value_counts()


# In[ ]:


iris['species'].unique()


# In[ ]:


iris['species'].nunique()


# In[ ]:


iris.T


# In[ ]:


# Bivariate 

iris.corr()


# In[ ]:


# Univariate ----  
iris['species'].value_counts().plot.bar()


# In[ ]:


# Scatter plot two column relation
iris.plot.scatter('petallengthcm','petalwidthcm')


# In[ ]:


# Scatter plot two column relation
iris.plot.scatter('sepallengthcm','sepalwidthcm')


# In[ ]:


# Scatter plot two column relation
iris.plot.scatter('sepallengthcm','petalwidthcm')


# In[ ]:


## checking na values ----

iris.isna().sum()


# In[ ]:


iris['sepalwidthcm'].describe()


# In[ ]:


iris.sepalwidthcm.plot.box()


# In[ ]:


#
iris['sepalwidthcm'].loc[iris['sepalwidthcm']>4] = np.mean(iris['sepalwidthcm'])


# In[ ]:


iris.shape


# In[ ]:


iris.sepalwidthcm


# In[ ]:


iris.sepalwidthcm.plot.box()


# In[ ]:


iris.shape


# In[ ]:


iris.sepalwidthcm.describe()


# In[ ]:


np.mean(iris.sepalwidthcm)


# In[ ]:


np.median(iris.sepalwidthcm)


# In[ ]:


np.var(iris.sepalwidthcm)


# In[ ]:


np.std(iris.sepalwidthcm)


# In[ ]:


iris['species'].mode()


# In[ ]:


cn={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[ ]:


iris['species']=iris.species.map(cn)


# In[ ]:


iris['species'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:





# In[ ]:


# train-test-split   
train , test = train_test_split(iris,test_size=0.2,random_state=0)

print('shape of training data : ',train.shape)
print('shape of testing data',test.shape)


# In[ ]:


# seperate the target and independent variable
train_x = train.drop(columns=['species'],axis=1)
train_y = train['species']
test_x = test.drop(columns=['species'],axis=1)
test_y = test['species']


# In[ ]:


model = LogisticRegression()

model.fit(train_x,train_y)

predict = model.predict(test_x)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))


# In[ ]:


predict
p1=pd.DataFrame(predict,columns=['Predicted'])
p1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




