#!/usr/bin/env python
# coding: utf-8

# The datasets i have attached with this notebook does not belong to me. I have downloaded it from an unknow source.

# The code uploaded in this notebook is tested in py pc and works  fine with 70% accuracry

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt


# In[ ]:


dataset_train=pd.read_csv("/kaggle/input/bostondataset/boston_train.csv")
dataset_test=pd.read_csv("/kaggle/input/bostondataset/boston_test.csv")
dataset_predict=pd.read_csv("/kaggle/input/bostondataset/boston_predict.csv")


# In[ ]:


dataset_train.isnull().sum()
dataset_train=dataset_train.dropna()


# In[ ]:


dataset_train.isnull().sum()


# In[ ]:


dataset_test=dataset_test.dropna()
dataset_test.isnull().sum()


# In[ ]:


x_train=dataset_train.drop('TAX',axis=1)
x_train=x_train.drop('MEDV',axis=1)
y_train=dataset_train['TAX']


# In[ ]:


np.shape(y_train)
print(dataset_train.head())


# In[ ]:


reg=linear_model.LinearRegression()
reg.fit(x_train[:],y_train[:])


# In[ ]:


reg.score(x_train,y_train)


# In[ ]:


x_test=dataset_test.drop('TAX',axis=1)
x_test=x_test.drop('MEDV',axis=1)
y_test=dataset_test['TAX']
reg.score(x_test,y_test)


# In[ ]:


plt.tight_layout(2,1)
plt.scatter(x_train['CRIM'],y_train,c='r')
plt.scatter(x_train['ZN'],y_train,c='b')
plt.scatter(x_train['INDUS'],y_train,c='y')

plt.show()


# In[ ]:


dataset_predict.isnull().sum()

x_predict=dataset_predict.drop("TAX",axis=1)
yy_predict=dataset_predict['TAX']
# x_predict=x_predict.drop("PTRATIO",axis=1)
print(x_predict.head())
y_predict=reg.predict(x_predict)
print(y_predict)


# In[ ]:


plt.scatter(x_predict['ZN'],yy_predict,c='r')
plt.scatter(x_predict['ZN'],y_predict,c='b')
plt.show()

