#!/usr/bin/env python
# coding: utf-8

# This notbook works completely fine in my pc but is not able to load data here.
# It ran with an accuracy score of 95%

# In[ ]:


# This dataset does not belong to me

from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
# from sklearn.model_selection import train_test_split


# In[ ]:


dataset=pd.read_csv("/kaggle/input/lung-cancer/datasets_85411_197066_lung_cancer_examples.csv")
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


y=dataset["Result"]
y.head()


# In[ ]:


dataset=dataset.drop("Name",axis=1)


# In[ ]:


dataset=dataset.drop("Surname",axis=1)
dataset.head()


# In[ ]:


dataset=dataset.drop("Result",axis=1)


# In[ ]:


x=dataset
x.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y)
x_test.head()


# In[ ]:


reg=linear_model.LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)


# In[ ]:


accuracy_score(y_predict,y_test)


# In[ ]:


confusion_matrix(y_predict,y_test)


# In[ ]:


# Plotting the data
sns.pairplot(dataset,hue="Result")

