#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from fbprophet import Prophet
from pandas import DataFrame
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[ ]:


Avacado = pd.read_csv("../input/avocado-prices/avocado.csv")


# In[ ]:


Avacado.head()


# In[ ]:


Avacado.info()


# In[ ]:


Avacado.describe()


# In[ ]:


df = pd.read_csv("../input/avocado-prices/avocado.csv")
df= df.drop(['Unnamed: 0'], axis = 1)
df.head()


# In[ ]:


Avacado.groupby("year").mean()


# In[ ]:


import seaborn as sns


# In[ ]:


corr = df.corr()


# In[ ]:


plt.figure(figsize = (12,6))
sns.heatmap(corr, cmap = 'coolwarm', annot=True)


# In[ ]:


Avacado1 = pd.read_csv("../input/avocado-prices/avocado.csv")
df = pd.DataFrame(Avacado1)


# In[ ]:


X = df[["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags","Large Bags", "XLarge Bags"]]
Y = df["AveragePrice"]
regr = linear_model.LinearRegression()
regr.fit(X, Y)
coef = (regr.coef_)
print(coef)


# In[ ]:


import random
df1 = pd.DataFrame(Avacado)
X1 = df[["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags","Large Bags", "XLarge Bags"]]
Y1 = df["AveragePrice"]
target = regr.predict(X1)
biases = [random.uniform(0,1) for j in range(len (X1))]


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn import metrics
print('MSE : ', metrics.mean_squared_error(Y1, target))


# In[ ]:


residuals = Y1-target


# In[ ]:




