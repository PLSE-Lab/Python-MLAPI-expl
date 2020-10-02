#!/usr/bin/env python
# coding: utf-8

# In[52]:


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


# In[53]:



file =pd.read_csv("../input/PlasticSales.csv")
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
file.head()


# ## converting the data into datatime index

# In[54]:


from datetime import datetime
con=file['Month']

file['Month']=pd.to_datetime(file['Month'])
file.set_index('Month', inplace=True)


ts = file['Sales']
ts.head(10)


# ## Now creating autocorrelation plot

# In[55]:


from statsmodels.graphics.tsaplots import plot_acf
sales_diff =file.diff(periods=1)

sales_diff=sales_diff[1:]
sales_diff
plot_acf(sales_diff)


# In[56]:


plt.plot(sales_diff)


# *  Intrigated order of 1 ,one of the parameter of  Arima model

# ## Creating training and testing to use various different models

# In[57]:


x=file.Sales
train=x[0:45]
test=x[44:60]


# ### Using AR model

# In[58]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


# In[59]:


armodel=AR(train)
armodel_fit =armodel.fit()
train.shape
a=armodel_fit.predict(start=45 ,end=60)
plt.plot(test)
plt.plot(a,color="red")


# In[60]:


mean_squared_error(a,test)


# ### We got 15552 as our mean_square_error using AR model

# ## Arime model

# In[61]:


from statsmodels.tsa.arima_model  import ARIMA
modelarima =ARIMA(train,order=(3,1,1))
arimafit =modelarima.fit()
print(arimafit.aic)


# In[62]:


predictarim =arimafit.forecast(steps=16)[0]
plt.plot(predictarim)

predictarim.shape


# In[63]:


plt.plot(test)
mean_squared_error(predictarim,test)


# ## We will compare aic score  among all the models and choose the one that has the least score
# 
# * Now we will find the best combination 

# In[64]:


import sys
import itertools
p=range(0,10)
d=q=range(0,5)
pdf=list(itertools.product(p,d,q))

import warnings
warnings.filterwarnings("ignore")
for allthevalue in pdf:
    try:
      model_arima=ARIMA(train,order=allthevalue)
      fitarimaa=model_arima.fit()
      print(allthevalue,fitarimaa.aic)
    except:
        continue


# In[66]:


# (7,2,2) gave us the least aic value so we will choose it to build the model
from statsmodels.tsa.arima_model  import ARIMA
modelarima1 =ARIMA(train,order=(7,2,2))
arimafit1 =modelarima1.fit()
prediction1 =arimafit1.forecast(steps=16)[0]
print(arimafit1.aic)


# In[ ]:





# In[ ]:





# In[ ]:




