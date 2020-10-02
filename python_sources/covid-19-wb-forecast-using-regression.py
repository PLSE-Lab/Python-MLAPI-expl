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


# In[ ]:


sld_df = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/complete.csv')
sld_df.head()


# In[ ]:


wbd_df = sld_df[sld_df['Name of State / UT']=='West Bengal']
wbd_df.head()


# In[ ]:


y = np.array(wbd_df['Total Confirmed cases'])
x = np.array(wbd_df['Date'])


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('ggplot')
##x = np.arange(y.shape[0])
x1 = [x[i] for i in range(0,x.shape[0],20)]
print(x1)
plt.xticks(np.arange(len(x1))*20,x1)
plt.bar(x,y)
plt.show()


# In[ ]:


def poly_feat(d1,d2):
    d = np.arange(d1,d2)+1
    ps = [.1,.25,.33,.5,.75,.8,.9,2,3,4,5]
    x_data = np.zeros((d2-d1,len(ps)+1))
    x_data[:,0] = d
    for i,p in enumerate(ps):
        x_data[:,i+1] = d**p
    return x_data


# In[ ]:


X_data = poly_feat(0,y.shape[0])
X_data[:5]


# In[ ]:


from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LinearRegression,Ridge
from keras.models import Model
from keras.layers import LSTM,GRU,SimpleRNN,Input,Dense,Reshape,Multiply,Lambda
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import Huber
import tensorflow as tf
tf.random.set_seed(42)


# In[ ]:


lr = LinearRegression()
lr.fit(X_data,y)


# In[ ]:


y_pred = lr.predict(X_data)


# In[ ]:


x_ax = np.arange(y_pred.shape[0])
plt.plot(x_ax,y_pred,color='blue',label='Prediction')
plt.bar(x_ax,y,color='red',label='Actual')
plt.legend()
plt.show()


# In[ ]:


y.shape


# In[ ]:


def clip(y_pred):
    maxv = max(y_pred)
    maxi = y_pred.index(max(y_pred))
    for i,v in enumerate(y_pred):
        if i>maxi:
            y_pred[i] = maxv
    return y_pred


# In[ ]:


def testing(model,y_part,Ts,T):
    y_pred = list(y_part)
    X_test = poly_feat(Ts,T)
    y_test = model.predict(X_test)
    y_pred = y_pred+list(y_test)
    return y_pred


# In[ ]:


Ts = 0
T = 80
y_part = y[:Ts]
y_pred = testing(lr,y_part,Ts,T)


# # ** Linear Regressor** #

# In[ ]:


y_pred = clip(y_pred)
x_ax1 = np.arange(len(y_pred))
plt.plot(x_ax1,np.array(y_pred),color='blue',label='Prediction')
x_ax2 = np.arange(y.shape[0])
plt.bar(x_ax2,y,color='red',label='Actual')
plt.axvline(x=x_ax2[-1],color='orange')
plt.legend()
plt.show()


# In[ ]:


lr.coef_


# # **Ridge Regressor** #

# In[ ]:


rd = Ridge()
rd.fit(X_data,y)
y_pred = testing(rd,y_part,Ts,T)
y_pred = clip(y_pred)
x_ax1 = np.arange(len(y_pred))
plt.plot(x_ax1,np.array(y_pred),color='blue',label='Prediction')
x_ax2 = np.arange(y.shape[0])
plt.bar(x_ax2,y,color='red',label='Actual')
plt.axvline(x=x_ax2[-1],color='orange')
plt.legend()
plt.show()


# In[ ]:


rd.coef_


# In[ ]:





# # ** Curve Fitting **

# In[ ]:


def sigmoid(x,a,b,c,d):
    return a/(1+np.exp(-c*x+d))


# In[ ]:


from scipy.optimize import curve_fit
X_cf = X_data[:,0]
popt,_ = curve_fit(f=sigmoid,xdata=X_cf,ydata=y)


# In[ ]:


a,b,c,d = popt
y_pred = []
T = 120
for x in range(1,T):
    y_pred.append(sigmoid(x,a,b,c,d))
x_ax1 = np.arange(len(y_pred))
plt.plot(x_ax1,np.array(y_pred),color='blue',label='Prediction')
x_ax2 = np.arange(y.shape[0])
plt.bar(x_ax2,y,color='red',label='Actual')
plt.axvline(x=x_ax2[-1],color='orange')
plt.legend()
plt.show()


# In[ ]:




