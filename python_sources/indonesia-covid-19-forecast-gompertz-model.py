#!/usr/bin/env python
# coding: utf-8

# In this notebook, I'll try to apply Gompertz model for confirmed Covid-19 cases prediction in Indonesia, inspired by [this paper](https://arxiv.org/ftp/arxiv/papers/2003/2003.05447.pdf). First of all, let's import all the needed libraries.
# 
# # Import Library

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/confirmed_acc.csv').iloc[39:]
df['days']= df['date'].map(lambda x : (datetime.strptime(x, '%m/%d/%Y') - datetime.strptime("3/1/2020", '%m/%d/%Y')).days  )
df[['date','days','cases']] #reorder column


# # Gompertz Model
# 
# The formula of Gompertz model is as follows:
# $ Q_t = ae^{be^{-c(t-t_0)}} $
# 
# $ Q_t $ is the cumulative confirmed cases; $ a $ is the predicted maximum of confirmed cases; $ b $ and $ c $ are fitting coefficients; $ t $ is the number of days since the first case; $ t_0 $ is the time when the first case occurred. 
# 
# We'll predict the next 60 days after March 1st.
# 

# In[ ]:


def gompertz(a, c, t, t_0):
    Q = a * np.exp(-np.exp(-c*(t-t_0)))
    return Q

x = list(df['days'])
y = list(df['cases'])


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, test_size=0.1, shuffle=False)

x_test_added = x_test + list(range((max(x_test)+1), 60))

popt, pcov = curve_fit(gompertz, x_train, y_train, method='trf', bounds=([100, 0, 0],[6*max(y_train),0.15, 70]))
a, estimated_c, estimated_t_0 = popt
y_pred = gompertz(a, estimated_c, x_train+x_test_added, estimated_t_0)

y_pred


# # Plot prediction data with real data

# In[ ]:


plt.plot(x_train+x_test_added, y_pred, linewidth=2, label='predict data') 
plt.plot(x, y, linewidth=2, color='r', linestyle='dotted', label='train data')
#plt.plot(x_test, y_test, linewidth=2, color='g', linestyle='dotted', label='test data')
plt.title('prediction vs trained data on covid-19 cases in indonesia\n')
plt.xlabel('days since March 1st 2020')
plt.ylabel('confirmed positive')
plt.legend(loc='upper left')


# In[ ]:


#the prediction made into table
prediksi = pd.DataFrame({'day_pred': x_test_added, 'cases_pred':np.around(y_pred[36:])})
prediksi

