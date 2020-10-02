#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from datetime import datetime
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/gold-interest-rate-reg/regression test(1).csv')
df


# In[ ]:


df = df.iloc[:,1:] 
df


# In[ ]:


x = df['interest_rate_pec_change'].values.reshape(-1, 1)
y = df['gold_returns'].values.reshape(-1, 1)
x , y


# In[ ]:


model = LinearRegression()
model.fit(x, y)


# In[ ]:


model.coef_, model.intercept_


# In[ ]:


predict_y = model.predict(x)


# In[ ]:


pred = {}
pred['intercept'] = model.intercept_ 
pred['coefficient'] = model.coef_    
pred['predict_value'] = predict_y   


# In[ ]:


plt.title('interest_rate_pec_change vs gold_returns')
plt.xlabel('interest_rate_pec_change')
plt.ylabel('gold_returns')
plt.xlim(-0.02,0.02)
plt.ylim(-0.005,0.005)
plt.scatter(x, y, color = 'green')


# In[ ]:


sns.regplot(x, y, data = df)

plt.xlim(-0.02,0.02)
plt.ylim(-0.005,0.005)
plt.xlabel('interest_rate_pec_change')
plt.ylabel('gold_returns')


# In[ ]:


plt.scatter(x, y, color = 'blue')
plt.plot(x, predict_y, color = 'green', linewidth = 1)
plt.xlim(-0.02,0.02)
plt.ylim(-0.005,0.005)
plt.title('predict interest_rate_pec_change vs gold_returns')
plt.xlabel('interest_rate_pec_change')
plt.ylabel('gold_returns')
plt.show()


# In[ ]:


plt.plot(x, predict_y, color = 'red', linewidth = 4)
plt.title('predict interest_rate_pec_change vs gold_returns')
plt.xlabel('interest_rate_pec_change')
plt.ylabel('gold_returns')

