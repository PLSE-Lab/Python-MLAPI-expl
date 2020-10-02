#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


az = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = 'iso-8859-1')


# In[ ]:


# Number of Dimensions , Shape and data structure
az.ndim


# In[ ]:


az.shape


# In[ ]:


az.head(5)


# In[ ]:


# To determine count , mean , std, min and quartiles
az.describe()


# In[ ]:


#Checking for Null Values
az.isnull().count()


# In[ ]:


az.isnull()


# In[ ]:


#Plotting different variables to see the distribution
import seaborn as sns
plt.figure(figsize=(15,8))
plot1 = sns.barplot(x= 'year', y ='number', data = az)
# As we can see in below graph 2003, 12, 15 and 16 had more forest fires compared to other years


# In[ ]:


plt.figure(figsize=(15,10))
plot2 = sns.barplot(x= 'state', y ='number', data = az)


# In[ ]:


plt.figure(figsize=(15,10))
plot3 = sns.barplot(x= 'month', y ='number', data = az)


# In[ ]:


#Label Encoding for running Correlation and for running analysis
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(az['state'].drop_duplicates())
az['state']=encoder.transform(az['state'])
encoder.fit(az['month'].drop_duplicates())
az['month']=encoder.transform(az['month'])
encoder.fit(az['date'].drop_duplicates())
az['date']=encoder.transform(az['date'])


# In[ ]:


corr = az.corr()
plt.figure(figsize=(15,10))
sns.color_palette("Blues")
sns.heatmap(corr , annot=True , cbar=False)
plt.show()


# In[ ]:


#From the above visualizations we can see that
# 1.Forest Fires are significant in months of FEB, OCT, NOV
# 2.The state which was frequently subject to forest fire is SAO PALO
# 3.Years with most forest fire is 2003, 15,16 and 17 which seems very recent


# In[ ]:


X = az[['state','year','month','date']]
y = az['number']
X_train = X[:-30]
X_test  = X[-30:]
y_train = y[:-30]
y_test  = y[-30:]


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
gbm=GradientBoostingRegressor(n_estimators=100)
gbm.fit(X_train,y_train)
y_pred_gbm = gbm.predict(X_test)
error_gbm = metrics.mean_squared_error(y_test,y_pred_gbm)
print(np.sqrt(error_gbm))


# In[ ]:


gbm.feature_importances_
#From this it can be determined that Year and Month are most significant in predicting the Forest Fires 

