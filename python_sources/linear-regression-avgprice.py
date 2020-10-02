#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_cshttps://www.kaggle.com/izatolokin/d/orgesleka/used-cars-database/notebook6c2350f473v)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv')
#autos = pd.read_csv('../input/autos.csv')


# In[ ]:


data.head()


# **The search for correlations.**

# In[ ]:


#
data.corr()


# In[ ]:


data[['km','year','powerPS']].corrwith(data['avgPrice'])


# In[ ]:


#Average values
pd.DataFrame.mean(data, axis=0)


# **Look at the statistics of features in this data set.**

# In[ ]:


data.plot(y='year', kind='hist')


# In[ ]:


data.plot(y='powerPS', kind='hist')


# In[ ]:


data.plot(y='km', kind='hist')


# In[ ]:


data.plot(y='avgPrice', kind='hist', color='yellow')


# **Look at the graph, as the target is dependent on another**

# In[ ]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
for idx, feature in enumerate(data[['km','year','powerPS']]):
    data.plot(feature, "avgPrice", subplots=True, kind="scatter")
    


# In[ ]:


from sklearn.preprocessing import scale
from sklearn.utils import shuffle
data_shuffled = shuffle(data, random_state=123)
X = data[['km','year','powerPS']]
ones = np.ones(len(X))
ones = ones.reshape(len(ones),1)
X = np.hstack((ones,X))
y = np.log(data['avgPrice'])
X_train = X[:-400]
X_test = X[-400:]
y_train = y[:-400]
y_test = y[-400:]


# **Create linear regression object and train the model using the training sets.**

# In[ ]:


from sklearn.linear_model import LassoCV, LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Coefs 0 =", lr.coef_[0])
print("Coefs 1 =", lr.coef_[1])
print("Coefs 2 =", lr.coef_[2])
print("Coefs 3 =", lr.coef_[3])
y_predicted = lr.predict(X_test)


# **Estimation of the model using the mean square error**

# In[ ]:


def mserror(y, y_pred):
    return np.sum(np.square(y - y_pred))/len(y)
mserror(y_test, y_predicted)


# **Some predictions for the average price**

# In[ ]:


powerPS = np.arange(50, 200, 5)
ones = np.ones(len(powerPS))
year = ones.reshape(len(powerPS),1)*2000
km = ones.reshape(len(powerPS),1)*100000
ones = ones.reshape(len(ones),1)
d = np.hstack((ones, km))
d = np.hstack((d, year))
d = np.hstack((d, powerPS.reshape(len(powerPS),1)))


# In[ ]:


plt.scatter(powerPS, np.exp(lr.predict(d)))
plt.title('predicted average price for \n year = 2000 and km = 100000')
plt.xlabel('powerPS')
plt.ylabel('predicted avgPrice')


# In[ ]:


km = np.arange(50000, 200000, 10000)
ones = np.ones(len(km))
year = ones.reshape(len(km),1)*2000
powerPS = ones.reshape(len(km),1)*150
ones = ones.reshape(len(ones),1)
d = np.hstack((ones, km.reshape(len(km),1)))
d = np.hstack((d, year))
d = np.hstack((d, powerPS))


# In[ ]:


plt.scatter(km, np.exp(lr.predict(d)))
plt.title('predicted average price for \nyear = 2000 and powerPS = 150')
plt.xlabel('km')
plt.ylabel('predicted avgPrice')


# In[ ]:


year = np.arange(1991, 2015, 1)
ones = np.ones(len(year))
km = ones.reshape(len(year),1)*100000
powerPS = ones.reshape(len(year),1)*150
ones = ones.reshape(len(ones),1)
d = np.hstack((ones, km))
d = np.hstack((d, year.reshape(len(year),1)))
d = np.hstack((d, powerPS))


# In[ ]:


plt.scatter(year, np.exp(lr.predict(d)))
plt.title('predicted average price for \n km = 10000 and powerPS = 150')
plt.xlabel('year')
plt.ylabel('predicted avgPrice')


# In[ ]:


auto = [1,100000,1991,40] # Features from the first row
np.exp(lr.predict(auto)[0]) # Prediction of the average price (the real value is 648.3158)


# **Dependencies of the  predicted values**

# In[ ]:


p_data = np.hstack((data[['km','year','powerPS']][-400:], np.exp(y_predicted.reshape(len(y_predicted),1))))
df = pd.DataFrame(p_data, columns = ['km','year','powerPS', 'predicted avgPrice'])
for idx, feature in enumerate(df[['km','year','powerPS']]):
    df.plot(feature, "predicted avgPrice", subplots=True, kind="scatter")


# In[ ]:


df.head()

