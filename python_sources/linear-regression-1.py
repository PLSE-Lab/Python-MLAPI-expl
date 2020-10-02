#!/usr/bin/env python
# coding: utf-8

# ###  Importing data and libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('../input/california-housing-prices/housing.csv')
data.head(3)


# In[ ]:


data.info()


# to find null values

# In[ ]:


data['total_bedrooms'].isnull().sum()


# fill the NaN values with mean value

# In[ ]:


data['total_bedrooms'][data['total_bedrooms'].isnull()] = np.mean(data['total_bedrooms'])


# In[ ]:


data.info()


# calculating avg room and avg bed room

# In[ ]:


data['avg_rooms'] = data['total_rooms'] / data['households']
data['avg_bedrooms'] = data['total_bedrooms'] / data['households']
data.head(3)


# findnig corelation

# In[ ]:


data.corr()


# calculating population per household

# In[ ]:


data['popu_per_house'] = data['population'] / data['households']


# In[ ]:


data.head(3)


# In[ ]:


data['ocean_proximity'].unique()


# create a columns with the above output

# In[ ]:


data['NEAR BAY'] = 0
data['<1H OCEAN'] = 0
data['INLAND'] = 0
data['NEAR OCEAN'] = 0
data['ISLAND'] = 0


# In[ ]:


data.head(2)


# add the values to the new columns

# In[ ]:


data.loc[data['ocean_proximity'] == 'NEAR BAY','NEAR BAY'] = 1
data.loc[data['ocean_proximity'] == '<1H OCEAN','<1H OCEAN'] = 1
data.loc[data['ocean_proximity'] == 'INLAND','INLAND'] = 1
data.loc[data['ocean_proximity'] == 'ISLAND','ISLAND'] = 1


# In[ ]:


data.head(3)


# ## Appling Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop(['total_rooms','total_bedrooms','households','ocean_proximity','median_house_value'],axis = 1)
y = data['median_house_value']
print(X.shape)
print(y.shape)


# In[ ]:


train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2)
print(train_X.shape)
print(train_y.shape)


# In[ ]:


lnr_clf = LinearRegression()
lnr_clf.fit(np.array(train_X),train_y)


# In[ ]:


import math
def roundUp(x):
    return int(math.ceil(x/100))*100


# In[ ]:


pred = list(map(roundUp,lnr_clf.predict(test_X)))


# In[ ]:


print(pred[:5])
print(test_y[:5])


# calculating root mean square error

# In[ ]:


from sklearn.metrics import mean_squared_error

prediction = lnr_clf.predict(test_X)
mse = mean_squared_error(test_y,prediction)
rmse = np.sqrt(mse)
print(rmse)


# In[ ]:




