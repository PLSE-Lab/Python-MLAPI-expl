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


# # Load the Data

# In[ ]:


data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')


# # Data Exploration

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data['ocean_proximity'].value_counts()


# # Data Visualization

# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

california_img = mpimg.imread('../input/california-housing-feature-engineering/california.png')
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.show()


# ## Histograms 

# In[ ]:


_ = data.hist(bins=50, figsize=(20,15))


# In[ ]:


data[data['median_house_value'] >= 500001].count()


# # Data Preprocessing

# ## Capped variables 

# // Todo

# ## Missing values

# // Todo

# In[ ]:


clean_data = data.dropna()


# ## Categorical variable: ocean proximity (encode it)

# // Todo

# In[ ]:


clean_data.drop('ocean_proximity', axis='columns', inplace=True)


# In[ ]:


clean_data.head()


# # Feature Engineering

# - Polynomial features
# - Divide 'total_rooms', 'total_bedrooms' by 'households'
# - Find more!! (look at the kernels..)

# # Data Splitting 

# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(clean_data, test_size=0.25, random_state=42)


# In[ ]:


train.info(), test.info()


# ## Data Normalization

# // Todo

# # Training a Model 

# In[ ]:


test.columns


# In[ ]:


features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'households', 'median_income']
target = 'median_house_value'

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# In[ ]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[ ]:


_ = model.fit(X_train, y_train)


# # Predict

# In[ ]:


predictions = model.predict(X_test)


# # Model Scoring 

# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, predictions)


# In[ ]:


from sklearn.metrics import r2_score

r2_score(y_test, predictions)


# In[ ]:


model.score(X_test, y_test)

