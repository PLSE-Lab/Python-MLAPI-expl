#!/usr/bin/env python
# coding: utf-8

# Importing the Libraries

# In[ ]:


import pandas as pd
import numpy as np


# Importing the DataSet

# In[ ]:


import os
print(os.listdir("../input/california-housing-prices"))


# In[ ]:


housing_data = pd.read_csv('../input/california-housing-prices/housing.csv')


# In[ ]:


housing_data.info()


# Finding the Columns with NULL Values

# In[ ]:


housing_data.isnull().sum()[housing_data.isnull().sum() > 0]


# Filling the Null Values With Mean Value of the Column

# In[ ]:


housing_data.fillna(value = housing_data['total_bedrooms'].mean(), axis = 1, inplace = True) 


# 
# Converting the 'ocean_proximity' column into FLOAT

# In[ ]:


ocean_proximity = pd.get_dummies(housing_data['ocean_proximity'])


# In[ ]:


housing_data = housing_data.join(ocean_proximity)


# In[ ]:


housing_data.drop(['ocean_proximity'], axis = 1, inplace = True)


# In[ ]:


x = housing_data.drop('median_house_value', axis = 1)


# In[ ]:


y = housing_data['median_house_value']


# Importing Standard Scaler to scale the Input Values

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


housing_data_scaled = scaler.fit_transform(x)


# In[ ]:


housing_data_final = pd.DataFrame(data = housing_data_scaled, columns = [['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']] )


# Importing the LinearRegression to predict the price of the houses

# In[ ]:


from sklearn.linear_model import LinearRegression


# Splitting the Data into Training and Test Set

# In[ ]:


from sklearn.model_selection import train_test_split


# Importing the GridSearchCV to find out the optimal value of the parameters of the model

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


x1 = housing_data_final
y1 = housing_data['median_house_value']


# Splitting the Data into Training and Test Set

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state = 101)


# In[ ]:


param = {'n_jobs' : [ 0.0001, 0.001, 0.001, 0.01]}


# In[ ]:


GridSearch = GridSearchCV(estimator = LinearRegression(), param_grid = param, verbose = 5)


# In[ ]:


GridSearch.fit(x_train, y_train)


# In[ ]:


GridSearch.best_estimator_


# In[ ]:


LinearModel = LinearRegression(n_jobs = 0.0001)


# In[ ]:


LinearModel.fit(x_train, y_train)


# In[ ]:


predictions = LinearModel.predict(x_test)


# Importing Metrics to evaluate the performance of the Model

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


print(mean_absolute_error(y_test, predictions))


# In[ ]:


print(np.sqrt(mean_squared_error(y_test,predictions)))


# In[ ]:




