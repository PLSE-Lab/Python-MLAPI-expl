#!/usr/bin/env python
# coding: utf-8

# In[144]:


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# In[30]:


data = pd.read_csv('../input/housing.csv')
data.info()


# In[31]:


print ("Description : \n\n", data.describe())


# In[32]:


data.head()


# In[155]:


data.hist(figsize=(20,15), color = 'green')
plt.show()


# # Check for NULL values

# In[61]:


print('Let\'s check for null values\n')
print(data.isnull().sum())     


# In[62]:


# Droping NaN value


# In[77]:


data = data.dropna(axis=0)
print("\nNew Shape after dropping NULL value : ", data.shape)


# In[78]:


print('Let\'s check for null values\n')
print(data.isnull().sum())  


# # Input Output Data

# In[79]:


# Dropping ['median_house_value', ocean_proximity]
x_data = data.drop(data.columns[[8, 9]], axis = 1)

y_data = data['median_house_value']


# In[80]:


x_data.head()


# In[81]:


y_data.head()


# # Train Test Split

# In[82]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=101)


# # Scaling Data

# In[83]:


scaler = MinMaxScaler()
scaler.fit(x_train)


# In[85]:


x_train = pd.DataFrame(data = scaler.transform(x_train), columns = x_train.columns, index= x_train.index)


# In[86]:


x_train.head()


# In[87]:


x_test = pd.DataFrame(data = scaler.transform(x_test), columns = x_test.columns, index= x_test.index)


# In[88]:


x_test.head()


# # Creating Feature Columns in TensorFlow

# In[90]:


data.columns


# In[110]:


longitude = tf.feature_column.numeric_column('longitude')
latitutde = tf.feature_column.numeric_column('latitude')
age = tf.feature_column.numeric_column('housing_median_age')
rooms = tf.feature_column.numeric_column('total_rooms')
bedroom = tf.feature_column.numeric_column('total_bedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('median_income')


# In[111]:


# Aggregating the feature columns
feat_cols = [longitude, latitutde, age, rooms, bedroom, pop, households, income]


# In[112]:


feat_cols


# # Creating input 

# In[113]:


input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size = 20, num_epochs = 2000, shuffle = True)


# # Creating the model using Deep Neural Nets Regression 

# In[114]:


model = tf.estimator.DNNRegressor(hidden_units = [8, 8, 8, 8, 8], feature_columns = feat_cols)


# # Training model for 50000 steps

# In[116]:


model.train(input_fn = input_func, steps = 50000)


# # Predicting the value

# In[117]:


predict_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size = 20, num_epochs = 1, shuffle = False)


# In[125]:


pred_gen = model.predict(predict_input_func)    


# In[126]:


predictions = list(pred_gen) 


# In[127]:


predictions


# In[128]:


final_y_preds = []

for pred in predictions:
    final_y_preds.append(pred['predictions'])


# In[132]:


final_y_preds


# In[133]:


# Fianl RMSE Value using DNN Regressor


# In[135]:


mean_squared_error(y_test, final_y_preds) ** 0.5


#                          RSME VALUE USING DNN REGRESSOR  = 72474.108532716637
#                 -> I have used Neural Net with 4 hidden layers, each having 8 input values 
#                 -> Model trained using batch_size = 20, num_epochs = 2000 and Steps = 50,000

# # Working on Sklearn Models

# # Random Forest Regressor

# In[140]:


# Training Model
rf_regressor = RandomForestRegressor(n_estimators=500, random_state = 0)
rf_regressor.fit(x_train, y_train)


# In[141]:


# Predicting the values
y_pred = rf_regressor.predict(x_test)


# In[142]:


p = mean_squared_error(y_test, y_pred)
print(p ** 0.5)    


#             RSME VALUE USING 
#                 -> Random Forest Regressor using 500 estimators =  49775.266747
Thanks for having a look. If u like the Deep Neural Network approach, Pls give a LIKE to my kernel :)