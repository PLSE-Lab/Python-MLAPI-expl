#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# In[104]:


from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)


# In[105]:


# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# **Model Tunning**

# In[103]:


from sklearn.metrics import mean_absolute_error
error_test = []
error_train = []
range_plot = [20,30,40,50,100,150,200,500,700,1000]
for i in range_plot:
    my_model2 = XGBRegressor(n_estimators=i, learning_rate=0.05)
    my_model2.fit(train_X, train_y,early_stopping_rounds=1000, eval_set=[(test_X, test_y)], verbose=False)
    predictions2 = my_model2.predict(test_X)
    prediction_training = my_model2.predict(train_X)
    error_train.append(mean_absolute_error(prediction_training,train_y))
    error_test.append(mean_absolute_error(predictions2, test_y))  
print(min(error_test))

import matplotlib.pyplot as plt
plt.plot(range_plot,error_train, label='training')
plt.plot(range_plot,error_test, label='test')
plt.xlabel('n_estimators')
plt.ylabel('Mean_squared_error')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

