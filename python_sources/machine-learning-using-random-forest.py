#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
meal_df = pd.read_csv('../input/meal_info.csv')
center_df = pd.read_csv('../input/fulfilment_center_info.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


meal_df.head()


# In[ ]:


center_df.head()


# In[ ]:


train_2 = train_df.merge(meal_df, on='meal_id')
final_train = train_2.merge(center_df, on='center_id')


# In[ ]:


train_3 = test_df.merge(meal_df, on = 'meal_id')
final_test = train_3.merge(center_df, on= 'center_id')


# In[ ]:


final_train.head()


# In[ ]:


final_test.head()


# In[ ]:


y = final_train['num_orders']
#val_y = final_test['num_orders']
features = ['center_id', 'meal_id']
X = final_train[features]
val_X = final_test[features]


# In[ ]:


# to internally test your model on test data
#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# # Define model
# meal_model = RandomForestRegressor(random_state= 0)
# # Fit model
# meal_model.fit(train_X, train_y)


# val_predictions = meal_model.predict(val_X)


# In[ ]:



# Define model
meal_model = RandomForestRegressor(n_estimators = 300, random_state= 0)
# Fit model
meal_model.fit(X, y)

# get predictions on validation data
val_predictions = meal_model.predict(val_X)


# In[ ]:


#print(val_y)


# In[ ]:


print(val_predictions)


# In[ ]:


#print(mean_absolute_error(val_y, val_predictions))


# In[ ]:


my_submission = pd.DataFrame({'id': final_test.id, 'num_orders': val_predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




