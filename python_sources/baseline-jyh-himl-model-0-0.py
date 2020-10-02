#!/usr/bin/env python
# coding: utf-8

# # JON'S HiML Competition Baseline Model (v 0.0)
# ## Make Date: 04/05/18
# This is my baseline ML model.  The model was built using scikit's DecisionTreeRegressor ML model.

# In[47]:


#Some initialization procedures:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics import mean_absolute_error

# load in data files
FILE_DIR = '../input/hawaiiml-data'
for f in os.listdir(FILE_DIR):
    print('{0:<30}{1:0.2f}MB'.format(f, 1e-6*os.path.getsize(f'{FILE_DIR}/{f}')))
df_train = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1') #write training data to dataframe
df_test = pd.read_csv(f'{FILE_DIR}/test.csv', encoding='ISO-8859-1') # Read the test data

#ML algorithm initialization
from sklearn.tree import DecisionTreeRegressor
myModel = DecisionTreeRegressor()

#define the error function:
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))


# # Prediction Target
# We want to predict the quantity data field.  
# By convention, we define this target as 'y'

# In[48]:


y = df_train.quantity


# # Define ML Predictors

# ### Here is the list of columns we can choose predictors from. To keep it simple, just select from numeric data types.

# In[49]:


print('Column Names & Data Types: \n', df_train.dtypes)


# In[50]:


ls_mypredictors = ['invoice_id', 'stock_id']
X = df_train[ls_mypredictors]


# # Fit model using predictors

# In[51]:


myModel.fit(X, y)


# # Estimate Model's Accuracy

# In[52]:


predictions = myModel.predict(X)
rmsle(y, predictions)


# # Submit Model's Predictions
# ## First, output model's predictions for test data set:

# In[ ]:


test_X = df_test[ls_mypredictors]
# Use the model to make predictions
predicted_vals = myModel.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_vals)


# ## Next, submit predicted values

# In[25]:


my_submission = pd.DataFrame({'Id': df_test.id, 'quantity': predicted_vals})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




