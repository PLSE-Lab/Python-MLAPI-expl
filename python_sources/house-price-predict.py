#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[109]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
import matplotlib.pyplot as plt
from torch.autograd import Variable


# ---

# # Import Raw Data

# In[110]:


import os

print(os.listdir("../input"))


# In[111]:


raw_data = pd.read_csv('../input/train.csv')


# ---

# # Explore Data

# In[112]:


raw_data.describe()


# In[113]:


raw_data.head(10)


# ---

# # Numeric Data

# ### Find Numeric Data Columns

# In[114]:


numeric_colmuns = []
numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == np.int64].index))
numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == np.float64].index))


# In[115]:


numeric_colmuns
for i in numeric_colmuns:
    if 'index' in str(i):
        numeric_colmuns.remove(i)


# #### SalePrice to Last Index

# In[116]:


numeric_colmuns.remove('total_price')
numeric_colmuns.append('total_price')


# #### Remove Id

# numeric_colmuns.remove('Id')

# ### Get Numeric Data

# In[117]:


numeric_data = DataFrame(raw_data, columns=numeric_colmuns)


# #### Explore Numeric Data

# In[118]:


numeric_data.describe()


# In[119]:


numeric_data.head(10)


# ### NAN Data

# In[120]:


nan_columns = np.any(pd.isna(numeric_data), axis = 0)
nan_columns = list(nan_columns[nan_columns == True].index)


# In[121]:


nan_columns


# #### Assume NAN Values as 0

# In[122]:


for i in nan_columns:
    numeric_data[i] = numeric_data[i].fillna(0)


# #### Check NAN Data

# In[123]:


nan_columns = np.any(pd.isna(numeric_data), axis = 0)
nan_columns = list(nan_columns[nan_columns == True].index)


# In[124]:


nan_columns


# ---

# # Linear Regression with Numeric Data

# In[125]:


import torch
import torch.nn as nn


# In[126]:


numeric_x_columns = list(numeric_data.columns)
numeric_x_columns.remove('total_price')
'''
numeric_x_columns.remove('building_material')
numeric_x_columns.remove('building_use')
numeric_x_columns.remove('parking_way')
'''
numeric_y_columns = ['total_price']


# In[127]:


numeric_x_df = DataFrame(numeric_data, columns=numeric_x_columns)
numeric_y_df = DataFrame(numeric_data, columns=numeric_y_columns)


# In[128]:


numeric_x = torch.tensor(numeric_x_df.values, dtype=torch.float)
numeric_y = torch.tensor(numeric_y_df.values, dtype=torch.float)


# #### Check Shape

# ### Normalize Data

# #### Saving Mean, Max, Min for each Columns

# In[129]:


means, maxs, mins = dict(), dict(), dict()


# In[130]:


for col in numeric_data:
    means[col] = numeric_data[col].mean()
    maxs[col] = numeric_data[col].max()
    mins[col] = numeric_data[col].min()


# In[131]:


numeric_data2 = (numeric_data - numeric_data.mean()) / (numeric_data.max() - numeric_data.min())


# In[132]:


nan_columns = np.any(pd.isna(numeric_data), axis = 0)
nan_columns = list(nan_columns[nan_columns == True].index)
print(nan_columns)
for i in nan_columns:
    numeric_data2[i] = numeric_data2[i].fillna(0)
    


# In[133]:


numeric_x_df = DataFrame(numeric_data2, columns=numeric_x_columns)
numeric_y_df = DataFrame(numeric_data2, columns=numeric_y_columns)
numeric_x_df.describe()
#numeric_x_df.head(10)
#numeric_y_df.describe()


# In[134]:


import xgboost as xgb
dtrain = xgb.DMatrix(numeric_x_df, label = numeric_y_df)

params = {"max_depth":6, "eta":0.03}
model = xgb.cv(params, dtrain,  num_boost_round=2000, early_stopping_rounds=500)


# In[ ]:


model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.03) #the params were tuned using xgb.cv
model_xgb.fit(numeric_x_df, numeric_y_df)


# # Test Data

# In[ ]:


raw_test_data = pd.read_csv('../input/test.csv')


# In[ ]:


raw_test_data.describe()


# In[ ]:


raw_test_data.head(10)


# In[ ]:


test_x = DataFrame(raw_test_data,columns=numeric_x_columns)
t_id =  DataFrame(raw_test_data)


# In[ ]:


for col in numeric_x_columns:
    test_x[col].fillna(0)
#test_x=test_x.drop('total_price',axis=1)
test_x.describe()


# In[ ]:


test_x.describe()


# # Normalize

# __NOTE: Normalizing should be based on train data's mean, max, min__

# In[ ]:


for col in test_x.columns:
    test_x[col] = (test_x[col] - means[col]) / (maxs[col] - mins[col])
test_x.describe()


# # Make a Prediction

# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(test_x))
predictions = pd.DataFrame({"xgb":xgb_preds})


# In[ ]:


solution = pd.DataFrame({"building_id":t_id.building_id, "total_price":xgb_preds* (maxs['total_price'] - mins['total_price']) + means['total_price']})
print(solution)
solution.to_csv("./ridge_sol.csv", index = False)


# In[ ]:




