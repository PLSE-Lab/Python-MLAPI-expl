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


# In[ ]:


train_data = pd.read_csv('/kaggle/input/used-cars-price-prediction/train-data.csv',index_col=None)
test_data = pd.read_csv('/kaggle/input/used-cars-price-prediction/test-data.csv',index_col=None)


# In[ ]:


train_data.drop("New_Price", axis=1, inplace=True)
test_data.drop("New_Price", axis=1, inplace=True)


# In[ ]:


train_data.drop("Unnamed: 0", axis=1, inplace=True)


# In[ ]:


test_data.drop("Unnamed: 0", axis=1, inplace=True)


# In[ ]:


train_y = train_data["Price"]


# In[ ]:


train_data.drop("Price", axis=1, inplace=True)


# In[ ]:



train_data['Transmission'].value_counts()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


full = pd.concat([train_data, test_data], axis=0)
full



# In[ ]:


encoder=OneHotEncoder(sparse=False)
encoder.fit(full[['Name']])


# In[ ]:


column = encoder.get_feature_names(['Name'])
test_X_encoded = pd.DataFrame (encoder.transform(test_data[['Name']]))
test_X_encoded.columns = column
test_data.drop(['Name'] ,axis=1, inplace=True)
test_data_encoded = pd.concat([test_data, test_X_encoded ], axis=1)


# In[ ]:


train_X_encoded = pd.DataFrame (encoder.transform(train_data[['Name']]))
train_X_encoded.columns = column
train_data.drop(['Name'] ,axis=1, inplace=True)
train_data_encoded = pd.concat([train_data, train_X_encoded ], axis=1)


# In[ ]:


encoder=OneHotEncoder(sparse=False)
encoder.fit(full[['Fuel_Type']])


# In[ ]:


column = encoder.get_feature_names(['Fuel_Type'])
test_X_encoded_fuel = pd.DataFrame (encoder.transform(test_data[['Fuel_Type']]))
test_X_encoded_fuel.columns = column
test_data_encoded.drop(['Fuel_Type'] ,axis=1, inplace=True)
test_data_encoded = pd.concat([test_data_encoded, test_X_encoded_fuel ], axis=1)

train_X_encoded_fuel = pd.DataFrame (encoder.transform(train_data[['Fuel_Type']]))
train_X_encoded_fuel.columns = column
train_data_encoded.drop(['Fuel_Type'] ,axis=1, inplace=True)
train_data_encoded = pd.concat([train_data_encoded, train_X_encoded_fuel ], axis=1)


# In[ ]:


encoder=OneHotEncoder(sparse=False)
encoder.fit(full[['Location']])


# In[ ]:


column = encoder.get_feature_names(['Location'])
test_X_encoded_loc = pd.DataFrame (encoder.transform(test_data[['Location']]))
test_X_encoded_loc.columns = column
test_data_encoded.drop(['Location'] ,axis=1, inplace=True)
test_data_encoded = pd.concat([test_data_encoded, test_X_encoded_loc ], axis=1)

train_X_encoded_loc = pd.DataFrame (encoder.transform(train_data[['Location']]))
train_X_encoded_loc.columns = column
train_data_encoded.drop(['Location'] ,axis=1, inplace=True)
train_data_encoded = pd.concat([train_data_encoded, train_X_encoded_loc ], axis=1)


# In[ ]:


replace_map = {'Owner_Type' : {'First' : 1 , 'Second' : 2 , 'Third' : 3 , 'Fourth & Above' : 4}}


# In[ ]:


train_data_encoded.replace(replace_map , inplace = True)


# In[ ]:


test_data_encoded.replace(replace_map , inplace = True)


# In[ ]:


replace_map_transmission = {'Transmission' : {'Manual' : 0 , 'Automatic' : 1 }}


# In[ ]:


train_data_encoded.replace(replace_map_transmission , inplace = True)
test_data_encoded.replace(replace_map_transmission , inplace = True)


# In[ ]:




