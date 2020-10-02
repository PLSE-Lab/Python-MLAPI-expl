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


# # Load our Data

# In[ ]:


#I loaded my data from home data for ML course
home_data_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(home_data_path)


# In[ ]:


home_data.head()


# In[ ]:


home_data.describe()


# In[ ]:


#checking the columns 
home_data.columns


# # Splitting tha data into train and test By using train_test_split function of Sklearn

# In[ ]:


#import
from sklearn.model_selection import train_test_split
#creating the target data which is called y
y = home_data.SalePrice
#creating features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

#split the data into train and val
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)




# # Specifying Model

# In[ ]:


#import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
home_model = RandomForestRegressor(random_state = 1)


# # Fit the model

# In[ ]:


home_model.fit(train_X,train_y)


# ## Make Predictions

# In[ ]:


home_prediction = home_model.predict(val_X)


# In[ ]:


print(home_prediction)


# ## Let's compare our prediction with the actual test value

# In[ ]:


#importing metrics from sklearn
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(home_prediction,val_y)


# In[ ]:


print(val_mae)


# In[ ]:


y.head()


# In[ ]:





# In[ ]:




