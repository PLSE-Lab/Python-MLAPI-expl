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


# # For Practice Purpose Only

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv') 


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.columns


# In[ ]:


data['gender'] = data['gender'].replace('M', 0)
data['gender'] = data['gender'].replace('F', 1)


# In[ ]:


data.head()


# In[ ]:


data['ssc_b'] = data['ssc_b'].replace(["Others"], 0) 
data['ssc_b'] = data['ssc_b'].replace(["Central"], 1)


# In[ ]:


data.head()


# In[ ]:


data['hsc_b'] = data['hsc_b'].replace(["Others"], 0) 
data['hsc_b'] = data['hsc_b'].replace(["Central"], 1)


# In[ ]:


data.head()


# In[ ]:


data['hsc_s'] = data['hsc_s'].replace(["Commerce"], 0)
data['hsc_s'] = data['hsc_s'].replace(["Science"], 1) 
data['hsc_s'] = data['hsc_s'].replace(["Arts"], 2) 


# In[ ]:


data.head()


# In[ ]:


data['degree_t'] = data['degree_t'].replace(["Sci&Tech"], 1) 
data['degree_t'] = data['degree_t'].replace(["Comm&Mgmt"], 0) 
data['degree_t'] = data['degree_t'].replace(["Others"], 2)


# In[ ]:


data.head()


# In[ ]:


data['specialisation'] = data['specialisation'].replace(["Mkt&HR"], 1) 
data['specialisation'] = data['specialisation'].replace(["Mkt&Fin"], 0) 


# In[ ]:


data['workex'] = data['workex'].replace(["Yes"], 1) 
data['workex'] = data['workex'].replace(["No"], 0)


# In[ ]:


data['status'] = data['status'].replace(["Placed"], 1)
data['status'] = data['status'].replace(["Not Placed"], 0)


# In[ ]:


data


# In[ ]:


def impute_salary(cols):
    sal = cols[0]
    status = cols[1]
    
    if pd .isnull(sal):
        
        if status == 0:
            return 0.0
    else:
        return sal


# In[ ]:


data['salary'] = data[['salary', 'status']].apply(impute_salary, axis=1)


# In[ ]:


data.isnull().sum()


# # We can also see null values with graphs

# In[ ]:


sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Specify the Model and Fit it

# In[ ]:


y = data[['status', 'salary']]


# In[ ]:


feature_list = ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex']


# In[ ]:


X = data[feature_list]


# In[ ]:


# Divide Data using validation and training data
train_X, val_X, train_y, val_y = tts(X, y, random_state=1)


# # First use Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor as dtr


# In[ ]:


model1 = dtr()


# In[ ]:


model1.fit(train_X, train_y)


# In[ ]:


predict1 = model1.predict(val_X)


# In[ ]:


print(mae(val_y, predict1))


# # Now use Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor as rfr


# In[ ]:


model2 = rfr()


# In[ ]:


model2.fit(train_X, train_y)


# In[ ]:


predict2 = model2.predict(val_X)


# In[ ]:


print(mae(val_y, predict2))

