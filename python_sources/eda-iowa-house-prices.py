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


# ## Data exploration and Cleaning

# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks",color_codes=True)


# In[ ]:


#Reading the file from the directory and creating a dataframe for exploration
iowa_file_path = '/kaggle/input/iowa-house-prices/train.csv'

housing_data = pd.read_csv(iowa_file_path)


# #### 1. Check available features for data exploration

# In[ ]:


housing_data.columns


# #### 2. Checking the dimensions of the dataframe

# In[ ]:


housing_data.shape


# #### 3. Explore what the data looks like

# In[ ]:


housing_data.head()


# As you can see we have missing values/NaN

# #### 4. Couting Missing Data for each feature

# In[ ]:


housing_data.isna().sum()


# #### 5. Select interesting features for exploration and create a separate dataframe

# In[ ]:


feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','SalePrice']


# In[ ]:


housing_data_cleaned =  housing_data[feature_names]
housing_data_cleaned


# #### 6. Check if there are null values from the selected features

# In[ ]:


housing_data_cleaned.isna().sum()


# #### 7. Get a short insight for what is inside those features

# In[ ]:


housing_data_cleaned.describe()


# ## Visualize Data

# ### Pair plot to check relationship between each feature

# In[ ]:


plt.figure(dpi=30)
sns.pairplot(housing_data_cleaned)
plt.show()


# ### Exploring distributions of each feature

# #### 1. Sale Price in 1000 USD

# In[ ]:


plt.figure(dpi=120)
sns.distplot(housing_data_cleaned[['SalePrice']]/1000)
plt.show()


# #### 2. Lot Area in 1000 USD

# In[ ]:


plt.figure(dpi=120)
sns.distplot(housing_data_cleaned[['LotArea']]/1000)
plt.show()


# #### 3. Year built

# In[ ]:


plt.figure(dpi=120)
sns.distplot(housing_data_cleaned[['YearBuilt']])
plt.show()


# ## Exploring Relationships between Price and Features

# #### 1. Relationship between Price and Lot area

# In[ ]:


sns.relplot(x='LotArea',y='SalePrice',hue="YearBuilt",data=housing_data_cleaned)
sns.despine()


# In[ ]:


housing_data_cleaned[['LotArea','SalePrice']].corr()


# With a correlation of **0.263843** we can say that there is a weak positive correlation between Sales Price and Lot Area

# #### 2. Relationship between Price and Year built

# In[ ]:


sns.relplot(x='YearBuilt',y='SalePrice',hue='YearBuilt',data=housing_data_cleaned)
plt.show()


# In[ ]:


housing_data_cleaned[['YearBuilt','SalePrice']].corr()


# With a corralation value of **0.522897** it indicates that there is a strong correlation between Year Built and Price

# ## Predicting Prices base on features

# #### 1. Import Sklearn Packages

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# #### 2. Split data into test and training

# In[ ]:


### Selecting Features
train_data = housing_data_cleaned
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
target = ['SalePrice']

##split into dependent and independent variables
X = train_data[feature_names]
y = train_data[target]
##split into test and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# #### 3. Finding the best leaf node for training

# In[ ]:


from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X,train_y)
    pred_val = model.predict(val_X)
    return mean_absolute_error(val_y,pred_val)
    

mae = 10**100
for max_leaf_nodes in [5,50,500,500]:
    current_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Leaf Node",max_leaf_nodes,"MAE:",current_mae)
    if(mae>current_mae):
        mae = current_mae
        best_leaf_node = max_leaf_nodes

print(best_leaf_node)


# #### 4. Train the model

# In[ ]:


regressor = DecisionTreeRegressor(max_leaf_nodes=best_leaf_node,random_state=1)
regressor.fit(X,y)


# #### 4.Predict prices from test.csv

# In[ ]:


test_file_path ='/kaggle/input/iowa-house-prices/test.csv'
test_data = pd.read_csv(test_file_path, index_col='Id')
test_data.columns


# In[ ]:


test_data = test_data[feature_names]
pred = regressor.predict(test_data)


# In[ ]:


output = pd.DataFrame({'Id':test_data.index,'SalePrice':pred})


# In[ ]:


output.to_csv('submission_final.csv', index= False)


# In[ ]:




