#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
color = sns.color_palette()
sns.set_style('darkgrid')


# In[ ]:


X = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
X.head()


# # Analysing the Total Rent ('total (R)')
# We analyse the index to be predicted to check and ensure that the minimum value of total rent is not zero. (It is not practically possible). We can not do away with zeroes in this index since it can potentially destroy our .
# 

# In[ ]:


X['total (R$)'].describe()


# Luckily there wasn't any row with **zero** or **NaN** as its value.

# # Handling NaN's using imputation
# We check the number of NaNs in each index (if any) and Remove them by using imputation. There are various strategies that can be used for imputation like replacing NaNs with **min**, **mean**, **median**, **max**, **zero**.
# 

# In[ ]:


X.isnull().sum()


# To my surprise, there wasn't any index with NaN's in it. Well... who am I to complain. :p

# # Check dtype of columns
# Then we check the datatype of each index. This will help us read into the data and decide the strategy for preprocessing for each column.

# In[ ]:


for i in X.columns:
    print(i, ':', X[i].dtype) 


# There seems to be discrepancy in the dtype of **floor** column. Floor column tells us about the floor the apartment/house is on therefore it should be of dtype **int64** whereas it is of dtype **object** in the given data.
# Let's go though the values in floor index to get a better hang of it.

# In[ ]:


X['floor'].head(20)


# There are many **"-"** in the data which I think represent the missing values in the column.
# So, we will replace those with "null"/"NaN" and convert the dtype to integer.

# In[ ]:


X['floor'].replace('-',np.NaN, inplace=True)
print("Number of null values - ",X['floor'].isnull().sum(),'\n')
X['floor'] = pd.to_numeric(X['floor'])
print(X['floor'].describe())


# On further looking into "floor" index, the maximum number of floors looked off. Getting a max value of 301 in floor number is alarming given the worlds' tallest buliding (Burj Khalifa) has 163 floors.
# So we will consider this an outlier and hence remove it from the data.

# In[ ]:


import math
X['floor'].replace(301,X['floor'].mean(), inplace=True)
X['floor'].fillna((math.floor(X['floor'].mean())), inplace=True)
print(X['floor'].describe())


# In[ ]:


X['floor'].head(20)


# # Checking correlation between different numerical features and total price
# Correlation essentially helps us determine the relationship between Total price and all other indices i.e it will help us understand which variables affect Total Price more than others. 

# In[ ]:


num_dtypes = X.select_dtypes(exclude='object')
numcorr = num_dtypes.corr()
plt.figure(figsize=(15,1))
sns.heatmap(numcorr.sort_values(by ='total (R$)',ascending=False).head(1),cmap='Blues')


# It can be inferred from the plot that features **floor** and **area** have very less to do with the Total Price so it might be better to drop them out of our features list. 

# # Plotting all the categorical features
# We will first plot all categorical features and try to find their relationship with output.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['animal'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['city'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['furniture'], y=X['total (R$)'])


# # Plotting scatterplots for all the numerical features
# 

# In[ ]:


plt.figure(figsize=(15,5))
sns.scatterplot(X['area'],X['total (R$)'])


# Tere seem to be some outliers in the data. 
# So let's drop the outliers and then see the graph

# In[ ]:


X = X.drop(X[(X['area']>10000) | ((X['total (R$)']>50000))].index)
plt.figure(figsize=(15,5))
sns.scatterplot(X['area'],X['total (R$)'])


# Now, that's more like it.

# In[ ]:


plt.figure(figsize=(15,5))
sns.lineplot(x = X['fire insurance (R$)'],y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.scatterplot(x = X['hoa (R$)'],y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['parking spaces'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['bathroom'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['floor'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = X['rooms'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.lineplot(x = X['rent amount (R$)'], y=X['total (R$)'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.scatterplot(x = X['property tax (R$)'], y=X['total (R$)'])


# # Feature Engineering
# 
# We will add all the fixed costs i.e. property tax, rent amount and fire insurance to calcutalte the total fixed cost.
# In retrospect it significantly improves the performance.

# In[ ]:


X['totprice'] = X['property tax (R$)']+X['rent amount (R$)']+ X['fire insurance (R$)']
plt.figure(figsize=(15,5))
sns.scatterplot(x = X['totprice'], y=X['total (R$)'])


# We add the number of rooms, bathrooms and parking paces to create another feature.
# This slightly improves the performance

# In[ ]:


X['totbhk'] = X['rooms'] + X['bathroom'] + X['parking spaces']
plt.figure(figsize=(15,5))
sns.barplot(x = X['totbhk'], y=X['total (R$)'])


# In[ ]:


y_data = X['total (R$)']
X_data = X.drop(['total (R$)','floor','property tax (R$)','rent amount (R$)','fire insurance (R$)','area','parking spaces','rooms','bathroom'], axis=1) 

X_train,X_valid,y_train,y_valid = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=0)


# In[ ]:


low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique()<15 and X_train[cname].dtype=="object"]


# In[ ]:


X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)


# In[ ]:


my_model = XGBRegressor(n_estimators=2000, learning_rate=0.0099)
my_model.fit(X_train,y_train)


# In[ ]:


predictions = my_model.predict(X_valid)
mae = mean_absolute_error(predictions,y_valid)
print('Mean absolute error is', mae)


# Please leave an upvote if you like the notebook.
# This was already a pretty clean dataset so didnt really require much to be done.
# All inputs and suggestions are most welcome.
# Finally, I'd like to thank the author @rubens jr. for providing us with the dataset.
