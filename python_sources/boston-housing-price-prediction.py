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


import pandas as pd


# In[ ]:


# Importing the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()


# In[ ]:


data = pd.DataFrame(boston.data)


# In[ ]:


data.head()


# In[ ]:


#Adding the feature names to the dataframe
data.columns = boston.feature_names


# In[ ]:


data['PRICE'] = boston.target 


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()
#No null values


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


import seaborn as sns
sns.distplot(data.PRICE)
#checking the distribution of the target variable


# In[ ]:


sns.boxplot(data.PRICE)
#Distribution using box plot


# In[ ]:


correlation = data.corr()
correlation.loc['PRICE']


# In[ ]:


import matplotlib.pyplot as plt
fig,axes = plt.subplots(figsize=(15,12))
sns.heatmap(correlation,square = True,annot = True)


# By looking at the correlation plot LSAT is negatively correlated with -0.75 and RM is positively correlated to the price and PTRATIO is correlated negatively with -0.51

# In[ ]:


plt.figure(figsize = (20,5))
features = ['LSTAT','RM','PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = data[col]
    y = data.PRICE
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')


# In[ ]:


X = data[['LSTAT','RM','PTRATIO']]
y= data.PRICE


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


# Predicting RMSE the Test set results
from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))


# In[ ]:


rmse


# In[ ]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)


# In[ ]:


r2

