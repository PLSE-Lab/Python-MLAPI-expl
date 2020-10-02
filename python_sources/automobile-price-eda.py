#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.getcwd()


# In[ ]:


your_local_path = os.getcwd()


# ###  Import required modules

# In[ ]:


import pandas as pd               
import numpy as np
import pickle

from sklearn.model_selection import train_test_split   #splitting data

from sklearn.linear_model import LinearRegression         #linear regression
from sklearn.metrics.regression import mean_squared_error #error metrics
from sklearn.metrics import mean_absolute_error

import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# ###  Read data

# In[ ]:


automobile_data = pd.read_csv('../input/Automobile_data.csv')


# ### Inspect and explore 

# In[ ]:


automobile_data.head()                               # check the head of the data


# In[ ]:


automobile_data.describe()                        # summary statistucs


# In[ ]:


automobile_data.info()                         


# ### Data cleaning

# #### Replace unwanted symbol "?" with NaN 

# In[ ]:


automobile_data.replace('?', np.nan, inplace=True)


# #### Count the number of missing values in the dataframe

# In[ ]:


# count the number of NaN values in each column
print(automobile_data.isnull().sum())


# #### Remove columns that are 90% empty

# In[ ]:


## remove columns that are 90% empty


thresh = len(automobile_data) * .1
automobile_data.dropna(thresh = thresh, axis = 1, inplace = True)


# Here axis refers to a direction along which aggregation will take place in the matrix, axis=0 refers to row wise aggregation whereas axis=1 refers to column wise aggregation
# When inplace=True is passed, the data is renamed in place .i.e in the same cell itself.

# In[ ]:


# count the number of NaN values in each column
print(automobile_data.isnull().sum())


# ### Data Imputation
# 
# Fill the null value cells using appropriate values of the particular column using aggregation functions such as mean, median or mode. 

# In[ ]:


## Define a function impute_median
def impute_median(series):
    return series.fillna(series.median())

#automobile_data['num-of-doors']=automobile_data['num-of-doors'].transform(impute_median)
automobile_data.bore=automobile_data['bore'].transform(impute_median)
automobile_data.stroke=automobile_data['stroke'].transform(impute_median)
automobile_data.horsepower=automobile_data['horsepower'].transform(impute_median)
automobile_data.price=automobile_data['price'].transform(impute_median)


# In[ ]:


automobile_data['num-of-doors'].fillna(str(automobile_data['num-of-doors'].mode().values[0]),inplace=True)
automobile_data['peak-rpm'].fillna(str(automobile_data['peak-rpm'].mode().values[0]),inplace=True)
automobile_data['normalized-losses'].fillna(str(automobile_data['normalized-losses'].mode().values[0]),inplace=True)


# In[ ]:


# count the number of NaN values in each column
print(automobile_data.isnull().sum())


# In[ ]:


automobile_data.head()


# ###  Data Visualization

# #### Count the number of vehicles by brand

# In[ ]:


automobile_data.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title("Number of vehicles by make")
plt.ylabel('Number of vehicles')
plt.xlabel('Make');


# #### Distribution of sales price of automobiles

# In[ ]:


#histogram
automobile_data['price']=pd.to_numeric(automobile_data['price'],errors='coerce')
sns.distplot(automobile_data['price']);


# The distribution is highly skewed towards the left which implies there are lesser vehicles that have a very high price range.  

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % automobile_data['price'].skew())
print("Kurtosis: %f" % automobile_data['price'].kurt())


# #### Correlation between selected variables
# 
# The heat map produces a correlation plot between variables of the dataframe.

# In[ ]:


plt.figure(figsize=(20,10))
c=automobile_data.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# We observe that price and engine-size is positively correlated, so as **the size of the engine increases price also increases** as illustrated by the **scatter plot** below.

# In[ ]:


sns.lmplot('engine-size', # Horizontal axis
           'price', # Vertical axis
           data=automobile_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="make", # Set color
           palette="Paired",
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size


# We observe that price and city-mpg(mileage) is negatively correlated, so as **the  city-mpg increases price decreases** as illustrated by the **scatter plot** below.

# In[ ]:


sns.lmplot('city-mpg', # Horizontal axis
           'price', # Vertical axis
           data=automobile_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="body-style", # Set color
           palette="Paired",
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size


# We also observe that diesel automobiles have a larger price range as compared to automobiles using gas as a fuel. But there are many outliers amongst gas type vehicles that are highly expensive cars.

# In[ ]:


sns.boxplot(x="fuel-type", y="price",data = automobile_data)

