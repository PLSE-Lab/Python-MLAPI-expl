#!/usr/bin/env python
# coding: utf-8

# ### Kaggle Dataset - US Cars (New and Used)

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os


# In[ ]:


# Read dataset
base_dir = '/kaggle/input/usa-cers-dataset/'
dataset = pd.read_csv(os.path.join(base_dir, 'USA_cars_datasets.csv'))


# In[ ]:


# Explore dataset
dataset.head(10)


# In[ ]:


# Explore dataset
dataset.describe()


# In[ ]:


# Observations
# 1. Price range - 0 to 84900 - Zero price is not correct, we will have to
# handle this later
# 2. Year - 25% quartile in 2016 - Seems that more than 75% cars are new as
# they are from year 2016 and after
# 3. Mileage - 0 to 1017936 - Some cars haven't been driven - New.
# 4. Unnamed: 0 column - Just for index, can be removed.


# #### Variable Identification

# In[ ]:


# Column names
dataset.columns


# In[ ]:


# Column datatypes
dataset.dtypes


# #### Feature Engineering

# In[ ]:


# Number of records with price as zero
dataset.loc[dataset['price'] == 0].count()


# In[ ]:


# Adjusting price where it is zero
# Using median to replace zero price
price_median = dataset.price.median()
dataset['price'].replace(0, price_median, inplace=True)
dataset.describe()


# In[ ]:


# Categorize variables
continuous_vars = {
    'price': dataset.price,
    'mileage': dataset.mileage,
    'year': dataset.year
}

categorical_vars = {
    'brand': dataset.brand, 
    'model': dataset.model,
    'title_status': dataset.title_status, 
    'color': dataset.color,
    'state': dataset.state, 
    'country': dataset.country, 
    'condition': dataset.condition
}
# Misc variables: 'vin', 'lot'


# #### Univariate analysis
# To understand Central tendency and spread of the variable.
# Also to highlight missing and outlier values.

# In[ ]:


# Continuous variables
for name, data in continuous_vars.items():
  plt.hist(data, bins=100, color='purple')
  plt.xlabel(name)
  plt.show()

  plt.boxplot(data)
  plt.xlabel(name)
  plt.show()


# In[ ]:


# Categorical variables
# Plan to use Bar chart and Frequency table
# TBD


# #### Bivariate analysis

# In[ ]:


# Brand and Model
brand_model = dataset.groupby('brand')['model'].count()
brand_model = brand_model.reset_index().sort_values('model', ascending=False)
brand_model = brand_model.rename(columns = {'model':'count'})
fig = px.bar(brand_model, x='brand', y='count', color='count')
fig.show()


# #### Variable correlation - Scatter Plot
# This will help us understand how the continuours variables are being spread out with respect to each other.

# In[ ]:


# Scatter Plot
# Variables to use - price, mileage
sns.pairplot(dataset[['price', 'mileage']],
            kind='scatter',
            diag_kind='auto')
plt.show()


# In[ ]:


# Observations:
# Price and Mileage are inversely related
# Usually, matching with the real-world scenario.


# ### Next steps

# 1. There are no target variables in the dataset, although 'price' could be considered as one.
# 2. With 'price' as target variable, Regression models can be implemented to predict the prices.
# 3. 'Unnamed: 0' feature seems to be an internal index which can be removed for model processing.
# 4. Categorical features (represented by list 'categorical_vars' above) could be encoded for model data.
# 5. PCA and correlation matrix will be useful to find features which may strongly predict 'price' of the car.
# 6. For EDA, more Univariate and Bivariate analysis could be performed.

# I'm expanding my knowledge in data analysis and this notebook is one of the steps in that direction.
# My aim is to submit new notebooks periodically and learn this field with fellow Kagglers.
# 
# I referred the below notebook for help. Thank you for a helpful resource.
# Credits: https://www.kaggle.com/tanersekmen/us-car-data-analysis-eda-visualization
