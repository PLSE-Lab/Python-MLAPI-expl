#!/usr/bin/env python
# coding: utf-8

# # Forecasting web traffic with Prophet in Python
# ### **Hyungsuk Kang, Sungkyunkwan University**
# #### 2017/07/23
# 
# * **1. Introduction**
# * **2. Data preparation**
#     * 2.1 Load data
#     * 2.2 Check for null and missing values
#     * 2.3 Reshape data to fit the model
# * **3. Forecasting**
#     * 3.1  Blocking outliers
#     * 3.2 Forecasting Growth
# * **4. Prediction and submission**
#     * 4.1 Predict and Submit results
# 

# **# 1. Introduction
# 
# This is a guide for forecasting future web traffic values on previous web traffic dataset provided by Google. I chose to build it with Prophet from facebook because it gives faster result than LSTM and get predictions from patterns of the previous data. First, I will prepare the data (previous web traffic time-series) then I will focus on prediction and processing exceptions(outliers).
# 
# For more information on Prophet, click this link.
# 
# # [Prophet](https://facebookincubator.github.io/prophet/)
# 
# 
# This Notebook follows three main parts:
# 
# * Data preparation
# * Processing data and forecasting
# * Results prediction and submission
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet


# # 2. Data preparation
# ## 2.1 Load data

# In[ ]:


# Load the data
train = pd.read_csv("../input/train_1.csv")
keys = pd.read_csv("../input/key_1.csv")
ss = pd.read_csv("../input/sample_submission_1.csv")


# In[ ]:


train.head()


# In[ ]:


# Drop Page column
X_train = train.drop(['Page'], axis=1)
X_train.head()


# ## 2.2 Check for null and missing values

# In[ ]:


# Check the data
X_train.isnull().any().describe()


# Prophet ignores nan value for calculation, so nan values does not affect predictions.
# For 
# I replace null values into 0 because it may affect the calculation.

# ## 2.3 Reshape data to fit the model

# In[ ]:


y = X_train.as_matrix()[0]
df = pd.DataFrame({ 'ds': X_train.T.index.values, 'y': y})


# # 3. Forecasting
# ## 3.1  Blocking outliers

# Outliers can distort predicting from data points with similar magnitudes.
# To remove them, I supposed data points less than 5 percentile and bigger than 95 percentile as *outliers* and removed them replacing with *None* value.
# 
# ![Outliers](https://facebookincubator.github.io/prophet/static/outliers_files/outliers_4_0.png)

# In[ ]:


# With outliers
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=10)
forecast = m.predict(future)
m.plot(forecast);


# The best way to block outlier is to remove them.

# In[ ]:


# Remove outliers
y = X_train.dropna(0).as_matrix()[0] # Replace NaN to 0 for list comprehension
y = [ None if i >= np.percentile(y, 95) or i <= np.percentile(y, 5) else i for i in y ]
df_na = pd.DataFrame({ 'ds': X_train.T.index.values, 'y': y})


# In[ ]:


# Fit the modal
m = Prophet()
m.fit(df_na)


# In[ ]:


# Show future dates
future = m.make_future_dataframe(periods=10)
future.tail()


# In[ ]:


# Forecast future data
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


# Plot forecaset
m.plot(forecast);


# In[ ]:




