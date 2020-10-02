#!/usr/bin/env python
# coding: utf-8

# # M5 Forecasting Challenge
# 

# ## 1. Introduction
# 
# The goal of this notebook is to give a brief overview of M5 Forecasting competition.
# 
# > Note: This is one of the two complementary competitions that together comprise the M5 forecasting challenge.
# 
# 1. **This Competition:** The objective of this competition is to estimate as precisely as possible, the point forecasts of the unit sales of various products sold in the USA by Walmart? 
# 
#     * Metric Used for evalueation: **Weighted Root Mean Squared Scaled Error** (RMSSE)
# 
# 
# 2. **Second Competition:** The objective of this competition in to estimate the uncertainty distribution of the realized values of the above competition.
# 
#     * Metric Used for evalueation: **Weighted Scaled Pinball Loss** (WSPL)

# ## 2. Data Overview
# 
# In the challenge, you are predicting item sales at stores in various locations for two 28-day time periods. Information about the data is found in the M5 Participants Guide.
# 
# ### Files
# * `calendar.csv` - Contains information about the dates on which the products are sold.
# * `sales_train_validation.csv` - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
# * `sample_submission.csv` - The correct format for submissions. Reference the Evaluation tab for more info.
# * `sell_prices.csv` - Contains information about the price of the products sold per store and date.
# 
# > Note: `sales_train_evaluation.csv` not available yet
# * `sales_train_evaluation.csv` - Available once month before competition deadline. Will include sales [d_1 - d_1941]
# 
# 
# https://storage.googleapis.com/kaggle-forum-message-attachments/772349/15032/M5-Competitors-Guide-Final-10-March-2020.pdf

# ## 3. Peek of the input data folder

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# There are 4 data files available for now in this competition

# ## 4. Importing important packages and libraries

# In[ ]:


import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# ## 5. Loading Data

# In[ ]:


INPUT_DIR_PATH = '../input/m5-forecasting-accuracy/'

sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv')
calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')
sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv')
submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv')


# ## 6. Peek of the data
# 
# ### 6.1. Shape of data files

# In[ ]:


print(f'Shape of sell_prices_df is: {sell_prices_df.shape}')
print(f'Shape of calendar_df is: {calendar_df.shape}')
print(f'Shape of sales_train_validation_df is: {sales_train_validation_df.shape}')
print(f'Shape of submission_df is: {submission_df.shape}')


# ### 6.2. Head of data files
# 
# * **sell_prices_df**

# In[ ]:


sell_prices_df.head()


# > Note: sell_prices_df contains information about the price of the products sold per store and date.

# * **calendar_df**

# In[ ]:


calendar_df.head()


# > Note: calender_df contains information about the dates on which the products are sold.

# * **sales_train_validation_df**

# In[ ]:


sales_train_validation_df.head()


# > Note: We are given historic sales data in the `sales_train_validation` dataset.
#     * rows exist in this dataset for days d_1 to d_1913. We are given the department, category, state, and store id of the item.
#     * d_1914 - d_1941 represents the `validation` rows which we will predict in stage 1
#     * d_1942 - d_1969 represents the `evaluation` rows which we will predict for the final competition standings.

# * **submission_df**

# In[ ]:


submission_df.head()


# > Note: Brief overview of submission file
#     * The submission file has 29 columns, col1 for id and the remaining 28 columns represent the 28 forecast days.
#     * Each represent a specific item. This id tells us the item type, state, and store. We don't know what these items are exactly.

# ### 6.3. Profiling of each DataFrame
# 
# For profiling i have used [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) library
# 
# * **sell_prices_df profile**

# In[ ]:


spd_profile = ProfileReport(sell_prices_df, title='sell_prices_df Profiling Report', html={'style':{'full_width':True}})


# In[ ]:


spd_profile.to_file(output_file="spd_profile.html")
spd_profile.to_notebook_iframe()


# > Note: sell_prices_df contains following interesting information:
#     * store_id: there are 10 different store_id, which shows data is collected from 10 different stores
#     * item_id: there are 3049 different item_ids, which shows 3049 different items are collected for forecasting. (Also item_id has lots of unique values that's why it has `HIGH CARDINALITY`)
#     * There are no missing or null values in this dataframe
#     * The corelation heat map shows there is no or 0 corelation between variables.

# > * **calendar_df profile**

# In[ ]:


cd_profile = ProfileReport(calendar_df, title='calendar_df Profiling Report', html={'style':{'full_width':True}})


# In[ ]:


cd_profile.to_file(output_file="cd_profile.html")
cd_profile.to_notebook_iframe()


# > Note: calendar_df contains following interesting information:
#     * date: there are 1969 different dates, which shows this df has data for 1969 different dates.
#     * d: there are 1969 different d values (Also d has lots of unique values that's why it has `HIGH CARDINALITY`)
#     * event_name and event_type columns denotes special promotional events thus event_name_1, event_type_1, event_name_2, event_type_2 contains a lot of missing values.
#     * date features has very corelation and features snap_CA, snap_TX and snap_WI also show some corelation between them.
#     

# > * **sales_train_validation_df profile**

# In[ ]:


# using minimal=True to avoid heavy computation
# stvd_profile = ProfileReport(sales_train_validation_df, title='sales_train_validation_df Profiling Report', html={'style':{'full_width':True}}, minimal=True)


# In[ ]:


# stvd_profile.to_file(output_file="stvd_profile.html")
# stvd_profile.to_notebook_iframe()


# ### 6.4. Plotting sales of 10 random items

# In[ ]:


# selecting 10 random rows from dataframe
stvd10 = sales_train_validation_df.sample(n = 10)
d_cols = [c for c in stvd10.columns if 'd_' in c] # sales data columns
stvd10 = stvd10.set_index('id')[d_cols].T


# In[ ]:


plt.figure(figsize=(40, 40))
plt.subplots_adjust(top=1.2, hspace = 0.8)
for i,item_id in enumerate(list(stvd10.columns)):
    plt.subplot(5, 2, i + 1)
    stvd10[item_id].plot(figsize=(20, 12),
          title=f'{item_id} sales by "d" number',
          color=next(color_cycle))
    plt.grid(False)


# The above plot shows that there is lots of variation in item sales, this effect can be seasonal or due to some particular events on high sale days.

# ## 7. Merging DataFrames 
# 
# #### 7.1. Merging Calendar df with sales_train_validation_df
# 
# Merging only few important columns

# In[ ]:


cal = calendar_df[['d','date','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
       'snap_CA', 'snap_TX', 'snap_WI']]


# In[ ]:


stvd_1 = sales_train_validation_df.set_index('id')[d_cols].T
# rename id column to 'd', to perform merge operation
stvd_1 = stvd_1.reset_index().rename(columns={'index': 'd'})
# merging df cal and sales_train_validation_df on 'd'
stvd_merged = stvd_1.merge(cal, how='left', validate='1:1')


# ## 8. Plotting sales of above 10 items on actual dates.

# In[ ]:


stvd10.head()


# In[ ]:


# rename id column to 'd', to perform merge operation
stvd10 = stvd10.reset_index().rename(columns={'index': 'd'})
stvd10 = stvd10.merge(cal, how='left', validate='1:1')
stvd10_date = stvd10.set_index('date')


# In[ ]:


stvd10_date.head()


# In[ ]:


plt.figure(figsize=(40, 40))
plt.subplots_adjust(top=1.2, hspace = 0.8)
for i,item_id in enumerate(list(stvd10_date.columns[1:11])):
    plt.subplot(5, 2, i + 1)
    stvd10_date[item_id].plot(figsize=(20, 12),
          title=f'{item_id} sales by "d" number',
          color=next(color_cycle))
    plt.tight_layout()
    plt.grid(False)


# - Observations:
#     - It is common to see an item unavailable for a period of time.
#     - Some items only sell 1 or less in a day, making it very hard to predict.
#     - Other items show spikes in their demand (super bowl sunday?) possibly the "events" provided to us could help with these.

# ## 9. Simple Submission
# 
# simply setting last 30 days sales.

# In[ ]:


last_thirty_day_avg_sales = sales_train_validation_df.set_index('id')[d_cols[-30:]].mean(axis=1).to_dict()
fcols = [f for f in submission_df.columns if 'F' in f]
for f in fcols:
    submission_df[f] = submission_df['id'].map(last_thirty_day_avg_sales).fillna(0)
    
submission_df.to_csv('submission.csv', index=False)


# In[ ]:


submission_df.head()


# In[ ]:


## TODO:
# 1. Analyze sales of items by item_types i.e `Hobbies`, `Household`, `Foods`.
# 2. Analyze store wise sale of an item datewise
# 3. Analyze weekly trends or may be monthly.
# 4. Create new features
# 5. Try different models

