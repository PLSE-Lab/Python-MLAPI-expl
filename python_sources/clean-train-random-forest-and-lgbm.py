#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# # If we want to use the all dataset (55 m rows)
# I've got some problems when loading the all dataset (it runs out of time), so I found this interesting method in this kernel: https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows

# In[ ]:


TRAIN_PATH = '../input/train.csv'

# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())


# In[ ]:


with open(TRAIN_PATH) as file:
    n_rows = len(file.readlines())

print (f'Exact number of rows: {n_rows}')


# In[ ]:


chunksize = 5_000_000 # 5 million rows at one go. Or try 10 million
total_chunk = n_rows // chunksize + 1
print(f'Chunk size: {chunksize:,}\nTotal chunks required: {total_chunk}')


# In[ ]:


df_list = [] # list to hold the batch dataframe
i=0

for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize):
    
    i = i+1
    # Each chunk is a corresponding dataframe
    print(f'DataFrame Chunk {i:02d}/{total_chunk}')
    
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 


# In[ ]:


# Merge all dataframes into one dataframe
train_df = pd.concat(df_list)

del df_list

train_df.info()


# In[ ]:


display(train_df.tail())


# # Otherwise, let's use only 10 million rows

# In[ ]:


train_df = pd.read_csv("../input/train.csv", nrows = 1000000)
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


# Search for missing values
print(train_df.isnull().sum())


# In[ ]:


print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


# In[ ]:


train_df['fare_amount'].value_counts()


# In[ ]:


train_df['fare_amount'].describe()


# In[ ]:


# Delete the negative values
train_df = train_df.drop(train_df[train_df['fare_amount'] < 0].index, axis=0)
train_df.shape


# In[ ]:


train_df['passenger_count'].describe()


# In[ ]:


# Of course it's impossible that 208 passengers are on a single taxy. Let's delete this row
train_df = train_df.drop(train_df[train_df['passenger_count'] == 208].index, axis=0)


# In[ ]:


# Just curiosity, let's see the correlation map (actually the number of features is small, so nothing very useful here)
sns.set_style('white')
sns.set_context("paper",font_scale=2)
corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
           square=True, linewidths=0.5, cbar_kws={"shrink":0.5})


# In[ ]:


# Convert to datetime, so that we can split them in year, month, date, day and hour
train_df['key'] = pd.to_datetime(train_df['key'])
train_df['pickup_datetime']  = pd.to_datetime(train_df['pickup_datetime'])


# In[ ]:


test_df['pickup_datetime']  = pd.to_datetime(test_df['pickup_datetime'])


# In[ ]:


data = [train_df, test_df]
for i in data: 
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour


# In[ ]:


train_df.columns


# In[ ]:


test_df.columns


# In[ ]:


train_df = train_df.drop(['key','pickup_datetime'], axis = 1)
test_df = test_df.drop(['key', 'pickup_datetime'], axis = 1)


# In[ ]:


# Finally, prepare our data for the training phase
x_train = train_df.iloc[:, train_df.columns != 'fare_amount']
y_train = train_df['fare_amount'].values
x_test = test_df


# In[ ]:


train_df.info()


# ## 1) Random Forest (first model I tried) | Score: 3.69

# In[ ]:


'''from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)'''


# In[ ]:


'''submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = rf_predict
submission.to_csv('rnd_fst.csv', index=False)
submission.head(20)'''


# ## 2) LGBM

# In[ ]:


import lightgbm as lgbm


# In[ ]:


lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse'
}


# In[ ]:


pred_test_y = np.zeros(x_test.shape[0])

train_set = lgbm.Dataset(x_train, y_train, silent=True)

model = lgbm.train(lgbm_params, train_set = train_set, num_boost_round=300)

pred_test_y = model.predict(x_test, num_iteration = model.best_iteration)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = pred_test_y
submission.to_csv('lgbm_submission.csv', index=False)
submission.head(20)

