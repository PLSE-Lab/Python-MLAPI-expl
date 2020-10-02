#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Loading data
train_df = pd.read_csv('/kaggle/input/ltfs-2/train_fwYjLYX.csv', parse_dates=['application_date'])
test_df = pd.read_csv('/kaggle/input/ltfs-2/test_1eLl9Yf.csv', parse_dates=['application_date'])


# In[ ]:


print(f'Shape of training data is {train_df.shape}')


# In[ ]:


print(f'Shape of testing data is {test_df.shape}')


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# Null values are present in dataframe. For 2nd business segment, the branch_id and zone are null, as they are aggregated over state level.

# In[ ]:


train_df.describe()


# Since we have to calculate no. of cases at country level, case_count = 0, in some cases should not be a problem while caluating MAPE numbers.

# In[ ]:


# Oldest application date in training data
train_df.application_date.min()


# In[ ]:


# Newest application date in training data
train_df.application_date.max()


# In[ ]:


# No. of applications per segment
case_count_by_segment = pd.DataFrame(train_df.groupby(['segment'])['case_count'].agg(sum))
case_count_by_segment['%_Of_Total'] = case_count_by_segment['case_count'].apply (lambda x: x/case_count_by_segment['case_count'].sum()*100)


# In[ ]:


case_count_by_segment


# Total % of cases with segment 1 are low as compared to cases with segment 2. But somehow, they are presented branch wise. Don't understand why!

# In[ ]:


# No. of zones
list(train_df.zone.unique())


# In[ ]:


# List of states
print(f'Total no. of states in the data are {train_df.state.unique().size}')
list(train_df.state.unique())


# In[ ]:


# Case count by date and segment
case_count_by_date_segment = pd.DataFrame(train_df.groupby(['application_date','segment'])['case_count'].agg(sum))
case_count_by_date_segment.reset_index(drop=False,inplace=True)


# In[ ]:


plt.figure(figsize = (16,5))
sns.lineplot(x="application_date",
             y="case_count",
             hue="segment",
             data=case_count_by_date_segment,
             palette=sns.color_palette('hls', n_colors=2))


# There are few spikes for # of cases for business segment 1. Lets' find out why!

# In[ ]:


# Extracting features from application_date
train_df['month'] = train_df['application_date'].dt.month
train_df['day_of_month'] = train_df['application_date'].dt.day
train_df['day_of_week'] = train_df['application_date'].dt.dayofweek
train_df['year'] = train_df['application_date'].dt.year
train_df['year_month'] = train_df.apply(lambda x: str(x['year']) + '_' + str(x['month']), axis=1)


# In[ ]:


train_df.head()


# In[ ]:


# Case count by date and segment
case_count_by_year_month = pd.DataFrame(train_df.groupby(['year_month','segment'])['case_count'].agg(sum))
case_count_by_year_month.reset_index(drop=False,inplace=True)
plt.figure(figsize = (25,6))
sns.lineplot(x="year_month",
             y="case_count",
             hue="segment",
             data=case_count_by_year_month,
             palette=sns.color_palette('hls', n_colors=2))


# In[ ]:


# Case count by date and segment
case_count_by_month = pd.DataFrame(train_df.groupby(['month','segment'])['case_count'].agg(sum))
case_count_by_month.reset_index(drop=False,inplace=True)
plt.figure(figsize = (16,5))
sns.lineplot(x="month",
             y="case_count",
             hue="segment",
             data=case_count_by_month,
             palette=sns.color_palette('hls', n_colors=2))


# Sharp increase from Feb to March and then sharp decrease from March to April ???
# Again increase in applications from Apr to June and then continous decrese till Aug.
# Constant from Aug till Nov and then increase in applications till Dec
# Seg - 1
# Same observations but in small scale.
# Decrease in applications in month of December.

# In[ ]:


# Case count by date and segment
case_count_by_day_of_month = pd.DataFrame(train_df.groupby(['day_of_month','segment'])['case_count'].agg(sum))
case_count_by_day_of_month.reset_index(drop=False,inplace=True)
plt.figure(figsize = (16,5))
sns.lineplot(x="day_of_month",
             y="case_count",
             hue="segment",
             data=case_count_by_day_of_month,
             palette=sns.color_palette('hls', n_colors=2))


# Since salary gets credited till 10th of every month or in the last days of a month, the sharp increase from 10th day of the month and then the sharp decrease from 26th or 27th of the month could you attributed to the fact that retail people opt for loans maybe personal, home or vehical loan.
# Sharp increase in first business segment from 30th of every month. Why???

# In[ ]:


# Case count by date and segment
case_count_by_day_of_week = pd.DataFrame(train_df.groupby(['day_of_week','segment'])['case_count'].agg(sum))
case_count_by_day_of_week.reset_index(drop=False,inplace=True)
plt.figure(figsize = (16,5))
sns.lineplot(x="day_of_week",
             y="case_count",
             hue="segment",
             data=case_count_by_day_of_week,
             palette=sns.color_palette('hls', n_colors=2))


# The drop could be due to the fact that 6th day is saturday, and is usually off across most industries. 

# In[ ]:


train_df.head()


# In[ ]:


train_new_df = pd.DataFrame(train_df.groupby(['segment','month','day_of_month','day_of_week','year'])                            ['case_count'].agg(sum).reset_index(drop=False))


# In[ ]:


train_new_df.head()


# In[ ]:


# Separate the dataframes segment wise
#train_seg_1_df = train_df[train_df['segment']==1].reset_index(drop=True)
#train_seg_2_df = train_df[train_df['segment']==2].reset_index(drop=True)


# In[ ]:


#train_seg_1_df.head()


# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#train_seg_1_df['state'] = le.fit_transform(train_seg_1_df['state'])
#train_seg_1_df['zone'] = le.fit_transform(train_seg_1_df['zone'])


# In[ ]:


#train_seg_1_df.head()


# In[ ]:


categoricals = ['segment','month','day_of_month','day_of_week']
target = train_new_df.pop('case_count')
feat_cols = list(train_new_df.columns)
#remove_cols = ['application_date','segment']
#feat_cols = [cols for cols in feat_cols if cols not in remove_cols]
feat_cols.remove('year')
feat_cols


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'mape'},
            'subsample': 0.25,
            'subsample_freq': 1,
            'learning_rate': 0.3,
            'num_leaves': 20,
            'feature_fraction': 0.9,
            'lambda_l1': 1,  
            'lambda_l2': 1
            }

folds = 4
seed = 555

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

models = []

for train_index, val_index in kf.split(train_new_df, train_new_df['month']):
    train_X = train_new_df[feat_cols].iloc[train_index]
    val_X = train_new_df[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=(lgb_train, lgb_eval),
                early_stopping_rounds=100,
                verbose_eval = 100)
    models.append(gbm)


# In[ ]:


for model in models:
    lgb.plot_importance(model)
    plt.show()


# In[ ]:


test_df.head()


# In[ ]:


test_df['month'] = test_df['application_date'].dt.month
test_df['day_of_month'] = test_df['application_date'].dt.day
test_df['day_of_week'] = test_df['application_date'].dt.dayofweek


# In[ ]:


test_df.shape


# In[ ]:


temp_df = test_df[['id','application_date']]
test_df = test_df[feat_cols]


# In[ ]:


predictions = []
predictions = (sum([model.predict(test_df) for model in models])/folds)


# In[ ]:


len(predictions)


# In[ ]:


test_df = pd.read_csv('/kaggle/input/ltfs-2/test_1eLl9Yf.csv', parse_dates=['application_date'])
test_df['case_count'] = pd.Series(predictions)


# In[ ]:


test_df.to_csv('submission_1.csv', index=False)

