#!/usr/bin/env python
# coding: utf-8

# ## ASHRAE - Great Energy Predictor III
# 
# This Notebook is divided into couple of sections which we are going to discuss futher. This will give us an edge for understanding the dataset better and perform the feature engineering later
# 
# <b>Section I. Import dataset
# 
# Section II. Descriptive Stats
# 
# Section III. Train Test Comparison
# 
# Section IV. Normalization and Merge
# 
# Section V. Exploration of Null values
# 
# Section VI. Univariate Distribution
# 
# Section VII. Bivariate Distribution
# 
# Section VIII.Understanding seasonality
# 
# Section IX. Feature Normalization
# 
# Section X. Model Fitting
# 
# Section XI. Model Evaluation & Feature Importance
# 
# </b>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Let's see the discribution of target in train data
from matplotlib import pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")


# In[ ]:


train.head()


# In[ ]:


# Give an overall definition of the dataframe
def define_df(df):
    tup = df.shape
    print("Number of rows in dataframe: {0} & cols: {1}".format(tup[0], tup[1]))
    # Get number of null values in dataframe
    print("-----------Number of Null values-----------")
    print(df.isna().sum())
    #Descriptive stats
    print("-----------Descriptive Stats-----------")
    print(df.describe())
    #Total memory consumption
    print("-----------Data Types + Memory Consumption-----------")
    print(df.info())


# In[ ]:


define_df(train)


# In[ ]:


building_metadata = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")


# In[ ]:


building_metadata.head()


# In[ ]:


define_df(building_metadata)


# In[ ]:


weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")


# In[ ]:


weather_train.head()


# In[ ]:


define_df(weather_train)


# In[ ]:


# Let's use the usual code to minimize the size of dataframe
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        # Custom implementation for this dataset
        if col_type != object and col_type != np.datetime64 and col != 'timestamp':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int': #Encode with the most relevant datatype.g
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


from time import time


# In[ ]:


get_ipython().run_line_magic('time', '')
train = reduce_mem_usage(train)


# In[ ]:


get_ipython().run_line_magic('time', '')
building_metadata = reduce_mem_usage(building_metadata)


# In[ ]:


get_ipython().run_line_magic('time', '')
weather_train = reduce_mem_usage(weather_train)


# For a particular train row to match with its building metadata we have to understand the nuances of weather conditions at a particular time. It's entirely possible that due to some extreme weather the meters stopped working which can be captured by timestamp in weather date.
# 
# Processing date -- Get day, week, month and year of the energy consumption. This will help us in understanding if there is any seasonality within the data.

# In[ ]:


def process_date(df):
    # Thanks to the solution provided at https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column
    # This made my code way shorter
    lst = ['month', 'day', 'hour', 'dayofweek']
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df = df.join(pd.concat((getattr(df.timestamp.dt, i).rename(i) for i in lst), axis=1))
    
    return df


# In[ ]:


# Now merge the data
temp = pd.merge(building_metadata, weather_train, on = 'site_id', how = 'inner')
train = pd.merge(train, temp, on=['building_id', 'timestamp'], how='left')


# In[ ]:


# reduce memory usage
get_ipython().run_line_magic('time', '')
train = reduce_mem_usage(train)


# In[ ]:


del weather_train


# In[ ]:


import gc
gc.collect()


# In[ ]:


train = process_date(train)


# In[ ]:


# reduce memory usage
get_ipython().run_line_magic('time', '')
train = reduce_mem_usage(train)


# The 99.1 percentile of target i.e. meter_reading is at 5449. Thus, the target follows a pareto distribution. I will plot 2 distribution of meter reading which gives us an idea of how it looks like.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.distplot(train.meter_reading, ax=ax[0])
sns.distplot(train[train.meter_reading < 5450].meter_reading, ax=ax[1])


# There are zeros in the target which have multiple theories. Let's try to understand how many buildings in each site id have monthly meter reading as ~0.

# In[ ]:


def process_mr(reading):
    if reading < 1:
        return 1
    elif reading >= 1 and reading < 10:
        return 2
    elif reading >= 10 and reading < 100:
        return 3
    elif reading >= 100 and reading < 1000:
        return 4
    elif reading >= 1000 and reading < 5450:
        return 5
    else:
        return 6
temp = train.groupby(['site_id', 'building_id', 'month']).agg({'meter_reading':np.mean}).reset_index()
temp['groups'] = temp.meter_reading.apply(lambda x: process_mr(x))
temp = temp.groupby(['site_id', 'month', 'groups']).agg({'building_id': 'count'}).reset_index()


# Thanks to for some of the great plot at this kernel [https://www.kaggle.com/nroman/eda-for-ashrae]. I have used reused some of the codebase over in my kernel too.

# In[ ]:


for site in temp.site_id.unique():
    t = temp[temp.site_id == site]
    fig, ax = plt.subplots(1,2, figsize=(16,4))
    sns.heatmap(data=t.pivot_table(index='groups', columns='month', values='building_id'), ax=ax[0], cmap="YlGnBu")
    ax[0].set_title('Month vs Group - site_id: {}'.format(site))
    train[train['site_id'] == site][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=ax[1], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);
    train[train['site_id'] == site][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=ax[1], alpha=1, label='By day', color='tab:orange').set_xlabel('');
    ax[1].set_title('Train Month vs Meter reading - site_id: {}'.format(site))
    plt.show()


# This is quite fascinating with some useful insights. 
# 1. Group 6 i.e. with meter reading > 5450  is present in only few of the site id namely: 1,6,10,13,14.
# 2. There is been a lot of talk recently of how site id 0 having meter reading ~0 for first few months for most of the buildings and this is evident from the visualization where almost 90% of buildings have < 1 meter reading.
# 3. The groups are not equally distributed which means all site ids don't have equal distribution for the meter reading.
# 
# I am going to ignore the readings for point site id 0 for first 4 months. This may be a instrumental error which we have to look more closely into.

# #### Let's understand the types of meter and their respective reading. As defined in data description,we have 4 types: {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}. Let's plot an overall distribution for it.

# In[ ]:


test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")
weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")
get_ipython().run_line_magic('time', '')
test =reduce_mem_usage(test)


# In[ ]:


test.drop(columns=['row_id'], inplace=True)
# Now merge the data
temp = pd.merge(building_metadata, weather_test, on = 'site_id', how = 'inner')
test = pd.merge(test, temp, on=['building_id', 'timestamp'], how='left')
# reduce memory usage
get_ipython().run_line_magic('time', '')
test = reduce_mem_usage(test)


# In[ ]:


test = process_date(test)
test.drop(columns=['timestamp'], inplace=True)
# reduce memory usage
get_ipython().run_line_magic('time', '')
test = reduce_mem_usage(test)


# In[ ]:


temp = pd.DataFrame(train.meter.value_counts()).reset_index()
temp2 = pd.DataFrame(test.meter.value_counts()).reset_index()
temp.columns = ['meter', 'Counts']
temp2.columns = ['meter', 'Counts']
trace1 = go.Bar(x = temp.meter, y = temp.Counts, name='Train')
trace2 = go.Bar(x = temp2.meter, y = temp2.Counts, name='Test')
data = [trace1, trace2]
layout = go.Layout(
    title = "Meter Type Distribution",
    xaxis = dict(
        title = "Meter Type",
        tickfont = dict(
        color = 'rgb(107,107,107)'
        )
    ),
    yaxis = dict(
        title = "Counts",
        titlefont = dict(
        color = 'rgb(107,107,107)'
        ),
        tickfont=dict(
        color = 'rgb(107,107,107)'
        )
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig,image_height=12,image_width=15, filename = 'Meter Type Dist')


# In[ ]:


temp = pd.DataFrame(train.primary_use.value_counts()).reset_index()
temp2 = pd.DataFrame(test.primary_use.value_counts()).reset_index()
temp.columns = ['primary_use', 'Counts']
temp2.columns = ['primary_use', 'Counts']
trace1 = go.Bar(x = temp.primary_use, y = temp.Counts, name='Train')
trace2 = go.Bar(x = temp2.primary_use, y = temp2.Counts, name='Test')
data = [trace1, trace2]
layout = go.Layout(
    title = "Primary Use Distribution",
    xaxis = dict(
        title = "Primary Use",
        tickfont = dict(
        color = 'rgb(107,107,107)'
        )
    ),
    yaxis = dict(
        title = "Counts",
        titlefont = dict(
        color = 'rgb(107,107,107)'
        ),
        tickfont=dict(
        color = 'rgb(107,107,107)'
        )
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig,image_height=12,image_width=15, filename = 'Primary Use Dist')


# In[ ]:


temp = pd.DataFrame(train.year_built.value_counts()).reset_index()
temp2 = pd.DataFrame(test.year_built.value_counts()).reset_index()
temp.columns = ['year_built', 'Counts']
temp2.columns = ['year_built', 'Counts']
trace1 = go.Bar(x = temp.year_built, y = temp.Counts, name='Train')
trace2 = go.Bar(x = temp2.year_built, y = temp2.Counts, name='Test')
data = [trace1, trace2]
layout = go.Layout(
    title = "Year Built Distribution",
    xaxis = dict(
        title = "Year Built",
        tickfont = dict(
        color = 'rgb(107,107,107)'
        )
    ),
    yaxis = dict(
        title = "Counts",
        titlefont = dict(
        color = 'rgb(107,107,107)'
        ),
        tickfont=dict(
        color = 'rgb(107,107,107)'
        )
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig,image_height=12,image_width=15, filename = 'Year Built Dist')


# In[ ]:


temp = pd.DataFrame(train.floor_count.value_counts()).reset_index()
temp2 = pd.DataFrame(test.floor_count.value_counts()).reset_index()
temp.columns = ['floor_count', 'Counts']
temp2.columns = ['floor_count', 'Counts']
trace1 = go.Bar(x = temp.floor_count, y = temp.Counts, name='Train')
trace2 = go.Bar(x = temp2.floor_count, y = temp2.Counts, name='Test')
data = [trace1, trace2]
layout = go.Layout(
    title = "floor_count Distribution",
    xaxis = dict(
        title = "floor_count",
        tickfont = dict(
        color = 'rgb(107,107,107)'
        )
    ),
    yaxis = dict(
        title = "Counts",
        titlefont = dict(
        color = 'rgb(107,107,107)'
        ),
        tickfont=dict(
        color = 'rgb(107,107,107)'
        )
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig,image_height=12,image_width=15, filename = 'Floor Count Dist')


# In[ ]:


# 'year_built', 'floor_count'
for col in ['square_feet', 'air_temperature',
       'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
       'sea_level_pressure', 'wind_direction', 'wind_speed']:
    plt.figure(figsize=(10,5))
    sns.distplot(train[~train[col].isna()][col],  kde_kws={"lw": 3, "label": 'Train'})
    sns.distplot(test[~test[col].isna()][col],  kde_kws={"lw": 3, "label": 'Test'})
    plt.show()


# The Electricity meter type has the maximum readings. Also, this follows an exponentially decreasing relationship with 0>1>2>3
# 
# Let's try to understand their relationship with the target. Once again I am going to plot a general distribution of target but w.r.t Meter Type

# In[ ]:


fig, ax = plt.subplots(4,2, figsize=(15, 20))
meter_dict = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
meter = [0,1,2,3]
for i, label in enumerate(meter):
    sns.distplot(train[train.meter == label].meter_reading, ax=ax[int(i)][0], kde_kws={"lw": 3, "label": meter_dict[label]})
    sns.distplot(train[(train.meter == label) & (train.meter_reading < 5450)].meter_reading, ax=ax[int(i)][1], kde_kws={"lw": 3, "label": meter_dict[label]})
plt.show()


# This gives us some better idea about target distribution. Most of the zeros are coming in steam meter type while smallest one is for hot water. 

# Identify if there is seasonality according to week or month, 
# 
# Also make a time plot with days distribution along with hours values in it. Perform the same for month and days along with year and month

# In[ ]:


def seasonality(df, col):
    Z = df.groupby(col).agg({'meter_reading': sum}).reset_index()
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (10,5))
    sns.barplot(data=Z, x=col, y='meter_reading', ax = ax, palette="BrBG")
    plt.title('Overall Meter reading over the course of an year')
    plt.show()
    
    # Next plot if according to different meter types
    Z = df.groupby([col, 'meter']).agg({'meter_reading': sum}).reset_index()
    
    fig, ax = plt.subplots(ncols=4, sharey=True, figsize = (20,5))
    sns.barplot(data=Z[Z.meter == 0], x=col, y='meter_reading', ax = ax[0], palette="BrBG")
    sns.barplot(data=Z[Z.meter == 1], x=col, y='meter_reading', ax = ax[1], palette="BrBG")
    sns.barplot(data=Z[Z.meter == 2], x=col, y='meter_reading', ax = ax[2], palette="BrBG")
    sns.barplot(data=Z[Z.meter == 3], x=col, y='meter_reading', ax = ax[3], palette="BrBG")
    plt.show()
    
    # Let's truncate the values to smaller values
    Z = df[df.meter_reading<5450].groupby(col).agg({'meter_reading': sum}).reset_index()
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (10,5))
    sns.barplot(data=Z, x=col, y='meter_reading', ax = ax, palette="BrBG")
    plt.title('Overall Meter reading over the course of an year')
    plt.show()
    
    # Next plot if according to different meter types
    Z = df[df.meter_reading<5450].groupby([col, 'meter']).agg({'meter_reading': sum}).reset_index()
    
    fig, ax = plt.subplots(ncols=4, sharey=True, figsize = (20,5))
    sns.barplot(data=Z[Z.meter == 0], x=col, y='meter_reading', ax = ax[0], palette="BrBG")
    sns.barplot(data=Z[Z.meter == 1], x=col, y='meter_reading', ax = ax[1], palette="BrBG")
    sns.barplot(data=Z[Z.meter == 2], x=col, y='meter_reading', ax = ax[2], palette="BrBG")
    sns.barplot(data=Z[Z.meter == 3], x=col, y='meter_reading', ax = ax[3], palette="BrBG")
    plt.show()


# In[ ]:


seasonality(train,'month')


# In[ ]:


seasonality(train, 'day')


# In[ ]:


seasonality(train, 'hour')


# In[ ]:


seasonality(train, 'dayofweek')


# So far we have seen the relationship of one time variable w.r.t. target. Let's spice things up and see the distribution of 2 time variables w.r.t. target

# In[ ]:


month_day_agg_sum = train[['month', 'day', 'meter_reading']].groupby(['month', 'day']).meter_reading.sum().reset_index()
month_day_agg_sum_limited = train[['month', 'day', 'meter_reading']][train.meter_reading<5450].groupby(['month', 'day']).meter_reading.sum().reset_index()
fig, ax = plt.subplots(1,2, figsize=(16,8))
sns.heatmap(data=month_day_agg_sum.pivot_table(index='day', columns='month', values='meter_reading'), ax=ax[0], cmap="YlGnBu", label='Month vs day')
sns.heatmap(data=month_day_agg_sum_limited.pivot_table(index='day', columns='month', values='meter_reading'), ax=ax[1], cmap="YlGnBu", label='Month vs dat--limited')
ax[0].set_title('Month vs Day')
ax[1].set_title('Month vs Day--limited')
plt.show()


# In[ ]:


month_day_agg_sum = train[['day', 'hour', 'meter_reading']].groupby(['day', 'hour']).meter_reading.sum().reset_index()
month_day_agg_sum_limited = train[['day', 'hour', 'meter_reading']][train.meter_reading<5450].groupby(['day', 'hour']).meter_reading.sum().reset_index()
fig, ax = plt.subplots(2,1, figsize=(16,12))
sns.heatmap(data=month_day_agg_sum.pivot_table(index='hour', columns='day', values='meter_reading'), ax=ax[0], cmap="YlGnBu")
sns.heatmap(data=month_day_agg_sum_limited.pivot_table(index='hour', columns='day', values='meter_reading'), ax=ax[1], cmap="YlGnBu")
ax[0].set_title('Day vs hour')
ax[1].set_title('Day vs hour--limited')
plt.show()


# Few interesting things that we captured:
#     1. The meter reading is in general higher for steam (approximately 10 times). If we restrict the values to smaller extent then steam type changes its distribution. The overall distribution is then dominated by electricity type.
#     2. If we try to locate the highest consumption days in month then it usually starts from 6th and ends in 12th month with approximately even distribution over the days. 
#     3. The day distribution makes sense (for limited values) since most of the readings are in working hours compared to non working hours. 
#     4. If we try to understand the seasonality according to meter types per month then, it makes sense that chilled water is mostly used during the time of summer while steam and hot water is mostly used in the time of winter.
#     

# In[ ]:


# Now delete the test related data in order to understand feature importance otherwise the kernels will be overloaded.
del test, weather_test


# In[ ]:


gc.collect()


# In[ ]:


train.corr()['meter_reading']


# #### The weather condition for the time at which reading was taken won't be helpful. We have to consider other approaches. For an instance, we have to take mean, max, min values of different weather conditions over the course of month [for each site], week of year [for each site]. This will help out model in understanding the weather condition much better which helps in explaining the consumption of energy.
# #### Remove site id 0 for first few month since they are anomalous events.

# In[ ]:


# This function helps in aggregating the variables over different statistical functions
def agg_numeric(df, group_var = [], to_group = [], df_name = 'dummy'):
    
    if len(to_group) == 0 or len(group_var) == 0:
        raise ValueError('Please check grouping and to be grouped variables again!!!')
        
    cols = group_var + to_group
    
    agg = df[cols].groupby(group_var).agg([np.min, np.mean, np.max]).reset_index()
    
    columns = group_var 
    
    for var in agg.columns.levels[0]:
        if var not in group_var:
            for stat in agg.columns.levels[1][:-1]:
                 columns.append('%s_%s_%s' % (df_name, var, stat))
                    
    agg.columns = columns
    return agg


# In[ ]:


# Removing site id 0 for first few months
train = train[~((train.site_id == 0) & (train.month <= 5))]


# In[ ]:


temp = agg_numeric(train, group_var=['site_id', 'month'], 
                   to_group = ['air_temperature', 'cloud_coverage', 'wind_speed'],
                   df_name = 'weather')
train = pd.merge(train, temp, on = ['site_id', 'month'], how = 'inner')
reduce_mem_usage(train)


# In[ ]:


# Another feature engineering -- Age of building
train['age'] = 2016 - train['year_built']


# In[ ]:


train.corr()['meter_reading']


# In[ ]:


def weekend(dayofweek):
    if dayofweek in [5, 6]:
        return 1
    else:
        return 0
temp['dayofweek'] = train.dayofweek.apply(lambda x: weekend(x))


# In[ ]:


train.drop(columns = ['precip_depth_1_hr', 'timestamp', 'year_built', 'wind_direction', 'day', 'hour'], inplace=True)


# In[ ]:


gc.collect()


# In[ ]:


# building features over every month
temp = agg_numeric(train, group_var=['building_id', 'month'], 
                   to_group = ['meter_reading'],
                   df_name = 'building')
train = pd.merge(train, temp, on = ['building_id', 'month'], how = 'inner')
reduce_mem_usage(train)


# In[ ]:


train.corr()['meter_reading']


# #### Analyse the usefulness of variables using feature importance methodology.

# In[ ]:


# Taken from XGBoost documentation

def gradient(predt, dtrain) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

def hessian(predt, dtrain) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))

def squared_log(predt,
                dtrain) -> tuple([np.ndarray, np.ndarray]):
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


# In[ ]:


import math
def rmsle(predt, dtrain):
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y))), False


# In[ ]:


import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
 # lightbgm
params = {
     'max_depth' : 8,
     'boosting_type': 'gbdt',
     'objective': 'regression',
     'metric': {'rmse'},
     'subsample': 0.2,
     'learning_rate': 0.1,
     'feature_fraction': 0.8,
     'bagging_fraction': 0.9,
     'alpha': 0.1,
     'lambda': 0.1
}


# In[ ]:


cols = train.columns.tolist()
cols.remove("meter_reading"); #cols.remove("year_built")
train_label = np.log1p(train.meter_reading)
train = train[cols]
# Create a label encoder -- for object type
label_encoder = LabelEncoder()

for i, col in enumerate(cols):
    if train[col].dtype == 'object':
        # Map the categorical features to integer
        train[col] = label_encoder.fit_transform(np.array(train[col].astype(str)).reshape((-1,)))

# Define categorical cols as well
cat_cols = [0,1,2,3,11]
folds = 3
seed = 666
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
feat_importance=None
count = 0
for train_index, val_index in kf.split(train):
    train_X = train.iloc[train_index]
    val_X = train.iloc[val_index]
    train_y = train_label.iloc[train_index]
    val_y = train_label.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=cat_cols)
    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=cat_cols)
    gbm = lgb.train(params,
                     lgb_train,
                     num_boost_round=50, #300,
                     valid_sets=(lgb_train, lgb_eval),
                     early_stopping_rounds= 5,#100,
                    fobj = squared_log, 
                    feval = rmsle,
                     verbose_eval=2) #100)
    if count == 0:
        feat_importance = pd.DataFrame(zip(gbm.feature_importance(), gbm.feature_name()), columns=['Value','Feature'])
    else:
        feat_importance2 = pd.DataFrame(zip(gbm.feature_importance(), gbm.feature_name()), columns=['Value{0}'.format(count),'Feature'])
        feat_importance=pd.merge(feat_importance, feat_importance2, on='Feature', how='inner')
    count+=1


# In[ ]:


feat_importance['Value'] = (feat_importance.Value+feat_importance.Value1+feat_importance.Value2) / 3


# In[ ]:



fi_df =feat_importance.sort_values('Value', ascending=False)

trace = go.Bar(x = fi_df.Feature, y = fi_df.Value)

data = [trace]
layout = go.Layout(
    title = "Feature importance of LGBM Model",
    xaxis = dict(
        title = "Columns",
        tickfont = dict(
        color = 'rgb(107,107,107)'
        )
    ),
    yaxis = dict(
        title = "Feature Importance",
        titlefont = dict(
        color = 'rgb(107,107,107)'
        ),
        tickfont=dict(
        color = 'rgb(107,107,107)'
        )
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig,image_height=12,image_width=15, filename = 'feature importance status')


# #### Next Stop:
# 1. Lag variables for the same building will also help in identifying the potential energy consumption but that's not possible for target since we are doing projection for next 2 years. We can do lags on weather conditions though [This lag has to be applied by combining both train and test]. 
# 2. Holiday/weekend based features. -- Done
# 2. Building history in terms of energy consumption -- Done
# 3. Remove highly correlated variables or apply dimensionality reduction.
# 5. Implement customized function for RMSLE and use it within the algorithm -- Done.
# 6. Follow the corochann kernel to make a model fit according to different meter type since energy consumption is on different scales for them.

# In[ ]:




