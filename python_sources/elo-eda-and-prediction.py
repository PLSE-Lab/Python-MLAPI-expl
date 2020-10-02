#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Elo EDA Prediction</font></center></h1>
# 
# <img src="https://www.redhat.com/cms/managed-files/elo-225x158.png" width=400></img>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
#  - <a href='#21'>Load packages</a>  
#  - <a href='#22'>Load the data</a>   
# - <a href='#3'>Data exploration</a>   
#  - <a href='#31'>Check for missing data</a>  
#  - <a href='#32'>Train and test data</a>  
#  - <a href='#33'>Historical transaction data</a>  
#   - <a href='#34'>New merchant transaction data</a>  
#   - <a href='#35'>Merchant data</a>  
# - <a href='#4'>Feature engineering</a>
# - <a href='#5'>Model</a>
# - <a href='#6'>Submission</a>
# - <a href='#7'>References</a>    
#     

# # <a id='1'>Introduction</a>  
# 
# This Kernel will take you through the process of analyzing the data to understand the predictive values of various features and the possible correlation between different features, selection of features with predictive value, features engineering to create features with higher predictive value and creation of a baseline model.

# # <a id='2'>Prepare the data analysis</a>   
# 
# 
# Before starting the analysis, we need to make few preparation: load the packages, load and inspect the data.
# 

# ## <a id='21'>Load packages</a>
# 
# We load the packages used for the analysis.

# In[ ]:


import os
import gc
import sys
import random
import logging
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from plotly import tools
from pathlib import Path
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id='22'>Load the data</a>  
# 
# Let's see first what data files do we have in the root directory. 

# In[ ]:


IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/elo/"
else:
    PATH="../input/"
os.listdir(PATH)


# Let's load the data files.

# In[ ]:


train_df=pd.read_csv(PATH+'train.csv')
test_df=pd.read_csv(PATH+'test.csv')
historical_trans_df=pd.read_csv(PATH+'historical_transactions.csv')
new_merchant_trans_df=pd.read_csv(PATH+'new_merchant_transactions.csv')
merchant_df=pd.read_csv(PATH+'merchants.csv')


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# # <a id='3'>Data exploration</a>  
# 
# Let's check the dataframes created.

# In[ ]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))
print("Historical trans: rows:{} cols:{}".format(historical_trans_df.shape[0], historical_trans_df.shape[1]))
print("New merchant trans:  rows:{} cols:{}".format(new_merchant_trans_df.shape[0], new_merchant_trans_df.shape[1]))
print("Merchants: rows:{} cols:{}".format(merchant_df.shape[0], merchant_df.shape[1]))


# In[ ]:


train_df.sample(3).head()


# In[ ]:


test_df.sample(3).head()


# In[ ]:


historical_trans_df.sample(3).head()


# In[ ]:


new_merchant_trans_df.sample(3).head()


# In[ ]:


merchant_df.sample(3).head()


# Let's start by checking if there are missing data, unlabeled data or data that is inconsistently labeled. 
# 
# ## <a id='31'>Check for missing data</a>  
# 
# Let's create a function that check for missing data in the dataframes.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# Let's check missing data for all dataframes.

# In[ ]:


missing_data(train_df)


# In[ ]:


missing_data(test_df)


# In[ ]:


missing_data(historical_trans_df)


# In[ ]:


missing_data(new_merchant_trans_df)


# In[ ]:


missing_data(merchant_df)


# ## <a id='32'>Train and test data</a>  
# 
# Let's check the distribution of train and test features.
# 
# Both have the same features:
# * card_id;  
# * feature1, feature2, feature3;  
# * first_active_month;  
# 
# Train has also the target value, called **target**.   
# 
# Let's define few auxiliary functions.
# 

# In[ ]:


def get_categories(data, val):
    tmp = data[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


def get_target_categories(data, val):
    tmp = data.groupby('target')[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


def draw_trace_bar(data_df,color='Blue'):
    trace = go.Bar(
            x = data_df['index'],
            y = data_df['Number'],
            marker=dict(color=color),
            text=data_df['index']
        )
    return trace

def draw_trace_histogram(data_df,target,color='Blue'):
    trace = go.Histogram(
            y = data_df[target],
            marker=dict(color=color)
        )
    return trace


# In[ ]:


def plot_bar(data_df, title, xlab, ylab,color='Blue'):
    trace = draw_trace_bar(data_df, color)
    data = [trace]
    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=0,
                          tickfont=dict(
                            size=10,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')


# In[ ]:


def plot_two_bar(data_df1, data_df2, title1, title2, xlab, ylab):
    trace1 = draw_trace_bar(data_df1, color='Blue')
    trace2 = draw_trace_bar(data_df2, color='Lightblue')
    
    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=(title1,title2))
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    
    fig['layout']['xaxis'].update(title = xlab)
    fig['layout']['xaxis2'].update(title = xlab)
    fig['layout']['yaxis'].update(title = ylab)
    fig['layout']['yaxis2'].update(title = ylab)
    fig['layout'].update(showlegend=False)
    
    iplot(fig, filename='draw_trace')


# In[ ]:


def plot_target_distribution(var):
    hist_data = []
    varall = list(train_df.groupby([var])[var].nunique().index)
    for i, varcrt in enumerate(varall):
        classcrt = train_df[train_df[var] == varcrt]['target']
        hist_data.append(classcrt)
    fig = ff.create_distplot(hist_data, varall, show_hist=False, show_rug=False)
    fig['layout'].update(title='Target variable density plot group by {}'.format(var), xaxis=dict(title='Target'))
    iplot(fig, filename='dist_only')


# Let's show the distribution of **feature_1** for **train** and **test** set.

# In[ ]:


plot_two_bar(get_categories(train_df,'feature_1'), get_categories(test_df,'feature_1'), 
             'Train data', 'Test data',
             'Feature 1', 'Number of records')


# Let's see the distribution of **target** value groped on  **feature_1** values.

# In[ ]:


plot_target_distribution('feature_1')


# Let's see the distribution of **feature_2** for **train** and **test** set.

# In[ ]:


plot_two_bar(get_categories(train_df,'feature_2'), get_categories(test_df,'feature_2'), 
             'Train data', 'Test data',
             'Feature 2', 'Number of records')


# Let's see the distribution of **target** grouped on **feature_2** values.

# In[ ]:


plot_target_distribution('feature_2')


# Let's show now the distribution of **feature_3** for  **train** and **test**.

# In[ ]:


plot_two_bar(get_categories(train_df,'feature_3'), get_categories(test_df,'feature_3'), 
             'Train data', 'Test data',
             'Feature 3', 'Number of records')


# And let's see also the distribuiton of **target** grouped by values of **feature_3**.

# In[ ]:


plot_target_distribution('feature_3')


# Let's plot now the distribution of **first_active_month** from **train** and **test** datasets.

# In[ ]:


plot_two_bar(get_categories(train_df,'first_active_month'), get_categories(test_df,'first_active_month'), 
             'Train data', 'Test data',
             'First active month', 'Number of records')


# ## <a id='33'>Historical transaction data</a>  
# 
# Let's check the distribution of historical transaction data features.  
# 
# **historical_trans_df** is linked with **train_df** and **test_df** by the **card_id** key.
# 
# Let's plot **category_1**, **category_2**, **category_3** features distribution.
# 

# In[ ]:


plot_bar(get_categories(historical_trans_df,'category_1'), 
             'Category 1 distribution', 'Category 1', 'Number of records')


# In[ ]:


plot_bar(get_categories(historical_trans_df,'category_2'), 
             'Category 2 distribution', 'Category 2', 'Number of records','red')


# In[ ]:


plot_bar(get_categories(historical_trans_df,'category_3'), 
             'Category 3 distribution', 'Category 3', 'Number of records','magenta')


# Let's see **city_id**, **merchant_category_id**,  **state_id**, **subsector_id**.

# In[ ]:


plot_bar(get_categories(historical_trans_df,'city_id'), 
             'City ID distribution', 'City ID', 'Number of records','lightblue')


# In[ ]:


plot_bar(get_categories(historical_trans_df,'merchant_category_id'), 
             'Merchant Cateogory ID distribution', 'Merchant Category ID', 'Number of records','lightgreen')


# In[ ]:


plot_bar(get_categories(historical_trans_df,'state_id'), 
             'State ID distribution', 'State ID', 'Number of records','brown')


# In[ ]:


plot_bar(get_categories(historical_trans_df,'subsector_id'), 
             'Subsector ID distribution', 'Subsector ID', 'Number of records','orange')


# Let's show the purchase amount grouped by purchase time types.
# 
# Before this, let's extract the date.

# In[ ]:


historical_trans_df['purchase_date'] = pd.to_datetime(historical_trans_df['purchase_date'])
historical_trans_df['month'] = historical_trans_df['purchase_date'].dt.month
historical_trans_df['dayofweek'] = historical_trans_df['purchase_date'].dt.dayofweek
historical_trans_df['weekofyear'] = historical_trans_df['purchase_date'].dt.weekofyear


# In[ ]:


def plot_scatter_data(data, xtitle, ytitle, title, color='blue'):
    trace = go.Scatter(
        x = data.index,
        y = data.values,
        name=ytitle,
        marker=dict(
            color=color,
        ),
        mode='lines+markers'
    )
    data = [trace]
    layout = dict(title = title,
              xaxis = dict(title = xtitle), yaxis = dict(title = ytitle),
             )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='lines')


# Let's plot the amount of purchase per day of week, week of year and month.

# In[ ]:


count_all = historical_trans_df.groupby('dayofweek')['purchase_amount'].agg(['sum'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Day of week', 'Total','Total sum of purchase per day of week','green')


# In[ ]:


count_all = historical_trans_df.groupby('weekofyear')['purchase_amount'].agg(['sum'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Week of year', 'Total','Total sum of purchase per Week of Year','red')


# In[ ]:


count_all = historical_trans_df.groupby('month')['purchase_amount'].agg(['sum'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Month', 'Total','Total sum of purchase per month','blue')


# ## <a id='34'>New merchant transaction data</a>  
# 
# Let's check the distribution of new merchant transaction data features.   
# 
# **new_merchant_trans_df** is linked with **train_df** and **test_df** by the **card_id** key.
# 
# Let's plot **category_1**, **category_2**, **category_3** features distribution.

# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'category_1'), 
             'Category 1 distribution', 'Category 1', 'Number of records','gold')


# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'category_2'), 
             'Category 2 distribution', 'Category 2', 'Number of records','tomato')


# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'category_3'), 
             'Category 3 distribution', 'Category 3', 'Number of records','magenta')


# Let's see city_id, merchant_category_id,  state_id, subsector_id.

# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'city_id'), 
             'City ID distribution', 'City ID', 'Number of records','brown')


# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'merchant_category_id'), 
             'Merchant category ID distribution', 'Merchant category ID', 'Number of records','green')


# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'state_id'), 
             'State ID distribution', 'State ID', 'Number of records','darkblue')


# In[ ]:


plot_bar(get_categories(new_merchant_trans_df,'subsector_id'), 
             'Subsector ID distribution', 'Subsector ID', 'Number of records','darkgreen')


# Let's show the purchase amount grouped by purchase date types.

# In[ ]:


new_merchant_trans_df['purchase_date'] = pd.to_datetime(new_merchant_trans_df['purchase_date'])
new_merchant_trans_df['month'] = new_merchant_trans_df['purchase_date'].dt.month
new_merchant_trans_df['dayofweek'] = new_merchant_trans_df['purchase_date'].dt.dayofweek
new_merchant_trans_df['weekofyear'] = new_merchant_trans_df['purchase_date'].dt.weekofyear


# In[ ]:


count_all = new_merchant_trans_df.groupby('month')['purchase_amount'].agg(['sum'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Month', 'Total','Total sum of purchase per month','red')


# In[ ]:


count_all = new_merchant_trans_df.groupby('dayofweek')['purchase_amount'].agg(['sum'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Day of week', 'Total','Total sum of purchase per day of week','magenta')


# In[ ]:


count_all = new_merchant_trans_df.groupby('weekofyear')['purchase_amount'].agg(['sum'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Week of year', 'Total','Total sum of purchase per week of year','darkblue')


# Let's check the distribution of the purchase amount grouped by various features. We will represent  log(purchase_amount + 1).

# In[ ]:


def plot_purchase_amount_distribution(data_df, var):
    hist_data = []
    varall = list(data_df.groupby([var])[var].nunique().index)
    for i, varcrt in enumerate(varall):
        classcrt = np.log(data_df[data_df[var] == varcrt]['purchase_amount'] + 1)
        hist_data.append(classcrt)
    fig = ff.create_distplot(hist_data, varall, show_hist=False, show_rug=False)
    fig['layout'].update(title='Purchase amount (log) variable density plot group by {}'.format(var), xaxis=dict(title='log(purchase_amount + 1)'))
    iplot(fig, filename='dist_only')


# In[ ]:


plot_purchase_amount_distribution(new_merchant_trans_df,'category_1')


# In[ ]:


plot_purchase_amount_distribution(new_merchant_trans_df,'category_2')


# In[ ]:


plot_purchase_amount_distribution(new_merchant_trans_df,'category_3')


# In[ ]:


plot_purchase_amount_distribution(new_merchant_trans_df,'state_id')


# ## <a id='35'>Merchant data</a>  
# 
# Let's check the distribution of merchant data features.   
# 
# Let's start with  **merchant_category_id**, **subsector_id**.
# 

# In[ ]:


merchant_df.head(3)


# In[ ]:


plot_bar(get_categories(merchant_df,'merchant_category_id'), 
             'Merchant category ID distribution', 'Merchant category ID', 'Number of records','darkblue')


# In[ ]:


plot_bar(get_categories(merchant_df,'subsector_id'), 
             'Subsector ID distribution', 'Subsector ID', 'Number of records','blue')


# Let's follow with **category_1**, **category_2**, **category_4**.

# In[ ]:


plot_bar(get_categories(merchant_df,'category_1'), 
             'Category 1 distribution', 'Category 1', 'Number of records','lightblue')


# In[ ]:


plot_bar(get_categories(merchant_df,'category_2'), 
             'Category 2 distribution', 'Category 2', 'Number of records','lightgreen')


# In[ ]:


plot_bar(get_categories(merchant_df,'category_4'), 
             'Category 4 distribution', 'Category 4', 'Number of records','tomato')


# Let's check **most_recent_sales_range** and **most_recent_purchase_range**[](http://).

# In[ ]:


plot_bar(get_categories(merchant_df,'most_recent_sales_range'), 
             'Most recent sales range distribution', 'Most recent sales range', 'Number of records','red')


# In[ ]:


plot_bar(get_categories(merchant_df,'most_recent_purchases_range'), 
             'Most recent sales purchases distribution', 'Most recent purchases range', 'Number of records','magenta')


# Let's look to the **city_id**, **state_id**.

# In[ ]:


plot_bar(get_categories(merchant_df,'city_id'), 
             'City ID distribution', 'City ID', 'Number of records','brown')


# In[ ]:


plot_bar(get_categories(merchant_df,'state_id'), 
             'State ID distribution', 'State ID', 'Number of records','orange')


# Let's plot distribution of **numerical_1**, **numerical_2**, **avg_sales_lag3**, **avg_sales_lag6**, **avg_sales_lag12**.

# In[ ]:


def plot_distribution(df,feature,color):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    s = sns.boxplot(ax = ax1, data = df[feature].dropna(),color=color,showfliers=True)
    s.set_title("Distribution of %s (with outliers)" % feature)
    s = sns.boxplot(ax = ax2, data = df[feature].dropna(),color=color,showfliers=False)
    s.set_title("Distribution of %s (no outliers)" % feature)
    plt.show()   


# In[ ]:


plot_distribution(merchant_df, "numerical_1", "blue")


# In[ ]:


plot_distribution(merchant_df, "numerical_2", "green")


# In[ ]:


plot_distribution(merchant_df, "avg_sales_lag3", "blue")


# In[ ]:


plot_distribution(merchant_df, "avg_sales_lag6", "green")


# In[ ]:


plot_distribution(merchant_df, "avg_sales_lag12", "green")


# # <a id='4'>Feature engineering</a>  
# 
# 
# Before creating the model we will prepare the aggregated features.
# 

# ## Utility functions and data cleaning

# In[ ]:


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


# In[ ]:


# reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
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


logger = get_logger()
#process NAs
logger.info('Start processing NAs')
#process NA2 for transactions
for df in [historical_trans_df, new_merchant_trans_df]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    df['installments'].replace(-1, np.nan,inplace=True)
    df['installments'].replace(999, np.nan,inplace=True)
#define function for aggregation
def create_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


# In[ ]:


logger.info('process historical and new merchant datasets')
for df in [historical_trans_df, new_merchant_trans_df]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2}) 
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
logger.info('new features historical and new merchant datasets')
for df in [historical_trans_df, new_merchant_trans_df]:
    df['price'] = df['purchase_amount'] / df['installments']
    df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    df['Children_day_2017']=(pd.to_datetime('2017-10-12')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
    df['duration'] = df['purchase_amount']*df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']


# In[ ]:


logger.info('reduce memory usage for historical trans')
historical_trans_df = reduce_mem_usage(historical_trans_df)
logger.info('reduce memory usage for new merchant trans')
new_merchant_trans_df = reduce_mem_usage(new_merchant_trans_df)


# ## Process historical transaction data

# In[ ]:


#define aggregations with historical_trans_df
logger.info('Aggregate historical trans')
aggs = {}

for col in ['subsector_id','merchant_id','merchant_category_id', 'state_id', 'city_id']:
    aggs[col] = ['nunique']
for col in ['month', 'hour', 'weekofyear', 'dayofweek']:
    aggs[col] = ['nunique', 'mean', 'min', 'max']
    
aggs['purchase_amount'] = ['sum','max','min','mean','var', 'std']
aggs['installments'] = ['sum','max','min','mean','var', 'std']
aggs['purchase_date'] = ['max','min', 'nunique']
aggs['month_lag'] = ['max','min','mean','var','nunique']
aggs['month_diff'] = ['mean', 'min', 'max', 'var','nunique']
aggs['authorized_flag'] = ['sum', 'mean', 'nunique']
aggs['weekend'] = ['sum', 'mean', 'nunique']
aggs['year'] = ['nunique', 'mean']
aggs['category_1'] = ['sum', 'mean', 'min', 'max', 'nunique', 'std']
aggs['category_2'] = ['sum', 'mean', 'min', 'nunique', 'std']
aggs['category_3'] = ['sum', 'mean', 'min', 'nunique', 'std']
aggs['card_id'] = ['size', 'count']
aggs['Christmas_Day_2017'] = ['mean']
aggs['Children_day_2017'] = ['mean']
aggs['Black_Friday_2017'] = ['mean']
aggs['Mothers_Day_2018'] = ['mean']

for col in ['category_2','category_3']:
    historical_trans_df[col+'_mean'] = historical_trans_df.groupby([col])['purchase_amount'].transform('mean')
    historical_trans_df[col+'_min'] = historical_trans_df.groupby([col])['purchase_amount'].transform('min')
    historical_trans_df[col+'_max'] = historical_trans_df.groupby([col])['purchase_amount'].transform('max')
    historical_trans_df[col+'_sum'] = historical_trans_df.groupby([col])['purchase_amount'].transform('sum')
    historical_trans_df[col+'_std'] = historical_trans_df.groupby([col])['purchase_amount'].transform('std')
    aggs[col+'_mean'] = ['mean']    

new_columns = create_new_columns('hist',aggs)
historical_trans_group_df = historical_trans_df.groupby('card_id').agg(aggs)
historical_trans_group_df.columns = new_columns
historical_trans_group_df.reset_index(drop=False,inplace=True)
historical_trans_group_df['hist_purchase_date_diff'] = (historical_trans_group_df['hist_purchase_date_max'] - historical_trans_group_df['hist_purchase_date_min']).dt.days
historical_trans_group_df['hist_purchase_date_average'] = historical_trans_group_df['hist_purchase_date_diff']/historical_trans_group_df['hist_card_id_size']
historical_trans_group_df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_max']).dt.days
historical_trans_group_df['hist_purchase_date_uptomin'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_min']).dt.days

logger.info('reduce memory usage for historical trans')
historical_trans_df = reduce_mem_usage(historical_trans_df)

logger.info('Completed aggregate historical trans')


# In[ ]:


#merge with train, test
train_df = train_df.merge(historical_trans_group_df,on='card_id',how='left')
test_df = test_df.merge(historical_trans_group_df,on='card_id',how='left')
#cleanup memory
del historical_trans_group_df; gc.collect()


# ## Process new merchant transaction data

# In[ ]:


#define aggregations with new_merchant_trans_df 
logger.info('Aggregate new merchant trans')
aggs = {}
for col in ['subsector_id','merchant_id','merchant_category_id','state_id', 'city_id']:
    aggs[col] = ['nunique']
for col in ['month', 'hour', 'weekofyear', 'dayofweek']:
    aggs[col] = ['nunique', 'mean', 'min', 'max']

    
aggs['purchase_amount'] = ['sum','max','min','mean','var','std']
aggs['installments'] = ['sum','max','min','mean','var','std']
aggs['purchase_date'] = ['max','min', 'nunique']
aggs['month_lag'] = ['max','min','mean','var', 'nunique']
aggs['month_diff'] = ['mean', 'max', 'min', 'var','nunique']
aggs['weekend'] = ['sum', 'mean', 'nunique']
aggs['year'] = ['nunique', 'mean']
aggs['category_1'] = ['sum', 'mean', 'min', 'nunique']
aggs['category_2'] = ['sum', 'mean', 'min', 'nunique']
aggs['category_3'] = ['sum', 'mean', 'min', 'nunique']
aggs['card_id'] = ['size', 'count']
aggs['Christmas_Day_2017'] = ['mean']
aggs['Children_day_2017'] = ['mean']
aggs['Black_Friday_2017'] = ['mean']
aggs['Mothers_Day_2018'] = ['mean']

for col in ['category_2','category_3']:
    new_merchant_trans_df[col+'_mean'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('mean')
    new_merchant_trans_df[col+'_min'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('min')
    new_merchant_trans_df[col+'_max'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('max')
    new_merchant_trans_df[col+'_sum'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('sum')
    new_merchant_trans_df[col+'_std'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('std')
    aggs[col+'_mean'] = ['mean']

new_columns = create_new_columns('new_hist',aggs)
new_merchant_trans_group_df = new_merchant_trans_df.groupby('card_id').agg(aggs)
new_merchant_trans_group_df.columns = new_columns
new_merchant_trans_group_df.reset_index(drop=False,inplace=True)
new_merchant_trans_group_df['new_hist_purchase_date_diff'] = (new_merchant_trans_group_df['new_hist_purchase_date_max'] - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days
new_merchant_trans_group_df['new_hist_purchase_date_average'] = new_merchant_trans_group_df['new_hist_purchase_date_diff']/new_merchant_trans_group_df['new_hist_card_id_size']
new_merchant_trans_group_df['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_max']).dt.days
new_merchant_trans_group_df['new_hist_purchase_date_uptomin'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days

logger.info('reduce memory usage for new merchant trans')
new_merchant_trans_df = reduce_mem_usage(new_merchant_trans_df)

logger.info('Completed aggregate new merchant trans')


# In[ ]:


#merge with train, test
train_df = train_df.merge(new_merchant_trans_group_df,on='card_id',how='left')
test_df = test_df.merge(new_merchant_trans_group_df,on='card_id',how='left')
#clean-up memory
del new_merchant_trans_group_df; gc.collect()
del historical_trans_df; gc.collect()
del new_merchant_trans_df; gc.collect()


# ## Process train and test data

# In[ ]:


#process train
logger.info('Process train - outliers')
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1
outls = train_df['outliers'].value_counts()
print("Outliers: {}".format(outls))
logger.info('Process train and test')
## process both train and test
for df in [train_df, test_df]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['dayofyear'] = df['first_active_month'].dt.dayofyear
    df['quarter'] = df['first_active_month'].dt.quarter
    df['is_month_start'] = df['first_active_month'].dt.is_month_start
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_last_buy'] = (df['new_hist_purchase_date_max'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',                     'new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_hist_card_id_count']+df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_hist_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_hist_purchase_amount_max']+df['hist_purchase_amount_max']

    for f in ['feature_1','feature_2','feature_3']:
        order_label = train_df.groupby([f])['outliers'].mean()
        df[f] = df[f].map(order_label)

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    
##
train_columns = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = train_df['target']
del train_df['target']
logger.info('Completed process train')


# # <a id='5'>Model</a>  

# In[ ]:


#model
##model params
logger.info('Prepare model')
param = {'num_leaves': 51,
         'min_data_in_leaf': 35, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.008,
         "boosting": "gbdt",
         "feature_fraction": 0.85,
         "bagging_freq": 1,
         "bagging_fraction": 0.82,
         "bagging_seed": 42,
         "metric": 'rmse',
         "lambda_l1": 0.11,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 2019}
#prepare fit model with cross-validation
folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()
#run model
logger.info('Start running model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):
    strLog = "Fold {}".format(fold_)
    print(strLog)
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx][train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) / folds.n_splits
    logger.info(strLog)
    
strRMSE = "".format(np.sqrt(mean_squared_error(oof, target)))
print(strRMSE)


# ## Feature importance

# In[ ]:


##plot the feature importance
logger.info("Feature importance plot")
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# # <a id='6'>Submission</a>  

# In[ ]:


##submission
logger.info("Prepare submission")
sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)


# # <a id='7'>References</a>  
# 
# [1]  https://www.kaggle.com/gpreda/elo-world-high-score-without-blending    
# [2] https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending   
# 
