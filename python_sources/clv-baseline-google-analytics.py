#!/usr/bin/env python
# coding: utf-8

# [](http://)# Notes
# 
# # Version 17 is the best so far, you were able to get to 0.3687...
# 
# Link to Project: https://www.kaggle.com/c/ga-customer-revenue-prediction/data
# 
# References:
# 
# Top 40 Solution: python - https://www.kaggle.com/augustmarvel/base-model-v2-user-level-solution
# - Note: That the baseline model (all zeros) had an RMSE of 0.331 on his validation dataset, while the best model had 
# 
# Visualization of Original Features: https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
# 
# Source dataset for v2 datasets: https://www.kaggle.com/jsaguiar/parse-json-v2-without-hits-column
# 
# Simple Example using, predictive period: https://www.kaggle.com/super13579/ga-v2-future-purchase-prediction/comments
# 
# 36th place solution(R Code): https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82746#latest-483017

# In[ ]:


import os
import json
import datetime
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.io.json import json_normalize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import gc
gc.enable()
color = sns.color_palette()


from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        


# In[ ]:


# Not Used just here for reference:
# https://www.kaggle.com/super13579/ga-v2-future-purchase-prediction/comments

def load_df(csv_path, chunksize=100000):
    features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',
                'visitNumber', 'visitStartTime', 'device_browser',
                'device_deviceCategory', 'device_isMobile', 'device_operatingSystem',
                'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',
                'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',
                'geoNetwork_subContinent', 'totals_bounces', 'totals_hits',
                'totals_newVisits', 'totals_pageviews', 'totals_transactionRevenue',
                'trafficSource_adContent', 'trafficSource_campaign',
                'trafficSource_isTrueDirect', 'trafficSource_keyword',
                'trafficSource_medium', 'trafficSource_referralPath',
                'trafficSource_source']
    JSON_COLS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    print('Load {}'.format(csv_path))
    df_reader = pd.read_csv(csv_path,
                            converters={ column: json.loads for column in JSON_COLS },
                            dtype={ 'date': str, 'fullVisitorId': str, 'sessionId': str },
                            chunksize=chunksize)
    res = pd.DataFrame()
    for cidx, df in enumerate(df_reader):
        df.reset_index(drop=True, inplace=True)
        for col in JSON_COLS:
            col_as_df = json_normalize(df[col])
            col_as_df.columns = ['{}_{}'.format(col, subcol) for subcol in col_as_df.columns]
            df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)
        res = pd.concat([res, df[features]], axis=0).reset_index(drop=True)
        del df
        gc.collect()
        print('{}: {}'.format(cidx + 1, res.shape))
    return res



def process_format(df):
    print('process format')
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        df[col] = df[col].astype(float)
    df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return df

def process_device(df):
    print('process device')
    df['browser_category'] = df['device_browser'] + '_' + df['device_deviceCategory']
    df['browser_operatingSystem'] = df['device_browser'] + '_' + df['device_operatingSystem']
    df['source_country'] = df['trafficSource_source'] + '_' + df['geoNetwork_country']
    return df

def process_geo_network(df):
    print('process geo network')
    df['count_hits_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    df['sum_hits_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    df['count_pvs_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    df['sum_pvs_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    return df

def process_categorical_cols(train_df, test_df, excluded_cols):
    # Label encoding
    objt_cols = [col for col in train_df.columns if col not in excluded_cols and train_df[col].dtypes == object]
    for col in objt_cols:
        train_df[col], indexer = pd.factorize(train_df[col])
        test_df[col] = indexer.get_indexer(test_df[col])
    bool_cols = [col for col in train_df.columns if col not in excluded_cols and train_df[col].dtypes == bool]
    for col in bool_cols:
        train_df[col] = train_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)
    # Fill NaN
    numb_cols = [col for col in train_df.columns if col not in excluded_cols and col not in objt_cols]
    for col in numb_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    return train_df, test_df

def process_dfs(train_df, test_df, target_values, excluded_cols):
    print('Dropping repeated columns')
    cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
    train_df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
    print('Extracting features')
    print('Training set:')
    train_df = process_date_time(train_df)
    train_df = process_format(train_df)
    train_df = process_device(train_df)
    train_df = process_geo_network(train_df)
    print('Testing set:')
    test_df = process_date_time(test_df)
    test_df = process_format(test_df)
    test_df = process_device(test_df)
    test_df = process_geo_network(test_df)
    print('Postprocess')
    train_df, test_df = process_categorical_cols(train_df, test_df, excluded_cols)
    return train_df, test_df
  
def preprocess():
    # Load data set.
    train_df = load_df('../input/train_v2.csv')
    test_df = load_df('../input/test_v2.csv')
    # Obtain target values.
    target_values = np.log1p(train_df['totals_transactionRevenue'].fillna(0).astype(float))
    # Extract features.
    EXCLUDED_COLS = ['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals_transactionRevenue']
    train_df, test_df = process_dfs(train_df, test_df, target_values, EXCLUDED_COLS)
    test_fvid = test_df[['fullVisitorId']].copy()
    train_df.drop(EXCLUDED_COLS, axis=1, inplace=True)
    test_df.drop(EXCLUDED_COLS, axis=1, inplace=True)
    return train_df, target_values, test_df, test_fvid


# In[ ]:


NUMERIC_FEAT_COLUMNS = [
    'totals_hits',
    'totals_pageviews',
    'totals_timeOnSite',
    'totals_totalTransactionRevenue', 
    'totals_transactions'
]

def type_correct_numeric(df):
    for col in NUMERIC_FEAT_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)
    
    return df

def process_date_time(df):
    print('process date')
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.weekofyear
#    df['weekday'] = df['date'].dt.weekday
    return df

def add_index_and_deduplicate(df):
    n_rows, n_cols = df.shape

    df['unique_row_id'] = df.fullVisitorId.map(str) + '.' + df.visitId.map(str)
    df.index = df.unique_row_id
    deduped_df = df.loc[~df.index.duplicated(keep='first')]
    print('De dupliceated {} rows'.format(n_rows - deduped_df.shape[0]))
    return deduped_df

def fillnas(df):
    df = df['trafficSource_isTrueDirect'].fillna(False)
    return 


# # Load / Process Dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', 'path = "../input/google-analytics-preprocessed-dataset/"\ntrain_df = pd.read_pickle(path + \'train_v2_clean.pkl\')\ntest_df = pd.read_pickle(path + \'test_v2_clean.pkl\')')


# In[ ]:


print('Processing Training Data...')
train_df =  process_date_time(train_df)
train_df = type_correct_numeric(train_df)
train_df = add_index_and_deduplicate(train_df)
print()
print('Processing Test Data...')
test_df =  process_date_time(test_df)
test_df = type_correct_numeric(test_df)
test_df = add_index_and_deduplicate(test_df)

gc.collect()
# full_df = process_date_time(full_df)
# full_df = correct_dtypes(full_df)


# In[ ]:


print('Train Date Range', train_df.date.min(), ' - ', train_df.date.max())
print('Test Date Range', test_df.date.min(), ' - ',  test_df.date.max())


# In[ ]:


print('Train Data Shape', train_df.shape, 'From:', train_df.date.min(), 'To:', train_df.date.max(), 'Duration:', train_df.date.max() - train_df.date.min())
print('Trest Data Shape', test_df.shape, 'From:', test_df.date.min(), 'To:', test_df.date.max(), 'Duration:', test_df.date.max() - test_df.date.min())


# # Basic Visualizations to look for Trends / Seasonality 
#  * there doesnt appear to be any trends over time (increase in spend over time or significant seasonal effects (see rolling 30 day charts)

# ## Frequency by Date

# In[ ]:


train_df.groupby(train_df['date'].dt.date)['date'].count().plot(rot=90)
test_df.groupby(test_df['date'].dt.date)['date'].count().plot(rot=90)
plt.title('Sessions By Day')
plt.show()

train_df.groupby(train_df['date'].dt.date)['date'].count().rolling(7).mean().plot(rot=90)
test_df.groupby(test_df['date'].dt.date)['date'].count().rolling(7).mean().plot(rot=90)
plt.title('Sessions By Day Rolling 7 Day Mean')
plt.show()

train_df.groupby(train_df['date'].dt.date)['date'].count().rolling(30).mean().plot(rot=90)
test_df.groupby(test_df['date'].dt.date)['date'].count().rolling(30).mean().plot(rot=90)
plt.title('Sessions By Day Rolling 30 Day Mean')
plt.show()


# ## Revenue by Date

# In[ ]:


CONSTANT_MULTIPLE_FOR_REVENUE = 1000000

train_df.groupby(train_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().div(CONSTANT_MULTIPLE_FOR_REVENUE).plot(rot=90)
test_df.groupby(test_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().div(CONSTANT_MULTIPLE_FOR_REVENUE).plot(rot=90)
plt.title('Revenue By Day')
plt.show()

train_df.groupby(train_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().div(CONSTANT_MULTIPLE_FOR_REVENUE).rolling(7).mean().plot(rot=90)
test_df.groupby(test_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().div(CONSTANT_MULTIPLE_FOR_REVENUE).rolling(7).mean().plot(rot=90)
plt.title('Revenue By Day Rolling 7 Day Mean')
plt.show()

train_df.groupby(train_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().div(CONSTANT_MULTIPLE_FOR_REVENUE).rolling(30).mean().plot(rot=90)
test_df.groupby(test_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().div(CONSTANT_MULTIPLE_FOR_REVENUE).rolling(30).mean().plot(rot=90)
plt.title('Revenue By Day Rolling 30 Day Mean')
plt.show()

# train_df.groupby(train_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().plot(rot=90) #.rolling(30).mean().plot(rot=90)
# test_df.groupby(test_df['date'].dt.date)['totals_totalTransactionRevenue'].sum().plot(rot=90) #.rolling(30).mean().plot(rot=90)
# plt.title('Daily Revenue')


# In[ ]:


train_df.head()


# # Create Train/Validation Splits
# Per documentation going to use the rmse on nlog1p of predicted revenue 
# 
# 
# 
# ~~In order to simulate the example going to try and predict the revenue in the test period for users who were active in the 167 days prior to the end of train period~~
# 
# Update Tried the Above, but of the 378793 in the target period: 2017-11-15 to 2018-05-01, only 76 customers actually returned and transacted, also this doesnt accurately reflect the test client problem and let me leverage all historical data. Going to swith to predicting based on the full train set for customer that interacted in the last 365 days and predicting the next 150 Days total revenue
# 
# https://www.kaggle.com/c/ga-customer-revenue-prediction/data

# In[ ]:


DAYS_LOOK_BACK = 365
DAYS_PREDICT_FORWARD = 90

end_of_train_period = train_df.date.max() 
start_of_lookback_window = end_of_train_period + pd.Timedelta(days = -DAYS_LOOK_BACK)

print('End of Train Period', end_of_train_period)
print('Start of Target Window', start_of_lookback_window)
print('GOAL: is to predict transaction revenue in the test period for visitors active from this time until end of train window')
print()
print()
start_of_test_period = test_df.date.min() 
end_of_window_for_traget_revenue = start_of_test_period + pd.Timedelta(days = DAYS_PREDICT_FORWARD)
print('start_of_test_period', start_of_test_period)
print('end_of_window_for_traget_revenue', end_of_window_for_traget_revenue)


# In[ ]:


cust_in_train = set(train_df.fullVisitorId.unique())
cust_in_test = set(test_df.fullVisitorId.unique())
cust_in_lookback_window = set(train_df[(train_df['date']>=start_of_lookback_window)].fullVisitorId.unique())

cust_in_target_window = set(test_df[test_df.date <= end_of_window_for_traget_revenue].fullVisitorId.unique())

num_cust_in_target_window_and_test = len(cust_in_lookback_window.intersection(cust_in_target_window))

print('Num Customers in Train', len(cust_in_train))
print('Num Customers in Lookback Window (last {} days of Train)'.format(DAYS_LOOK_BACK), len(cust_in_lookback_window))
print('Num Customers in Target Window', len(cust_in_target_window))
print('Num Customers in Target Window and LookbackWindow', num_cust_in_target_window_and_test)
print('Num Customers in Target Window and All of Train', len(cust_in_train.intersection(cust_in_target_window)))


# # Build Target Features for Test Set
# 
# ## Goal: for Customers in Lookback Window(365 Days) Predict the sales for the first 150 days in the test period

# In[ ]:


target_revenue = test_df[test_df.date <= end_of_window_for_traget_revenue].groupby(['fullVisitorId'])['totals_totalTransactionRevenue'].sum().to_frame('target_revenue')
print(target_revenue.shape)
#target_revenue.head()


# In[ ]:


customers_in_lookback_period = train_df[(train_df['date']>=start_of_lookback_window)].groupby(['fullVisitorId'])['totals_totalTransactionRevenue'].sum().to_frame('lookback_revenue')
print(customers_in_lookback_period.shape)
#customers_in_lookback_period.head()


# In[ ]:


targets = customers_in_lookback_period.join(target_revenue, how='left')
print('Num overlapping Customers', targets.dropna().shape, 'Note: Should match results above')
targets['cust_in_lookback_and_target_windows'] = targets.target_revenue.notna()
targets = targets.fillna(0) #fill na with zero
targets['target_revenue_log1p'] = np.log1p(targets['target_revenue'].values)
print(targets.shape)
# targets.rename(columns={'sum': 'revenue_05012018_to_07302018'}, inplace=True) 
# targets.drop(columns=['fullVisitorId'], inplace=True)
targets.sort_values(['target_revenue'], ascending=False).head()
#targets.sort_values(['target_revenue'], ascending=False).reset_index().reset_index().plot.scatter(x='index', y='target_revenue_log1p')


# In[ ]:


num_ret_custs = (targets.target_revenue > 0.0).sum()
print('returning Customer that transact??', num_ret_custs, num_ret_custs/ len(cust_in_target_window)) #


# In[ ]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm


# # Analyze Repurchasing Patterns
# ### Important things to note:
# * If you transacted in the lookback period your likelihood of transacting is much higher in the target period
#  * +50x, likelihood for All Customers in Lookback Period
#  * +10x, increase in likelihood for returning customers
#  
# * That said amongst customers who end up transacting in the Target Period (who were also in the lookback period) ~2/3 are from non-transacting accts. Meaning they looked at the site in the Lookback Period, but purchased in the target window, in a purely transaction CLV, these customers would be excluded, but they make up the **

# In[ ]:


#create a did_transact_confusion_matrix:
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#cm = confusion_matrix(targets.lookback_revenue > 0, targets.target_revenue > 0)

print('True Label = Transacted In Lookback')
print('Predicted Label = Transacted In Target')
print()
_, cm = plot_confusion_matrix(
    y_true=(targets.lookback_revenue.values > 0).astype(int), 
    y_pred = (targets.target_revenue.values > 0).astype(int), classes=np.array([False, True]), 
    title='Confusion matrix')
print('% of Customers Who transacted in Lookback Window and Target Window', cm[1, 1]/cm[1, :].sum())
print('% of Customers Who Appeared in Lookback Window and Transacted in Target Window', cm[0, 1]/cm[0, :].sum())


# #### Let's look at the confusion matrix for returning customers only

# In[ ]:


print('Only for Returning Customers')
print('True Label = Transacted In Lookback')
print('Predicted Label = Transacted In Target')
print()
_, cm = plot_confusion_matrix(
    y_true=(targets[targets.cust_in_lookback_and_target_windows].lookback_revenue.values > 0).astype(int), 
    y_pred = (targets[targets.cust_in_lookback_and_target_windows].target_revenue.values > 0).astype(int), classes=np.array([False, True]), 
    title='Confusion matrix')
print('% of Lookie-loos(come back, but dont transact)', cm[0, 0]/cm.sum())
print('% of Customers Who transacted in Lookback Window and Target Window', cm[1, 1]/cm[1, :].sum())
print('% of Customers Who Appeared in Lookback Window and Transacted in Target Window', cm[0, 1]/cm[0, :].sum())


# # Create a Baseline Using All Zeros
# ### An all zeroes model performs pretty well, given that 

# In[ ]:


y_test = targets.target_revenue_log1p.values
print('RMSE on Test Set:', mean_squared_error(y_test, np.zeros_like(y_test))**0.5)


# # Analyzing Date Partitions From a Winning Post (Can use a similar technique for cross validation and training of the model??)
# The below is examining how a top 40 solution ended up "chunking" the data sets into multiple test/train periods for analysis
# https://www.kaggle.com/augustmarvel/base-model-v2-user-level-solution
# 
# #### Time period
# 
# the training set has a 45 days gap to its target set that is same as the test set
# the training set has almost the same duration as the test set
# the valiation set is set to Dec-Jan which is the same monthly period as the target peroid of the test set
# 
# 210 days of training period, 45 days of gap period, 2 months of traget perod.

# In[ ]:


target_period = pd.date_range(start='2016-08-01',end='2018-12-01', freq='2MS')
train_period = target_period.to_series().shift(periods=-210, freq='d',axis= 0)
time_to = train_period[train_period.index>pd.Timestamp('2016-08-01')]
time_end = target_period.to_series().shift(periods=-45, freq='d',axis= 0)[4:]


# In[ ]:


print('Target Period Every 2 Months')
print(target_period)
print()
print('train_period 210 day lookback from target period')
print(train_period)
print(train_period.shape)
print()
print('time_to filter train_period for windows that start since (August 1st 2016)')
print(time_to)
print()
print('time_end 45 Day windows for Target Predictions From the end period in train_period')
print(time_end)


# In[ ]:


train_df.date.min(), train_df.date.max(), test_df.date.min(), test_df.date.max()


# In[ ]:


alist = list(range(time_to.shape[0]-1))

for i in alist:
    print(i, 'Train(X) Date Range :', '  From:', time_to.index[i],  '  To:', time_end.index[i], '  Range:', time_end.index[i]-time_to.index[i])
    print(i, 'Gap Period(null)    :', '  From:', time_end.index[i], '  To:', time_to.iloc[i],   '  Range:', time_to.iloc[i] - time_end.index[i])
    print(i, 'Target(Y) Date Range:', '  From:', time_to.iloc[i],   '  To:', time_to.iloc[i+1], '  Range:', time_to.iloc[i+1] - time_to.iloc[i])
    print()


# # First Model

# In[ ]:


#Features are just going to get the median, max, min, values
DATE_COLUMNS = [
    'day', #removing features to cut back on memory
    #consider making one hot if performance drops
]

ONE_HOT_COLUMNS = [
    'dayofweek',
    'month',
    #'hour', #removing features to cut back on memory
    #'weekofyear', 
    #'year' always the same....
    'channelGrouping',
    #'device_browser', #Removed: Too Many Features
    'device_deviceCategory',
    'device_isMobile',
#    'geoNetwork_country', 
#    'trafficSource_adwordsClickInfo.page', #these fell to the bottom of feature importances, sum and mean... just ditiching
    #'trafficSource_adwordsClickInfo.isVideoAd', #Removed: All False All True In Fold
    'trafficSource_isTrueDirect', #Removed: All True In Fold
]

NUMERIC_FEAT_COLUMNS = [
    'totals_hits',
    'totals_pageviews',
    'totals_timeOnSite',
    'totals_totalTransactionRevenue.div1M',
    'totals_totalTransactionRevenue.log1p', #Added feature and remove native values so that you can convert fetures to int32 
    'totals_transactions'
]

# for these columns will just choose the most frequently occuring one by user....
LABEL_ENCODE_COLUMNS = [
    'geoNetwork_country',
    'geoNetwork_subContinent',
    'device_operatingSystem'
]


# In[ ]:


from sklearn.preprocessing import LabelEncoder

def get_label_encoded_features(df):
    tables = []
    le  = LabelEncoder()
    for column in LABEL_ENCODE_COLUMNS:
        encoded_labels = le.fit_transform(df[column])
        tables.append( pd.DataFrame({(column + '.encoded'): encoded_labels}).set_index(df.index) )
    return pd.concat(tables, axis=1)


# In[ ]:


def get_one_hot_features(df):
    """
    One hot encode categorical features...
    """
    tables = []
    for col in ONE_HOT_COLUMNS:
        tables.append( pd.get_dummies(df[col].fillna('NA')).add_prefix(col + '.') )
    return pd.concat(tables,axis=1)

# def get_date_columns(df):
#     tables = []
#     for col in DATE_COLUMNS:
#         tables.append( pd.get_dummies(df[col].fillna('NA')).add_prefix(col + '.') )
#     return pd.concat(tables,axis=1)

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def get_rececency(df, end_of_training_window_date):
    df['session_recency'] = (end_of_training_window_date - df['date']).dt.days
    recency = df.groupby('fullVisitorId')['session_recency'].agg(['min', 
                                                                  'max', 
                                                                  'mean',
                                                                  'median',
                                                                  'std',
                                                                  'skew', 
                                                                  #'median', These Stats are pretty slow to calculate, increaese run time 10x, all of the sorting??---
                                                                 percentile(25), percentile(75)
                                                                 ]).add_prefix('session_recency_')
    recency['session_recency_diff'] = recency['session_recency_max'] - recency['session_recency_min']
#    recency['session_recency_skew'] = (3 * (recency['session_recency_mean'] - recency['session_recency_median']))/recency['session_recency_std']
    recency['session_recency_iqr'] = recency['session_recency_percentile_75'] - recency['session_recency_percentile_25']
    
    return recency


def add_calculted_features(df):
    df['totals_totalTransactionRevenue.log1p'] = np.log1p(df['totals_totalTransactionRevenue'].values)
    df['totals_totalTransactionRevenue.div1M'] = df['totals_totalTransactionRevenue'].values/(10**6)
    return df


# In[ ]:


def feat_targets(df, split_date, lookback_window=DAYS_LOOK_BACK, target_fwd_window=DAYS_PREDICT_FORWARD):
    target_col = 'totals_totalTransactionRevenue'
    train_start_date = split_date + pd.Timedelta(days=-lookback_window)
    target_end_date = split_date + pd.Timedelta(days=+target_fwd_window)
    print('Date Range of Dataset', df.date.min(), df.date.max())
    print('lookback_window', lookback_window, 'target_fwd_window', target_fwd_window)
    print('train_start_date', train_start_date)
    print('split_date', split_date)
    print('target_end_date', target_end_date)
    print()
    if (train_start_date < df.date.min()) or (target_end_date > df.date.max()):
        raise ValueError('Periods are outside of dataframe time range')
    fold_train = df[(df.date >= train_start_date) & (df.date < split_date)]
    print('train at sessions level shape', fold_train.shape)
    #print('removing duplicate sessions')
    
    fold_val = df[(df.date >= split_date) & (df.date <= target_end_date)]
    fold_val_target = fold_val.groupby('fullVisitorId')[target_col].sum().to_frame()
    print('val agg by user shape', fold_val_target.shape)
    del fold_val
    gc.collect()
    
    print('Encoding session level features')
    print('adding calculated features')
    fold_train = add_calculted_features(fold_train)
    print('one_hot_features')
    one_hot_features = get_one_hot_features(fold_train)
    print('label_encoded_features')
    label_encoded_features = get_label_encoded_features(fold_train)
    
    print('creating session level features')
    # get session level features
    session_x = pd.concat([
        fold_train[['fullVisitorId'] + NUMERIC_FEAT_COLUMNS + DATE_COLUMNS],
#         date_features, 
         one_hot_features, 
         label_encoded_features
        ], axis=1, sort=True)
    print('session_x', session_x.shape)
    
    sum_cols = one_hot_features.columns.tolist() + NUMERIC_FEAT_COLUMNS
    mean_cols = one_hot_features.columns.tolist() + NUMERIC_FEAT_COLUMNS
    min_cols = NUMERIC_FEAT_COLUMNS + DATE_COLUMNS
    max_cols = NUMERIC_FEAT_COLUMNS + DATE_COLUMNS
    std_cols = NUMERIC_FEAT_COLUMNS
    skew_cols = NUMERIC_FEAT_COLUMNS
    median_cols = label_encoded_features.columns.tolist() + DATE_COLUMNS #these should be the same for all users

    print('aggregating session level features to user level')
    #aggregate session features by user
    train_x = pd.concat([
        
        session_x['fullVisitorId'].value_counts().to_frame(name='session_count'), 
        get_rececency(fold_train, split_date), #done to calculate recency stats

        session_x.groupby('fullVisitorId')[sum_cols].sum().add_suffix('_sum'), #this will handle frequency/monetary vaue
        session_x.groupby('fullVisitorId')[mean_cols].mean().add_suffix('_mean'),
        session_x.groupby('fullVisitorId')[min_cols].max().add_suffix('_min'),
        session_x.groupby('fullVisitorId')[max_cols].max().add_suffix('_max'),
        session_x.groupby('fullVisitorId')[std_cols].std().add_suffix('_std'),

        session_x.groupby('fullVisitorId')[median_cols].median().add_suffix('_median'),
        session_x.groupby('fullVisitorId')[skew_cols].skew().add_suffix('_skew'), #this made performance notably worse, for median/skew/percentile, it has to sort so O(n*log(n)
    ], axis = 1, sort=True) \
        .fillna(0) \
        .astype('int32') #this had a big effect on memory!!
    del session_x, one_hot_features, label_encoded_features
    gc.collect()
    
    print('getting target values')
    # get target for each user from fold_val, left join on a series from the train dataset to get all users in train and any target from fold_val
    merged=train_x['session_count'].to_frame().join(fold_val_target, how='left')
    train_y = merged[target_col].to_frame(name = 'target_revenue')
    train_y['is_returning'] = train_y.target_revenue.notna()
    train_y.fillna(0, inplace=True)
    
    print('Output shapes', 'X', train_x.shape, 'y', train_y.shape)
    gc.collect()
    return train_x, train_y


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_X, train_y = feat_targets(train_df, split_date=pd.Timestamp('2017-09-30'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "val_X, val_y = feat_targets(train_df, split_date=pd.Timestamp('2017-12-31'))")


# In[ ]:


#TODO: correct for non-overlapping features, there are a handful of features in val not in train and vice versa
feature_overlap = sorted(list(set(train_X.columns).intersection(val_X.columns)))
val_X = val_X[feature_overlap]
train_X = train_X[feature_overlap]


# In[ ]:


print('Baseline all zeros', mean_squared_error(np.log1p(val_y.target_revenue.values), np.zeros_like(val_y.target_revenue.values))**0.5)


# In[ ]:


#import lightgbm as lgb
# setting taken from here: https://www.kaggle.com/augustmarvel/base-model-v2-user-level-solution
from xgboost import XGBRegressor
xgb_params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456,
        'importance_type': 'total_gain'
    }

xgb = XGBRegressor(**xgb_params, n_estimators=1500)
xgb.fit(train_X, np.log1p(train_y.target_revenue.values),eval_set=[(val_X, np.log1p(val_y.target_revenue.values))],early_stopping_rounds=25,eval_metric='rmse',verbose=25)


# # Feature Importances / Threshold Analysis

# In[ ]:


feat_imp_df=pd.DataFrame({
                'feature': train_X.columns.tolist(),
                'importance': xgb.feature_importances_
            })
print('Total Number of Features', feat_imp_df.shape)


# In[ ]:


plt.figure(figsize=(8, 12))

sns.barplot(x='importance', y='feature', 
            data=pd.DataFrame({
                'feature': train_X.columns.tolist(),
                'importance': xgb.feature_importances_
            }).sort_values('importance', ascending=False) \
              .iloc[:50]
           )
plt.title('Top 50 Features')


# In[ ]:


feat_imp_df.sort_values('importance', ascending=True).iloc[:25]


# In[ ]:


preds = xgb.predict(val_X)


# In[ ]:


print('RMSE From model', mean_squared_error(np.log1p(val_y.target_revenue.values), preds)**0.5)
preds[preds < 0] = 0.0
print('RMSE From model with - preds set to zero', mean_squared_error(np.log1p(val_y.target_revenue.values), preds)**0.5)


# In[ ]:


pd.DataFrame({'actuals': np.log1p(val_y.target_revenue.values), 'predictions': preds})     .sort_values('predictions', ascending=False)     .head(n = 100)     .plot.scatter(x = 'predictions', y = 'actuals')


# In[ ]:


actual_preds_df = pd.DataFrame({'actuals': np.log1p(val_y.target_revenue.values), 'predictions': preds})
highest_pred = actual_preds_df     .sort_values('predictions', ascending=False)     .head(n = 25)
highest_pred


# In[ ]:


val_X.iloc[highest_pred.index, :]


# In[ ]:


lbs = []
num_samples = []
perc_transacteds = []
for lb in np.linspace(actual_preds_df.predictions.min(), actual_preds_df.predictions.quantile(1.0), 50):
    actuals_above_threshold = actual_preds_df[actual_preds_df.predictions >= lb]
    num_sample = actuals_above_threshold.shape[0]
    perc_transacted = (actuals_above_threshold.actuals > 0).mean()
    lbs.append(lb)
    num_samples.append(num_sample)
    perc_transacteds.append(perc_transacted)
    print('lb', round(lb, 4), 'num samples', num_sample, '% of users that transacted', round(perc_transacted, 4))


# In[ ]:


#sns.lineplot(x = np.array(lbs)[1:], y=np.array(perc_transacteds)[1:])
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax1.plot(lbs[1:], perc_transacteds[1:], color="r")

ax2.plot(lbs[1:], num_samples[1:], color="b")
ax2.set_yscale('log')

ax1.set_ylabel('% of Customers that Purchased >= Threshold', color='red')
ax2.set_ylabel('Number of Customers with Prediction >= Threshold', color='blue')
ax1.set_xlabel('Threshold')
plt.show()


# In[ ]:




