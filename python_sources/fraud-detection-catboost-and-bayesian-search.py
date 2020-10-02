#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings(action='ignore',category = DeprecationWarning)
warnings.simplefilter(action='ignore',category = DeprecationWarning)

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import re

import os
import time
import datetime
import gc
import shutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn import metrics
from sklearn.utils import class_weight

import catboost as cb

import tensorflow as tf

import skopt.plots
import scikitplot as skplt

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Installing the most recent version of skopt directly from Github (from fork which have some bugs fixed)
get_ipython().system('pip install git+https://github.com/darenr/scikit-optimize')

import skopt.plots

# Custom codes to make code clear
get_ipython().run_line_magic('run', '../input/imports/help_functions.py')
get_ipython().run_line_magic('run', '../input/imports/fixed_bayes_search.py')

LOCAL = False
fraud_data_dir = '../input/ieee-fraud-detection/'


# In[ ]:


data_dir


# In[ ]:


data_version = 'v5'
if LOCAL:
    data_dir = fraud_data_dir + 'data/data_' + data_version + '/'
else:
    data_dir = 'data/data_' + data_version + '/'

data_available = os.path.exists(data_dir + 'data.pkl') and os.path.exists(data_dir + 'submission.pkl')
load_data = False

use_catboost = True


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)


# ### Load data

# In[ ]:


def load_only_data(data_dir, load_data = False):
    
    if load_data or not os.path.exists(data_dir + 'data.pkl'):
        print("Loading data from CSV files")
        data_id = pd.read_csv(fraud_data_dir + 'train_identity.csv')
        data_trans = pd.read_csv(fraud_data_dir + 'train_transaction.csv')

        print("Merging data")
        data = pd.merge(data_trans, data_id, on='TransactionID', how='left')

        # Ensure that all columns use the proper data type
        data = reduce_mem_usage(data)
        
        print(f"Shape of data_id is: {data_id.shape}")
        print(f"Shape of data_trans is: {data_trans.shape}")
        print(f"Shape of data is: {data.shape}")

    else:
        print("Loading data from ",data_dir)

        with open(data_dir + 'data.pkl', 'rb') as file:
            data = pickle.load(file)

        # Submission deleted and not loaded in order to keep memory lower, and load it when needed
        #with open(data_dir + 'submission.pkl', 'rb') as file:
        #    submission = pickle.load(file)

        #if 'submission' in locals():
        #    del submission

        print("Data loaded")
        
    return data

def load_submission(data_dir, load_data = False):
    if load_data or not os.path.exists(data_dir + 'submission.pkl'):
        print("Loading submission from CSV files")
        submission_id = pd.read_csv(fraud_data_dir + 'test_identity.csv')
        submission_trans = pd.read_csv(fraud_data_dir + 'test_transaction.csv')
        
        print("Merging submission")
        submission = pd.merge(submission_trans, submission_id, on='TransactionID', how='left')
        
        # Ensure that all columns use the proper data type
        submission = reduce_mem_usage(submission)
        
        print(f"Shape of submission_id is: {submission_id.shape}")
        print(f"Shape of submission_trans is: {submission_trans.shape}")
        print(f"Shape of submission is: {submission.shape}")
    else:
        print("Loading submission from ",data_dir)
        
        with open(data_dir + 'submission.pkl', 'rb') as file:
            submission = pickle.load(file)

        print("Data loaded")
    
    return submission

data = load_only_data(data_dir, load_data)
#submission = load_submission(data_dir, load_data)


# In[ ]:


data.head()


# In[ ]:


print(f"There are {data.isnull().any().sum()} columns with missing values")


# In[ ]:


def remove_unique_and_NaN_columns(df):
    too_much_NaN_cols = [col for col in df.columns if df[col].isnull().mean() > 0.9]
    one_value_per_column = [col for col in df.columns if df[col].nunique() <= 1]

    remove_columns = too_much_NaN_cols + one_value_per_column

    print("The following columns are removed: ", remove_columns)
    return remove_columns


# ### Target variable analysis
# 
# Target data 'isFraud' is highly imbalanced, so we are not interested on accuracy, but ROC AUC.

# In[ ]:


data['isFraud'].value_counts(normalize = True)


# In[ ]:


print(f"{data['id_01'].isnull().mean().round(4) * 100}% of transactions does not have associated an identity")


# ### Rebalance data
# As the data is such unbalanced, we need to do some treatment here. There are two types of sampling, (1) downsampling (removing data from the dominant target class), (2) upsampling (duplicating data from the target class in minority). In this case we prefered downsampling the data.

# In[ ]:


if load_data or not os.path.exists(data_dir + 'data.pkl'):
    # Negative downsampling
    data_pos = data[data['isFraud']==1]
    data_neg = data[data['isFraud']==0]

    data_neg = data_neg.sample(3*int(data_pos.shape[0] ), random_state=42)
    data = pd.concat([data_pos,data_neg]).sort_index()
    print(f"Shape of data is: {data.shape}")


# ### Feature engineering
# 
# We are going to create several feature engineering treatments as:
# 
# 1) Identify TransactionID which does match with identity table
# 
# 2) Create new features from TransactionDT data
# 
# 3) Transform some float data with log, square or root square in case they are skewed
# 
# 4) Create new features from DeviceInfo data
# 
# 5) Divide by meaning grouping by different columns
# 
# 6) Replace -np.inf and np.inf for np.nan
# 
# 7) New columns which check where there are original NaN values
# 
# 8) Replace NaN values with mean of grouped data from columns of float64 type 

# In[ ]:


def unique_identifier(df):
    df['uid'] =  df['card1'].astype(str)+'_'+df['card2'].astype(str)
    df['uid_1'] =  df['uid'].astype(str)+'_'+df['card3'].astype(str)
    df['uid_2'] =  df['uid_1'].astype(str)+'_'+df['card5'].astype(str)
    df['uid_3'] =  df['uid_2'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
    df['uid_4'] = df['card4'].astype(str)+'_'+df['card6'].astype(str)
    df['uid_5'] = df['uid_3'].astype(str)+'_'+df['uid_4'].astype(str)


# In[ ]:


def transactionDT(df):
    # As commented in the forum discussion, this column probably is shown in seconds
    # If we count as initial moment the first entry which is 86400, we can create some kind of clock
    df['seconds'] = (df['TransactionDT'] % 60).astype('int8')
    df['minutes'] = ((df['TransactionDT'] / 60) % 60).astype('int8')
    df['hours'] = ((df['TransactionDT'] / 3600) % 24).astype('int8')
    df['days_week'] = ((df['TransactionDT'] / 86400) % 7).astype('int8')
    df['days_year'] = ((df['TransactionDT'] / 86400) % 365).astype('int16')
    df['harmonic_seconds'] =  make_harmonic_features(df['seconds'], 60)[1].astype('float32')
    df['harmonic_minutes'] =  make_harmonic_features(df['minutes'], 60)[1].astype('float32')
    df['harmonic_hours'] =  make_harmonic_features(df['hours'], 24)[1].astype('float32')
    df['harmonic_days_week'] =  make_harmonic_features(df['days_week'], 7)[1].astype('float32')
    df['harmonic_days_year'] =  make_harmonic_features(df['days_year'], 365)[1].astype('float32')
    
# As a clock pass from 23 to 0 and they are near as 13 to 14, we use an harmonic clock using cos() and sin()
def make_harmonic_features(value, period=24):
    return np.cos(value * 2 * np.pi / period), np.sin(value * 2 * np.pi / period)    


# In[ ]:


def time_from_fraud(df, isSubmission = False, time_from_fraud_dict = {}):
    group_by_columns = ['uid']
    columns_to_treat = ['minutes_from_last_fraud', 'hours_from_last_fraud']
    
    if not isSubmission:
        df['last_fraud'] = df['TransactionDT'][df['isFraud'] == 1]
        df['last_fraud'] = df['last_fraud'].shift(1)
        df['last_fraud'] = df['last_fraud'].fillna(method='ffill')

        df['minutes_from_last_fraud'] = ((df['TransactionDT'] - df['last_fraud']) / 60).round(0).astype('Int32')
        df['hours_from_last_fraud'] = ((df['TransactionDT'] - df['last_fraud']) / 3600).round(0).astype('Int32')

        # Create median column of time from last fraud by UID        
        for column_to_treat in tqdm_notebook(columns_to_treat):
            df['median_' + column_to_treat] = df.groupby(group_by_columns)[column_to_treat].transform('median')

            # For the few cases where all the times the UID is NaN we take the median of the whole feature
            df.loc[df['median_' + column_to_treat].isnull(), 'median_' + column_to_treat + '_check'] = 1
            df['median_' + column_to_treat + '_check'] = df['median_' + column_to_treat + '_check'].fillna(0)
            df.loc[df['median_' + column_to_treat].isnull(), 'median_' + column_to_treat] = df[column_to_treat].median()

            #time_from_fraud_dict = df.groupby(group_by_columns)['median_' + column_to_treat]
        df.drop(columns_to_treat + ['last_fraud'], axis = 1, inplace = True)
    else:
        
        df_medians = pd.DataFrame(df[group_by_columns],index = df.index)

        values_group_by = data.groupby(group_by_columns)[group_by_columns].transform('min').values
        values_group_by = values_group_by.reshape(len(values_group_by))

        # Create median column of time from last fraud by UID        
        for column_to_treat in tqdm_notebook(columns_to_treat):
            data_medians = pd.DataFrame(index = values_group_by)
            data_medians['median_' + column_to_treat] = data.groupby(group_by_columns)['median_' + column_to_treat].transform('median').values

            # For the few cases where all the times the UID is NaN we take the median used in data, which are the ones that have a check (all are the same, so we can use min() to get the value)
            data_medians.loc[data_medians['median_' + column_to_treat].isnull(), 'median_' + column_to_treat] = min(data.loc[data['median_' + column_to_treat + '_check'] == 1 ,'median_' + column_to_treat])
            data_medians = data_medians.drop_duplicates()

            # Now merge and set the new column to the output
            df_medians = df_medians.merge(data_medians, how = 'left', left_on = group_by_columns, right_index = True)
            df['median_' + column_to_treat] = df_medians['median_' + column_to_treat].copy()            


# In[ ]:


def add_check_null_columns(df):
    nan_pd = df.isnull().copy()
    nan_pd = nan_pd.T[nan_pd.any()].T
    nan_pd.columns = [column + '_isnull' if not '_isnull' in column else column for column in nan_pd.columns]
    df = df.merge(nan_pd, left_index = True, right_index = True)
    
def add_check_null_rows(df):
    bins = [0, 1, 5, 10, 25, 50, 75, 100]
    df['all_NaN_bins'] = pd.cut(abs(df.isna().sum(axis=1).astype(np.int8)), bins=bins)
    if data_version < 'v5':
        df['all_NaN'] = abs(df.isna().sum(axis=1).astype(np.int8))


# In[ ]:


def select_only_top_values(df, col, n_top_values, others_value = 'Others', nan_value = 'NaN'):
    value_counts = df[col].value_counts(dropna = True)
    if len(value_counts) > n_top_values:
        top_value_counts = value_counts.iloc[:n_top_values].index.values
        others_value_counts = value_counts.iloc[n_top_values:].index.values
        col_dict = dict(zip(list(top_value_counts) + list(others_value_counts), list(top_value_counts) + [others_value,] * len(value_counts)))
    else:
        col_dict = dict(zip(value_counts.index.values, value_counts.index.values))

    col_dict[np.nan] = nan_value
    return col_dict


# In[ ]:


def device_info_featuring(df, device_dict = {}, isSubmission = False, n_top_values = 100):
    # DeviceInfo_1 the main part of the DeviceInfo and DeviceInfo_2 gets the build version 
    df['DeviceInfo_1'] = df.DeviceInfo.str.split(" Build/", n=1, expand=True)[0]
    df['DeviceInfo_2'] = df.DeviceInfo.str.split(" Build/", n=1, expand=True)[1]
    
    # Identify IE browser
    ie_cond1 = df['DeviceInfo'].str.startswith('rv:11', na=False)
    ie_cond2 = df['DeviceInfo'].str.startswith('Trident/', na=False)
    df.loc[ie_cond2 & ie_cond2, 'DeviceInfo_1'] = "IE" 
    
    # Identify Mozilla browser
    mozilla_cond1 = df['DeviceInfo'].str.startswith('rv:', na=False)
    df.loc[mozilla_cond1 & ~ie_cond1,'DeviceInfo_1'] = "MOZILLA" 
    
    # DeviceInfo_3 get the first part of the DeviceInfo_1, till it finds one of the symbols
    df['DeviceInfo_3'] = df['DeviceInfo_1'].copy()
    for symbol in [" ", "-", "_", "(", "/"]:
        df['DeviceInfo_3'] = df['DeviceInfo_3'].str.extract(r'([^{}]+)'.format(symbol))
     
    # DeviceInfo_3 refining, shorting some brands
    brands = ['HTC', 'IdeaTab', 'KF', 'LG', 'Lenovo', 'ME', 'RCT', 'SGP', 'XT', 'verykool']
    df['DeviceInfo_3'] = df['DeviceInfo_3'].str.extract(r'({})'.format('^' + '|^'.join(brands)))
        
    # DeviceInfo_4 returns the main part of build version
    device_info_build_featuring(df)

    for col in ['DeviceInfo_1','DeviceInfo_2','DeviceInfo_3','DeviceInfo_4']:
        device_dict[col] = select_only_top_values(df, col, n_top_values)
        df[col] = df[col].map(device_dict[col]).fillna('Others')
    
    return device_dict
    
def device_info_build_featuring(df):
    cond1 = r'\bHUAWEI[a-z|A-Z]+[^-]+' # Starts with HUAWEI
    cond2 = r'^[a-z|A-Z]{3}' # Starts with 3 letters
    cond3 = r'^[0-9]+[.][0-9]+[.][a-z|A-Z]' # Starts with format like 33.4.A or 6.5.A

    condition = r'({})'.format('|'.join([cond1,cond2,cond3]))

    df['DeviceInfo_4'] = df['DeviceInfo_2'].str.upper()
    df['DeviceInfo_4'] = df['DeviceInfo_2'].str.extract(condition, flags = re.IGNORECASE)

    # Rest copy the same
    cond_rest = (df['DeviceInfo_2'].notnull()) & (df['DeviceInfo_4'].isnull())
    df.loc[cond_rest, 'DeviceInfo_4'] = df['DeviceInfo_2'].copy()


# In[ ]:


def emails_domains_featuring(df, n_top_domains = 10, emails_dict = {}, isSubmission = False):
    P_cols = ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
    R_cols = ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
    df[P_cols] = df['P_emaildomain'].str.split('.', expand=True)
    df[R_cols] = df['R_emaildomain'].str.split('.', expand=True)

    for col in P_cols + R_cols:
        if not isSubmission:
            emails_dict[col] = select_only_top_values(df, col, n_top_domains)

        df[col] = df[col].map(emails_dict[col]).fillna('Others')
    
    return emails_dict


# In[ ]:


def divide_mean_by_grouping(df, column_to_treat, group_by_column, isSubmission = False, group_by_means_dict = {}):
    if not isSubmission:
        group_by_means_dict[column_to_treat + "_to_mean_by_" + group_by_column] = df.groupby(group_by_column)[column_to_treat].mean()
    
    df[column_to_treat + "_to_mean_by_" + group_by_column] = df[column_to_treat].astype('float64') / df.merge(group_by_means_dict[column_to_treat + "_to_mean_by_" + group_by_column], how = 'left', left_on = group_by_column, suffixes = ('','_new'), right_index = True)[column_to_treat + '_new']
    
    return group_by_means_dict


# In[ ]:


def clean_inf_nan(df):
    float_cols = df.dtypes[df.dtypes.astype(str).str.startswith('float')].index.values
    for col in tqdm_notebook(float_cols):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)


# In[ ]:


def treat_NaN(df, group_by_columns):
    cond1 = df.isnull().any()
    cond2 = df.dtypes.astype(str).str.startswith('float')
    cond3 = df.dtypes == 'object'
    
    float_cols_with_nulls = df.dtypes.loc[cond1 & cond2].index.values
    
    for col in tqdm_notebook(float_cols_with_nulls):
        treat_NaN_by_groups(df, col, group_by_columns)
        
    # For categorical columns which have NaN, we replace the np.nan for 'NaN' which is accepted as a new category
    object_cols_with_nulls = df.dtypes.loc[cond1 & cond3].index.values
    
    for col in tqdm_notebook(object_cols_with_nulls):
        df.loc[df[col].isnull(), col] = 'NaN'
        

def treat_NaN_by_groups(df, column_to_treat, group_by_columns):
    # If percentage of outliers is less than the threshold, then use mean, otherwise median
    threshold_mean_median = 0.01
    threshold_outliers = 3
    mask_outliers = (np.abs(stats.zscore(df[column_to_treat][df[column_to_treat].notnull()])) > threshold_outliers)
    if mask_outliers.sum() / len(mask_outliers) > threshold_mean_median:
        # Firstly trying to set the NaN value from group mean
        df.loc[df[column_to_treat].isnull(), column_to_treat] = df.groupby(group_by_columns)[column_to_treat].transform('mean')
        # The remaining NaN will have mean value
        df.loc[df[column_to_treat].isnull(), column_to_treat] = df[column_to_treat].mean()
    else:
        # Firstly trying to set the NaN value from group median
        df.loc[df[column_to_treat].isnull(), column_to_treat] = df.groupby(group_by_columns)[column_to_treat].transform('median')
        # The remaining NaN will have median value
        df.loc[df[column_to_treat].isnull(), column_to_treat] = df[column_to_treat].median()


# In[ ]:


def skewed_data_transformation(df, columns_to_log):
    
    # Only do transformation on columns where all values are positive
    df_positive = df[(df[columns_to_log] >= 0).all().index.values].copy()
    
    # Set type to be sure there is no problem of limitation by data type (later they will be reset depending on the values)
    int_cols = df_positive.dtypes[df_positive.dtypes.astype(str).str.startswith('int')].index.values
    float_cols = df_positive.dtypes[df_positive.dtypes.astype(str).str.startswith('float')].index.values
    df_positive.loc[:, int_cols] = df_positive[int_cols].astype('int64')
    df_positive.loc[:, float_cols] = df_positive[float_cols].astype('float64')
    
    # Log transformation
    df_log = df_positive.apply(np.log).replace([np.inf, -np.inf], 0)
    df_log.columns = df_log.columns.map(lambda x : 'log_' + str(x))
    df = df.merge(df_log, how = 'left', left_index = True, right_index = True)
    
    # Square transformation
    df_square = df_positive.pow(2)
    df_square.columns = df_square.columns.map(lambda x : 'square_' + str(x))
    df = df.merge(df_square, how = 'left', left_index = True, right_index = True)
    
    # Transformation for left skewed data
    df_left_skew = -df_positive.pow(-1/2).replace([np.inf, -np.inf], 0)
    df_left_skew.columns = df_left_skew.columns.map(lambda x : 'left_skew_' + str(x))
    df = df.merge(df_left_skew, how = 'left', left_index = True, right_index = True)
    
    return df


# In[ ]:


def feature_engineering(df, isSubmission = False, emails_dict = {}, n_top_domains = 10, group_by_means_dict = {}, device_dict = {}, n_top_values = 100):
    print('Starting feature engineering')
    
    #Sort data by TransactionDT
    print('Sorting data by TransactionDT')
    df = df.sort_values('TransactionDT')
    
    # Featuring Unique identifier
    print('Creating Unique identifiers')
    unique_identifier(df)
    
    # Identify TransactionID which does match with identity table
    print('Featuring "Transaction_match_identity"...')
    df['Transaction_match_identity'] = ~ df['id_01'].isnull()
    
    # Extraction of information from TransactionDT
    print('Featuring "TransactionDT"...')
    transactionDT(df)
    time_from_fraud(df, isSubmission)
    
    # Matematical transformations of following columns:
    columns_to_log = ['TransactionAmt', 'dist1', 'id_17', 'id_19', 'id_20', 'C13', 'C1', 'V91', 'addr1', 'C14', 'V317', 'V258', 'D1', 'C6', 'D2', 'C4', 'V310', 'C5', 'C9', 'C11']
    df = skewed_data_transformation(df, columns_to_log)
    
    # Extract info from DeviceInfo
    print('Featuring "DeviceInfo"...')
    device_dict = device_info_featuring(df, device_dict, isSubmission, n_top_values)
    
    # Extract info from emails domains
    print('Featuring "Emails domain...')
    emails_dict = emails_domains_featuring(df, n_top_domains = n_top_domains, emails_dict = emails_dict, isSubmission = isSubmission)
    
    #Divide by meaning grouping by different columns
    print('Featuring "Divide by mean"...')
    columns_to_mean = ['TransactionAmt', 'dist1', 'id_17', 'id_19', 'id_20', 'C13', 'C1', 'V91', 'addr1', 'C14', 'V317', 'V258', 'D1', 'C6', 'D2', 'C4', 'V310', 'C5', 'C9', 'C11']
    group_by_columns = ['ProductCD','uid', 'uid_1', 'uid_2', 'uid_3', 'uid_4', 'uid_5', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6','P_emaildomain', 'DeviceInfo', 'M3', 'M4', 'M5', 'M6', 'P_emaildomain', 'R_emaildomain', 'id_19', 'id_20', 'id_31', 'id_33', 'hours', 'days_week', 'days_year']
    for group_by in tqdm_notebook(group_by_columns, desc = '1st Loop'):
        for col in tqdm_notebook(columns_to_mean, desc = '2nd Loop', leave = False):
            group_by_means_dict = divide_mean_by_grouping(df, col, group_by, isSubmission = isSubmission, group_by_means_dict = group_by_means_dict)
    
    # Replace -np.inf and np.inf for np.nan
    clean_inf_nan(df)
    
    # Add a column which check if the original had null values (only for columns which have null values)
    print('Featuring "add_check_null_columns"')
    add_check_null_columns(df)
    print('Featuring "add_check_null_rows"')
    add_check_null_rows(df)
    
    # Replace NaN values with mean of grouped data from columns which have nulls and are float64 type
    print('Featuring "Replace NaN on float64"...')
    group_by_columns = ['uid']
    #group_by_columns = ['ProductCD', 'card1']
    treat_NaN(df, group_by_columns)
    
    return df, emails_dict, group_by_means_dict, device_dict


# ### Encode categorical data
# 
# Transform all the categorical data into LabelEncoder (this will only be used if we are not using CatBoost algorithm)

# In[ ]:


def get_categorical_columns():
    
    categorical_columns = data.dtypes.iloc[np.where(~data.dtypes.astype(str).str.startswith('float'))[0]].index.values
    #categorical_columns = ['ProductCD',
    #     'card1','card2','card3','card4','card5','card6',
    #     'P_emaildomain','R_emaildomain',
    #     'M1','M2','M3','M4','M5','M6','M7','M8','M9',
    #     'id_12','id_13','id_14','id_15','id_16','id_17','id_18','id_19','id_20','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38',
    #     'DeviceType',
    #     'DeviceInfo','DeviceInfo_1','DeviceInfo_2','DeviceInfo_3','DeviceInfo_4',
    #     'seconds', 'minutes', 'hours', 'days_week', 'days_year', 
    #     'minutes_from_last_fraud', 'hours_from_last_fraud', 'median_minutes_from_last_fraud', 'median_hours_from_last_fraud'
    #     'uid', 'uid_1', 'uid_2', 'uid_3', 'uid_4', 'uid_5']

    return categorical_columns


# In[ ]:


def encode_categorical_data(data_df, submission_df):
    #categorical_columns = [col for col,col_type in train.dtypes.items() if col_type == 'object']

    print("Treating categorical columns")
    for col in tqdm_notebook(categorical_columns):
        if col in data_df.columns.values:
            #print(f"Treating column {col}, {categorical_columns.index(col) + 1} out of {len(categorical_columns)}")
            le = LabelEncoder()
            le.fit(list(data_df[col].astype(str).str.upper().values) + list(submission_df[col].astype(str).str.upper().values))
            data_df[col] = le.transform(list(data_df[col].astype(str).str.upper().values))
            submission_df[col] = le.transform(list(submission_df[col].astype(str).str.upper().values))


# ### Execute Data Treatment

# In[ ]:


get_ipython().run_cell_magic('time', '', 'if load_data or not data_available:\n    print("Starting data treatments")\n    try:\n        \n        remove_columns = remove_unique_and_NaN_columns(data)\n        data = data.drop(remove_columns, axis = 1)\n\n        n_top_domains = 10\n        n_top_values = 100\n        data, emails_dict, group_by_means_dict, device_dict = feature_engineering(data, n_top_domains  = n_top_domains, n_top_values = n_top_values)\n\n        # Ensure that all columns use the proper data type\n        data = reduce_mem_usage(data)\n        \n        submission = load_submission(data_dir, load_data)\n        submission = submission.drop(remove_columns, axis = 1)\n        submission, _, _, _ = feature_engineering(submission, isSubmission = True, \n                                         emails_dict = emails_dict, n_top_domains = n_top_domains, \n                                         group_by_means_dict = group_by_means_dict,\n                                        device_dict = device_dict, n_top_values = n_top_values)\n\n        categorical_columns = get_categorical_columns()\n        \n        if not use_catboost:\n            %time encode_categorical_data(data, submission)\n\n        # Ensure that all columns use the proper data type\n        submission = reduce_mem_usage(submission)\n        data = reduce_mem_usage(data)\n    finally:\n        if not os.path.exists(data_dir):\n            os.makedirs(data_dir)\n\n        with open(data_dir + \'data.pkl\', \'wb\') as file:\n            print("Saving data in ", data_dir + \'data.pkl\')\n            %time pickle.dump(data, file)\n\n        # Submission saved and then deleted in order to keep memory lower\n        with open(data_dir + \'submission.pkl\', \'wb\') as file:\n            print("Saving data in ", data_dir + \'submission.pkl\')\n            %time pickle.dump(submission, file)\n        del submission\n        gc.collect()')


# ### Train and validation split
# 
# Split data into train, validation and test in order to build and evaluate correctly the models. We are not shuffling because the data is ordered by TransactionDT and we prefer to evaluate in such order to be more realistic.

# In[ ]:


train, test = train_test_split(data.sort_values('TransactionDT'), test_size = 0.1, shuffle = False)

#del data

train, valid = train_test_split(train, test_size = 0.1, shuffle = False)

y = train.copy()['isFraud']
X = train.copy().drop('isFraud', axis = 1)

del train

y_valid = valid.copy()['isFraud']
X_valid = valid.copy().drop('isFraud', axis = 1)

del valid

y_test = test.copy()['isFraud']
X_test = test.copy().drop('isFraud', axis = 1)

del test

#del submission

gc.collect()

x_to_remove = ['TransactionID', 'isFraud', 'TransactionDT']
x_to_remove += ['uid', 'uid_1','uid_2','uid_3','uid_4','uid_5','card1','card2','card3','card4','card5','card6']
x_to_remove += ['P_emaildomain', 'R_emaildomain', 'DeviceInfo']
x_to_remove += ['all_NaN_bins']
x_columns = [col for col in list(X.columns) if col not in x_to_remove]


# ## Training using CatBoost Classifier in a Bayesian Search
# 
# We are going to create a model usuing CatBoost Classifier and looking for the best hyperparameters using a Bayesian Search

# In[ ]:


do_training = True

use_predifined_params = True

n_ensemble = 5

if LOCAL:
    task_type = "GPU"
else:
    task_type = "CPU"

catBoost_models_dirs = []
BayesSearchCV_dirs = []
for dir_file in os.listdir(data_dir):
    if dir_file.startswith('CatBoostClassifier'):
        catBoost_models_dirs.append(data_dir + dir_file)
    elif dir_file.startswith('BayesSearchCV'):
        BayesSearchCV_dirs.append(data_dir + dir_file)
        
catBoost_models_dirs.reverse()


# ### Column Feature Selection

# In[ ]:


get_ipython().run_cell_magic('time', '', 'if do_training or len(catBoost_models_dirs) == 0:\n    \n    # Create 3 random features which will serve as baseline to reject features\n    baseline_features = [\'random_binary\', \'random_uniform\', \'random_integers\']\n    X = X.drop(baseline_features, axis = 1, errors = \'ignore\')\n    X[\'random_binary\'] = np.random.choice([0, 1], X.shape[0])\n    X[\'random_uniform\'] = np.random.uniform(0, 1, X.shape[0])\n    X[\'random_integers\'] = np.random.randint(0, X.shape[0] / 2, X.shape[0])\n    x_columns = [col for col in list(X.columns) if col not in x_to_remove]\n    \n    # Get the indexes for the categorical columns which CatBoost requires to out-perform other algorithms\n    cat_features_index = [x_columns.index(col) for col in categorical_columns if col in x_columns]\n\n    estimator = cb.CatBoostClassifier(iterations = 100,\n                              eval_metric = "AUC",\n                              cat_features = cat_features_index,\n                              #rsm = 0.3,\n                              scale_pos_weight = y.value_counts()[0] / y.value_counts()[1],\n                              task_type = task_type,\n                              metric_period = 50,\n                              verbose = False\n                           )\n    \n    n_top_features = None\n    \n    catboost_feature_selection, df_catboost_feature_selection = shadow_feature_selection(\n        estimator, y, X[x_columns], \n        baseline_features = baseline_features, n_top_features = n_top_features,\n        collinear_threshold = 0.98, cum_importance_threshold = 0.99,\n        max_loops = 100, n_iterations_mean = 3, times_no_change_features = 3,\n        need_cat_features_index = True, categorical_columns = categorical_columns,\n        plot_correlation = True)\n\n    print("Features selected:")\n    df_catboost_feature_selection')


# ### Training CatBoost Classifier

# In[ ]:


def save_catboost_model(catboost_model, catboost_feature_selection):
    param_dict = {
                        'learning_rate' : 'lr',
                        'depth' : 'depth',
                        'l2_leaf_reg' : 'l2',
                        'random_strength' : 'rs',
                        'one_hot_max_size' : '1H',
                        'bagging_temperature' : 'bag_temp',
                        'min_data_in_leaf' : 'min_data',
                        'iterations' : 'iter',
                        'od_wait' : 'od_wait'
        }
    
    save_folder = data_dir
    save_folder += 'CatBoostClassifier'
    save_folder += '-' + str(np.round(catboost_model.get_best_score()['validation']['AUC'],6)) + '_score'

    for param, value in predefined_params.items():
        if param in param_dict.keys():
            if "." in str(value):
                save_folder += '-' + str(np.round(value, 4)) + '_' + param_dict[param]
            else:
                save_folder += '-' + str(value) + '_' + param_dict[param]

    # Create a folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    try:
        # Save model, feature_selection and results
        with open(save_folder + '/' + 'model.pkl', 'wb') as file:
            pickle.dump(catboost_model, file)

        with open(save_folder + '/' + 'feature_selection.pkl', 'wb') as file:
            pickle.dump(catboost_feature_selection, file)

        with open(save_folder + '/' + 'best_score.pkl', 'wb') as file:
            pickle.dump(catboost_model.get_best_score(), file)
    except CatBoostError:
        print("Issue saving model on: ", save_folder)
        shutil.rmtree(save_folder, ignore_errors=True)


# In[ ]:


def train_catboost(params, X, y, X_valid, y_valid, catboost_feature_selection, cat_features_index = None, save_models = True, task_type = "GPU", verbose = True, plot = True):
    
    catboost_model = cb.CatBoostClassifier(iterations = 50000,
                                              eval_metric = "AUC",
                                              cat_features = cat_features_index,
                                              scale_pos_weight = y.value_counts()[0] / y.value_counts()[1],
                                              task_type=task_type,
                                              metric_period = 100,
                                              od_pval = 0.00001,
                                              od_wait = 50)
    
    catboost_model.set_params(**params)
    catboost_model.fit(X[catboost_feature_selection], y, 
                       eval_set = (X_valid[catboost_feature_selection], y_valid),
                      use_best_model = True,
                      #early_stopping_rounds = True,
                      plot = plot,
                      verbose = verbose)

    print(f"Best score {catboost_model.get_best_score()} with params {params}")
    if save_models:
        save_catboost_model(catboost_model, catboost_feature_selection)
        
    return (catboost_model, catboost_feature_selection, catboost_model.get_best_score()['validation']['AUC'])


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlist_catboost_models = []\n\n# Kaggle have some restrictions on HDD space and we could have space issues if we save the models\nif LOCAL:\n    save_models = True\nelse:\n    save_models = False\n\nif not do_training and len(catBoost_models_dirs) > 0:\n    \n    for i in tqdm_notebook(range(n_ensemble)):\n        best_model_dir = catBoost_models_dirs[i]\n        if BayesSearchCV_dirs != []:\n            latest_BayesSearchCV_dir = max(BayesSearchCV_dirs, key=os.path.getctime)\n\n        print("Loading model and result_dict from folder: " + best_model_dir)\n\n        with open(best_model_dir + \'/\' + \'model.pkl\', \'rb\') as file:\n            catboost_model = pickle.load(file)\n\n        with open(best_model_dir + \'/\' + \'feature_selection.pkl\', \'rb\') as file:\n            catboost_feature_selection = pickle.load(file)    \n        \n        if os.path.exists(best_model_dir + \'/\' + \'best_score.pkl\'):\n            with open(best_model_dir + \'/\' + \'best_score.pkl\', \'rb\') as file:\n                catboost_best_score = pickle.load(file) \n            list_catboost_models.append((catboost_model, catboost_feature_selection, catboost_best_score[\'validation\'][\'AUC\']))\n        else:\n            start_index = best_model_dir.find(\'CatBoostClassifier\') + len(\'CatBoostClassifier\') + 1\n            score = float(best_model_dir[start_index : start_index + 7])\n            list_catboost_models.append((catboost_model, catboost_feature_selection, score))\n\n        # Stop looking for more models if there is not more\n        if len(catBoost_models_dirs) - 1 == i:\n            break\n    \n    if BayesSearchCV_dirs != []:\n        with open(latest_BayesSearchCV_dir + \'/\' + \'result_dict.pkl\', \'rb\') as file:\n            catboost_result_dict = pickle.load(file)\n            \n    print("Done")\n\nelif use_predifined_params:\n    cat_features_index = [catboost_feature_selection.index(col) for col in categorical_columns if col in catboost_feature_selection]\n\n    list_predefined_params = []\n\n    predefined_params = {\n                        \'learning_rate\' : 0.05,\n                        \'depth\' : 4,\n                        \'l2_leaf_reg\' : 5,\n                        \'random_strength\' : 1,\n                        \'one_hot_max_size\' : 2,\n                        #\'min_data_in_leaf\' : 5,\n                        \'bagging_temperature\' : 0.01\n        }\n    list_predefined_params.append(predefined_params.copy())\n    \n    predefined_params = {\n                        \'learning_rate\' : 0.05,\n                        \'depth\' : 5,\n                        \'l2_leaf_reg\' : 20,\n                        \'random_strength\' : 15,\n                        \'one_hot_max_size\' : 2,\n                        #\'min_data_in_leaf\' : 10,\n                        \'bagging_temperature\' : 0.01\n        }\n    list_predefined_params.append(predefined_params.copy())\n    \n    predefined_params = {\n                        \'learning_rate\' : 0.05,\n                        \'depth\' : 6,\n                        \'l2_leaf_reg\' : 40,\n                        \'random_strength\' : 15,\n                        \'one_hot_max_size\' : 2,\n                        #\'min_data_in_leaf\' : 20,\n                        \'bagging_temperature\' : 0.01\n        }\n    list_predefined_params.append(predefined_params.copy())\n    \n    predefined_params = {\n                        \'learning_rate\' : 0.05,\n                        \'depth\' : 7,\n                        \'l2_leaf_reg\' : 120,\n                        \'random_strength\' : 1,\n                        \'one_hot_max_size\' : 2,\n                        #\'min_data_in_leaf\' : 25,\n                        \'bagging_temperature\' : 0.01\n        }\n    list_predefined_params.append(predefined_params.copy())\n    \n    predefined_params = {\n                        \'learning_rate\' : 0.05,\n                        \'depth\' : 8,\n                        \'l2_leaf_reg\' : 200,\n                        \'random_strength\' : 1,\n                        \'one_hot_max_size\' : 25,\n                        #\'min_data_in_leaf\' : 50,\n                        \'bagging_temperature\' : 0.01\n                        \n        }\n    list_predefined_params.append(predefined_params.copy())\n\n\n    for params in tqdm_notebook(list_predefined_params):\n\n        list_catboost_models.append(\n            train_catboost(params, \n                       X, y, \n                       X_valid, y_valid, \n                       catboost_feature_selection,\n                       cat_features_index,\n                       save_models = save_models,\n                       task_type = task_type)\n        )\nelse:\n    \n    cat_features_index = [catboost_feature_selection.index(col) for col in categorical_columns if col in catboost_feature_selection]\n\n    search_spaces = {\n                    \'learning_rate\' : (0.01, 0.5, \'log-uniform\'),\n                    \'depth\' : (3,16),\n                    \'l2_leaf_reg\' : (20,150),\n                    \'random_strength\' : (1,20),\n                    \'one_hot_max_size\' : (2,25),\n                    \'bagging_temperature\' : (0.0, 1.0)\n    }\n    \n    bayes_search = FixedBayesSearchCV(\n                                estimator = cb.CatBoostClassifier(iterations = 300,\n                                                                  eval_metric = "AUC",\n                                                                  cat_features = cat_features_index,\n                                                                  scale_pos_weight = y.value_counts()[0] / y.value_counts()[1],\n                                                                  task_type="GPU",\n                                                                  metric_period = 40),\n                                search_spaces = search_spaces,\n                                scoring = \'roc_auc\',\n                                cv = KFold(n_splits=3),\n                                return_train_score = True,\n                                n_jobs = 1,\n                                n_iter = 50,   \n                                verbose = 1,\n                                refit = False)\n\n    %time bayes_search.fit(X[catboost_feature_selection], y)\n    \n    catboost_result_dict = bayes_search.cv_results_\n    print(f"Best score {bayes_search.best_score_} with params {bayes_search.best_params_}")\n    \n    search_spaces_folder = data_dir\n    search_spaces_folder += \'BayesSearchCV\'\n    for key in search_spaces.keys():\n        search_spaces_str += \'_\' + key\n        search_spaces_str += \'(\' + str(search_spaces[key][0]) + \'-\' + str(search_spaces[key][1]) + \')\'\n    \n    # Create a folder if it does not exist\n    if not os.path.exists(search_spaces_folder):\n        os.makedirs(search_spaces_folder)\n    \n    with open(search_spaces_folder + \'/\' + \'result_dict.pkl\', \'wb\') as file:\n        pickle.dump(catboost_result_dict, file)\n    \n    list_best_params = np.array(pd.DataFrame(catboost_result_dict).nlargest(n_ensemble, \'mean_test_score\')[\'params\'])\n\n    for i in tqdm_notebook(range(n_ensemble)):\n\n        list_catboost_models.append(\n            train_catboost(list_best_params[i], \n               X, y, \n               X_valid, y_valid, \n               catboost_feature_selection,\n               cat_features_index,\n               save_models = save_models,\n               task_type = task_type)\n        )')


# ### Plots of parameters effects on mean_test_score (AUC)

# In[ ]:


if not use_predifined_params:
    result_pd = pd.DataFrame(catboost_result_dict)
    plot_x_columns = ['bagging_temperature', 'depth', 'l2_leaf_reg','learning_rate','one_hot_max_size','random_strength']
    list_dict_scatters = []
    for param in plot_x_columns:
        dict_param = {}
        dict_param['x_column'] = 'param_' + param
        dict_param['y_column'] = 'mean_test_score'
        dict_param['title'] = param

        list_dict_scatters.append(dict_param)

    plot_list_scatters(result_pd, list_dict_scatters, subplot_cols = 3, subplot_rows = 2)


# ### Ensemble of CatBoost Classifier and Deep Neuronal Network

# In[ ]:


def ensemble_catboosts(X, list_models):
    sum_scores = 0
    
    y_ensemble = np.zeros(shape=len(X))
    
    for model,feature_selection,score in tqdm_notebook(list_models):
        y_pred_cat = model.predict_proba(X[feature_selection])[:,1] * score / len(list_models)
        sum_scores += score
        
        y_ensemble = np.add(y_ensemble, y_pred_cat)
        
    return y_ensemble / sum_scores


# ### Ploting ROC AUC

# In[ ]:


print("Plot Train Ensemble ROC AUC")
plot_roc_auc(y, ensemble_catboosts(X[catboost_feature_selection], list_catboost_models))

print("Plot Valid Ensemble ROC AUC")
plot_roc_auc(y_valid, ensemble_catboosts(X_valid[catboost_feature_selection], list_catboost_models))

print("Plot Test Ensemble ROC AUC")
plot_roc_auc(y_test, ensemble_catboosts(X_test[catboost_feature_selection], list_catboost_models))


# ## Predict Submission

# In[ ]:


submission = load_submission(data_dir, False)
y_submission = ensemble_catboosts(submission, list_catboost_models)

submission_pd = pd.DataFrame()
submission_pd['TransactionID'] = submission['TransactionID']
submission_pd['isFraud'] = y_submission
submission_pd.set_index('TransactionID', inplace = True)

submission_pd.to_csv('submission' + str(datetime.date.today()) + '.csv')

