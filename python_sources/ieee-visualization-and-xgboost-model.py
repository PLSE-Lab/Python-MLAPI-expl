#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.
# 
# The data is broken into two files identity and transaction, which are joined by TransactionID.
# 
# > Note: Not all transactions have corresponding identity information.
# 
# **Categorical Features - Transaction**
# 
# - ProductCD
# - emaildomain
# - card1 - card6
# - addr1, addr2
# - P_emaildomain
# - R_emaildomain
# - M1 - M9
# 
# **Categorical Features - Identity**
# 
# - DeviceType
# - DeviceInfo
# - id_12 - id_38
# 
# The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_profiling as pdp
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth', 5000)
pd.options.display.float_format = '{:.3f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

import os
print(os.listdir("../input/"))
DIR_NAME = "../input"
# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

import xgboost as xgb


# # import data

# In[ ]:


# load csv
def load_dir_csv(directory, csv_files=None):
    if csv_files is None:
        csv_files = sorted( [ f for f in os.listdir(directory) if f.endswith(".csv") ])    
    csv_vars  = [ filename[:-4] for filename in csv_files ]
    gdict = globals()
    for filename, var in zip( csv_files, csv_vars ):
        print(f"{var:32s} = pd.read_csv({directory}/{filename})")
        gdict[var] = pd.read_csv( f"{directory}/{filename}" )
        print(f"{'shape ':32s} = " + str(gdict[var].shape))
        display(gdict[var].head())


# In[ ]:


get_ipython().run_cell_magic('time', '', 'load_dir_csv(DIR_NAME, ["train_transaction.csv", "test_transaction.csv", "train_identity.csv", "test_identity.csv"])')


# There is data of column name that I don't understand well.  
# For example, CXX, DXX, MXX, VXX of `transaction.csv`.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# merge to data\ntrain = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')\ntest = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')")


# In[ ]:


print('Train dataset has {} rows and {} columns.'.format(train.shape[0], train.shape[1]))
print('Test dataset has {} rows and {} columns.'.format(test.shape[0], test.shape[1]))


# In[ ]:


del train_transaction, train_identity, test_transaction, test_identity


# In[ ]:


gc.collect()


# In[ ]:


train.head()


# In[ ]:


test.head()


# # EDA

# ## Basic statistics

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# The VXX data seems to have almost zero data.

# ## Missing value

# In[ ]:


def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

def missing_values_table_specified_value(df, value=0.5): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    if is_integer_num(value):
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns['Missing Values'] >= value]
        print('The number of columns with {} counts missing values is {}.'.format(value, len(mis_val_table_ren_columns)))
    else:
        value = value * 100
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns['% of Total Values'] >= value]
        print('The number of columns with {}% missing values is {}.'.format(value, len(mis_val_table_ren_columns)))
    return mis_val_table_ren_columns 

def missing_values_table(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[ ]:


missing_values_table_specified_value(train, 0.5).head()


# In[ ]:


missing_values_table_specified_value(test, 0.5).head()


# Test data contains fewer columns with 50% or more missing data than train data.

# In[ ]:


display(missing_values_table(train), missing_values_table(test))


# ## Looking Fraud

# In[ ]:


train['isFraud'].value_counts()


# In[ ]:


sns.countplot(train['isFraud'])


# It is imbalanced data, which has an overwhelming number of 0

# In[ ]:


print('{:.4f}% of data that are fraud in train.'.format(train['isFraud'].mean() * 100))


# ## Split into Quantitative and Qualitative data

# In[ ]:


quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
print(quantitative)
print('Counts: {}'.format(len(quantitative)))


# In[ ]:


qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
print(qualitative)
print('Counts: {}'.format(len(qualitative)))


# ## Looking qualitative columns

# In[ ]:


# Get columns with less than n unique values in the type specified in column_type
def unique_data_under_n_columns_list(df, column_type='object', n=0):
    # If n = 0, return all columns
    if n == 0:
        n = df.shape[0]
    
    columns_list = df.select_dtypes(include=column_type).columns.tolist()
    unique_n_list = []
    for colomn in columns_list:
        unique_count = len(df[colomn].unique())
        if unique_count < n:
            unique_n_list.append(colomn)
        else:
            print('{} Is excluded because it has a unique value {} greater than {}.'.format(colomn, n, unique_count))
    return unique_n_list


# In[ ]:


# Graph data of specified type
def visualization_object_data(df, fig_x=10, fig_y=5):
    # Get columns type of object
    object_list = unique_data_under_n_columns_list(df, 'object', 65)
    for colomn in object_list:
        print('column name:{} / unique value(axis x):{} / max value:{} / min value:{} / missing value:{} '.format(colomn, len(df[colomn].unique()), max(df[colomn].value_counts()), min(df[colomn].value_counts()), df[colomn].isnull().sum()))
        # Calculate the Optimal Horizontal Size of Shapes
        fig_x = len(df[colomn].unique())
        order = df[colomn].value_counts(ascending=False).index
        if fig_x <= 8:
            fig, ax = plt.subplots(1, 3, figsize=(fig_x*6,fig_y))
            sns.countplot(x=colomn, ax=ax[0], data=df, order=order)
            ax[0].set_title('All', fontsize=14)
            sns.countplot(x=colomn, ax=ax[1], data=df.loc[df['isFraud'] == 1], order=order)
            ax[1].set_title('isFraud = 1', fontsize=14)
            sns.countplot(x=colomn, ax=ax[2], data=df.loc[df['isFraud'] == 0], order=order)
            ax[2].set_title('isFraud = 0', fontsize=14)
        else:
            fig, ax = plt.subplots(1, 3, figsize=(32,10))
            sns.countplot(y=colomn, ax=ax[0], data=df, order=order)
            ax[0].set_title('All', fontsize=14)
            sns.countplot(y=colomn, ax=ax[1], data=df.loc[df['isFraud'] == 1], order=order)
            ax[1].set_title('isFraud = 1', fontsize=14)
            sns.countplot(y=colomn, ax=ax[2], data=df.loc[df['isFraud'] == 0], order=order)
            ax[2].set_title('isFraud = 0', fontsize=14)

        plt.show()
        plt.pause(0.05)


# In[ ]:


visualization_object_data(train)


# # Boosting Model + FE Importance

# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train = train.drop('isFraud', axis=1)\ny_train = train['isFraud'].copy()\nX_test = test.copy()")


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


del train, test
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Label Encoding\nfor f in qualitative:\n    lbl = preprocessing.LabelEncoder()\n    lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n    X_train[f] = lbl.transform(list(X_train[f].values))\n    X_test[f] = lbl.transform(list(X_test[f].values))  ')


# In[ ]:


# Check if it is encoded
print(len(X_train.select_dtypes(include='object').columns))
print(len(X_test.select_dtypes(include='object').columns))


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = reduce_mem_usage(X_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_test = reduce_mem_usage(X_test)')


# In[ ]:


gc.collect()


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


X_train.describe()


# In[ ]:


X_test.describe()


# # model xgboost

# In[ ]:


sub = pd.read_csv(DIR_NAME + '/sample_submission.csv')


# In[ ]:


sub.head()


# In[ ]:


NFOLDS = 5
kf = KFold(n_splits = NFOLDS, shuffle = True)
y_preds = np.zeros(sub.shape[0])
y_oof = np.zeros(X_train.shape[0])


# In[ ]:


get_ipython().run_cell_magic('time', '', "for tr_idx, val_idx in kf.split(X_train, y_train):\n\n    clf = xgb.XGBClassifier(\n        n_estimators=500,\n        max_depth=9,\n        learning_rate=0.05,\n        subsample=0.9,\n        colsample_bytree=0.9,\n        tree_method='gpu_hist'\n    )\n    \n    X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]\n    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    y_oof[val_idx] = y_pred_train\n    \n    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))\n    \n    y_preds += clf.predict_proba(X_test)[:,1] / NFOLDS")


# In[ ]:


sub['isFraud'] = y_preds
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()

