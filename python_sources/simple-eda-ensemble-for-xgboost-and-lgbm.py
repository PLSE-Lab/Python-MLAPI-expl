#!/usr/bin/env python
# coding: utf-8

# ## 0. Context

# - Loading Library
# - Read Data SET
# - EDA
# - Preprocessing
# * Feature Engineering
# - Modeling
# - Evaluation

# ## 1. Loading Library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from tqdm import tqdm_notebook
import random

import xgboost as xgb
#import lightgbm as lgb

from sklearn.decomposition import PCA, IncrementalPCA,SparsePCA, KernelPCA, FastICA, TruncatedSVD 

import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input/ieee-fraud-detection/"))

import gc


# * Install LightGBM GPU VERSION

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[ ]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[ ]:


import lightgbm as lgb
print("LightGBM version:", lgb.__version__)


# In[ ]:


get_ipython().system('nvidia-smi')


# ## 2. Reading Data SET

# ### 1) Reduce Memory using down sizing Data SET

# In[ ]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():  # case if row has NA value, then return False, also inverse returns True.
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train_tr = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')\ndf_train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train_tr = reduce_mem_usage(df_train_tr)\ndf_train_id = reduce_mem_usage(df_train_id)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test_tr = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')\ndf_test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_test_tr = reduce_mem_usage(df_test_tr)\ndf_test_id = reduce_mem_usage(df_test_id)')


# In[ ]:


submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = pd.merge(df_train_tr, df_train_id, on = 'TransactionID', how = 'left')\ndf_test = pd.merge(df_test_tr, df_test_id, on = 'TransactionID', how = 'left')")


# In[ ]:


del df_train_id, df_train_tr, df_test_id, df_test_tr
gc.collect()


# In[ ]:


df = pd.concat([df_train, df_test], axis=0, sort=False)
df.reset_index(inplace=True)


# In[ ]:


del df_train, df_test
gc.collect()


# ## 3. EDA

# - From above on, Usage of RAM is 3.8GB

# - In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.
# 
# - The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

# - In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.
# 
# - The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.
# ------------------------------
# - *Categorical Features - Transaction
# - ProductCD
# - card1 - card6
# - addr1, addr2
# - P_emaildomain
# - R_emaildomain
# - M1 - M9
# ------------------------------
# - *Categorical Features - Identity
# - DeviceType
# - DeviceInfo id_12 - id_38
# ------------------------------
# - The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).

# ### 1) Check Missing Data

# In[ ]:


def missing_data(df) :
    count = df.isnull().sum()
    percent = (df.isnull().sum()) / (df.isnull().count()) * 100
    total = pd.concat([count, percent], axis=1, keys = ['Count', 'Percent'])
    types = []
    for col in df.columns :
        dtypes = str(df[col].dtype)
        types.append(dtypes)
    total['dtypes'] = types
    
    return np.transpose(total)


# In[ ]:


missing_data(df)


# ### 2) Check Numeric Columns Properties

# In[ ]:


num_cols = [col for col in df.columns if df[col].dtype not in ['object']]
df[num_cols].describe()


# ### 3) Check Categorical Columns Properties

# In[ ]:


cat_cols = [col for col in df.columns if df[col].dtype in ['object']]
df[cat_cols].describe()


# In[ ]:


for col in cat_cols :
    
    print('-' * 50)
    print('# col : ', col)
    #print(df[df.index < 590540]['isFraud'].groupby(df[col]).sum(),df[df.index < 590540]['isFraud'].groupby(df[col]).count())
    print(100*df[df.index < 590540]['isFraud'].groupby(df[col]).sum()/
          df[df.index < 590540]['isFraud'].groupby(df[col]).count()) 


# In[ ]:


for col in cat_cols :
    uniq = np.unique(df[col].astype(str))
    print('-' * 100)
    print('# col {}, n_uniq {}, uniq {}'.format(col, len(uniq), uniq))


# ## 4. Feature Engineering

# ### 1) Correlation Analysis of Numeric Values

# In[ ]:


"""
 kernel died
"""
#%%time 
#cor = df[num_cols].astype(float).corr()
#cor = pd.DataFrame(cor)
#cor = cor['isFraud']
#cor = pd.DataFrame(cor)
#cor = cor[cor['isFraud'] > 0.2]
#cor_columns_over_zero_dot_tree = cor.index.tolist()


# In[ ]:


#del cor
#gc.collect()


# In[ ]:


#colormap = plt.cm.RdBu
#plt.figure(figsize = (20,20))
#plt.title('Correlation Analysis of Numeric Columns', y = 1.05, size = 15)
#sns.heatmap(df[cor_columns_over_zero_dot_tree].astype(float).corr(), linewidths = 0.1, vmax = 1.0, square = True,
#            cmap = colormap, linecolor = 'white', annot = True)


# In[ ]:


df['V39_V51_V52_cor'] = df['V39'] + df['V51'] + df['V52']
df['V40_V51_V52_cor'] = df['V40'] + df['V51'] + df['V52']
df['V44_V86_V87_cor'] = df['V44'] + df['V86'] + df['V87']
df['V45_V86_V87_cor'] = df['V45'] + df['V86'] + df['V87']
df['V86_V190_V199_V246_V257_cor'] = df['V86'] + df['V190'] + df['V199'] + df['V246'] + df['V257']
df['V87_V257_cor'] = df['V87'] + df['V257']
df['V148_V149_V154_V155_V156_V157_V158_cor'] = df['V148'] + df['V149'] + df['V154'] + df['V155'] + df['V156']+ df['V157'] + df['V158']
df['V170_V171_V188_V200_V201_cor'] = df['V170'] + df['V171'] + df['V188'] + df['V200'] + df['V201']
df['V171_V189_V200_V201_cor'] = df['V171'] + df['V189'] + df['V200'] + df['V201']
df['V188_V189_V200_V201_V242_V244_cor'] = df['V188'] + df['V189'] + df['V200'] + df['V201'] + df['V242'] + df['V244']
df['V189_V200_V201_V242_V233_V244_cor'] = df['V189'] + df['V200'] + df['V201'] + df['V242'] + df['V243'] + df['V244']
df['V190_V199_V228_V233_cor'] = df['V190'] + df['V199'] + df['V228'] + df['V233']
df['V199_V228_V230_V246_V257_V258_cor'] = df['V199'] + df['V228'] + df['V230'] + df['V246'] + df['V257'] + df['V258']
df['V200_V201_V244_cor'] = df['V200'] + df['V201'] + df['V244']
df['V201_V242_V244_cor'] = df['V201'] + df['V242'] + df['V244']
df['V228_V230_V246_V257_V258_cor'] = df['V228'] + df['V230'] + df['V246'] + df['V257'] + df['V258']
df['V230_V246_V257_V258_cor'] = df['V230'] + df['V246'] + df['V257'] + df['V258']
df['V242_V243_V244_cor'] = df['V242'] + df['V243'] + df['V244']
df['V243_V244_cor'] = df['V243'] + df['V244']
df['V246_V257_V258_cor'] = df['V246'] + df['V257'] + df['V258']


# ### 2) Feature Slicing in Time Series Data

# In[ ]:


START_DATE = '2019-01-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

df['TransactionDT_Weekdays'] = df['Date'].dt.dayofweek
df['TransactionDT_Days'] = df['Date'].dt.day
df['TransactionDT_Hours'] = df['Date'].dt.hour

df.drop(columns='Date', inplace=True)


# ### 3) Feature Values Filtering in Categorical Columns

# In[ ]:


def change_value_P_emaildomain(x) :
    if x in ['gmail.com', 'icloud.com', 'mail.com' , 'outlook.es', 'protonmail.com'] :
        return x
    else :
        return 'etc'
    
df.loc[:,'P_emaildomain'] = df['P_emaildomain'].apply(lambda x : change_value_P_emaildomain(x))


# In[ ]:


def change_value_R_emaildomain(x) :
    if x in ['gmail.com', 'icloud.com', 'mail.com', 'netzero.net', 'outlook.com', 'outlook.es', 'protonmail.com'] :
        return x
    else :
        return 'etc'
    
df.loc[:,'R_emaildomain'] = df['R_emaildomain'].apply(lambda x : change_value_R_emaildomain(x))


# In[ ]:


def change_value_id_30(x) :
    if x in ['Android 4.4.2', 'Android 5.1.1' 'iOS 11.0.1' 'iOS 11.1.0', 'iOS 11.4.0', 'other'] :
        return x
    else :
        return 'etc'

df.loc[:,'id_30'] = df['id_30'].apply(lambda x : change_value_id_30(x))


# In[ ]:


def change_value_id_31(x) :
    if x in ['Lanix/Ilium', 'Mozilla/Firefox' 'comodo' 'icedragon', 'opera', 'opera generic'] :
        return x
    else :
        return 'etc'

df.loc[:,'id_31'] = df['id_31'].apply(lambda x : change_value_id_31(x))


# In[ ]:


def change_value_id_33(x) :
    if x in ['1024x552', '1364x768' '1440x759' '1916x901', '1920x975', '2076x1080', '640x360'] :
        return x
    else :
        return 'etc'

df.loc[:,'id_33'] = df['id_33'].apply(lambda x : change_value_id_33(x))


# In[ ]:


tmp = 100*df[df.index < 590540]['isFraud'].groupby(df['DeviceInfo']).sum()/df[df.index < 590540]['isFraud'].groupby(df['DeviceInfo']).count()

Device_Info = []

for i in tqdm_notebook(range(len(tmp))) :
    if tmp[i] == 100.0 :
        Device_Info.append(tmp.index[i])

def change_value_DeviceInfo(x) :
    if x in Device_Info :
        return x
    else :
        return 'etc'

df.loc[:,'DeviceInfo'] = df['DeviceInfo'].apply(lambda x : change_value_DeviceInfo(x))


# ## 5. Modeling

# ### 1) PreProcessing

# In[ ]:


df_num = df.select_dtypes(exclude = ['object'])
df_cat = df.select_dtypes(include = ['object'])


# In[ ]:


pca_temp = df_num.drop(columns = ['isFraud', 'index', 'TransactionID', 'TransactionDT']).fillna(df_num.min()-1)
pca = PCA(n_components=5)
pca_X = pca.fit_transform(pca_temp)

for i in range(5) :
    df_num['PCA_' + str(i+1)] = pca_X[:,i]


# In[ ]:


del df, pca_temp
gc.collect()


# In[ ]:


df_cat_one_hot = pd.get_dummies(df_cat)


# In[ ]:


pca_temp_cat = df_cat_one_hot.fillna(df_cat_one_hot.min()-1)
pca_cat = PCA(n_components=5)
pca_X_cat = pca_cat.fit_transform(pca_temp_cat)

for i in range(5) :
    df_cat_one_hot['PCA_cat_' + str(i+1)] = pca_X_cat[:,i]


# In[ ]:


del pca_temp_cat
gc.collect()


# In[ ]:


df_total = pd.concat([df_num, df_cat_one_hot], axis=1)
df_total.shape


# In[ ]:


df_total.drop(columns = ['TransactionID', 'index'], inplace=True)


# In[ ]:


del df_num, df_cat
gc.collect()


# In[ ]:


df_train = df_total[df_total.index < 590540]
df_test = df_total[df_total.index >= 590540]


# In[ ]:


y = pd.DataFrame(df_train['isFraud'])
X = df_train.drop(columns=['isFraud'])
X_test = df_test.drop(columns=['isFraud'])


# In[ ]:


X.shape, y.shape, X_test.shape


# In[ ]:


del df_train, df_test, df_total
gc.collect()


# In[ ]:


np.unique(y['isFraud'])


# ### 2) Train / Validation Split

# In[ ]:


index_array = np.arange(len(X))
val_index = index_array[random.sample(range(0,X.shape[0]), X.shape[0]//5)]
train_index = np.delete(index_array[:X.shape[0]], val_index, axis=0)
len(train_index), len(val_index)


# In[ ]:


X_train, X_val = X.iloc[train_index], X.iloc[val_index]
y_train, y_val = y.iloc[train_index], y.iloc[val_index]


# ### 3) XGBoost Fitting

# In[ ]:


get_ipython().run_cell_magic('time', '', 'prediction_test_fold = []\n\nparam = {\'booster\' : \'gbtree\',\n         \'max_depth\' : 14,\n         \'nthread\' : -1,\n         \'num_class\' : 1,\n         \'objective\' : \'binary:logistic\',\n         \'silent\' : 1,\n         \'eval_metric\' : \'auc\',\n         \'eta\' : 0.01,\n         \'tree_method\' : \'gpu_hist\',\n         \'min_child_weight\' : 0,\n         \'colsample_bytree\' : 0.8,\n         \'colsample_bylevel\' : 0.8,\n         \'seed\' : 2019}\n\n\n\n    \nprint("Train Shape :", X_train.shape,\n      "Validation Shape :", X_val.shape,\n      "Test Shape :", X_test.shape)\n    \ndtrn = xgb.DMatrix(X_train, label=y_train, feature_names = X.columns)\ndval = xgb.DMatrix(X_val, label = y_val, feature_names = X.columns)\ndtst = xgb.DMatrix(X_test, feature_names = X.columns)\n    \nxgb1 = xgb.train(param, dtrn, num_boost_round=10000, evals = [(dtrn, \'train\'), (dval, \'eval\')],\n                 early_stopping_rounds = 200, verbose_eval=200)\n                 \nprediction_XGB = xgb1.predict(dtst)\n#prediction_test_fold.append(prediction_XGB)\n\nprediction_val_XGB = xgb1.predict(xgb.DMatrix(X_val, feature_names = X.columns))')


# In[ ]:


for i in range(2):
    
    if i == 0 :
        plt.figure(figsize = (25,6))
        plt.title('Validation Evaluation of XGBoost', y = 1.05, size = 15)
        plt.plot(y_val.reset_index()['isFraud'], '.', label = 'Real Validation Value', color = 'blue')
        plt.plot(prediction_val_XGB, '.', label = 'Predicted Validation Value', color = 'red')
        plt.legend()
        plt.show()
    
    else :
        plt.figure(figsize = (25,6))    
        plt.title('Error Evaluation of LightGBM', y = 1.05, size = 15)
        plt.plot(y_val.reset_index()['isFraud'] - prediction_val_XGB, '.', label = 'Error', color = 'green')
        plt.legend()
        plt.show()


# In[ ]:


del dtrn, dval, dtst, xgb1
gc.collect()


# #### 4) LightGBM Fitting

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nparams = {\'num_leaves\': 500,\n          \'min_child_weight\': 0.03,\n          \'feature_fraction\': 0.35,\n          \'bagging_fraction\': 0.35,\n          \'min_data_in_leaf\': 100,\n          \'objective\': \'binary\',\n          \'max_depth\': 14,\n          \'learning_rate\': 0.01,\n          "boosting_type": "gbdt",\n          "bagging_seed": 10,\n          "metric": \'auc\',\n          "verbosity": -1,\n          \'reg_alpha\': 0.2,\n          \'reg_lambda\': 0.6,\n          \'random_state\': 50,\n          \'device\': \'gpu\',\n          \'gpu_platform_id\': 0,\n          \'gpu_device_id\': 0\n         }\n\n\ndtrain = lgb.Dataset(X_train, label=y_train)\ndvalid = lgb.Dataset(X_val, label=y_val)\n\nmodel = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=100, early_stopping_rounds=200)\n    \n\nprediction_LGB = model.predict(X_test)\nprediction_val_LGB = model.predict(X_val)')


# In[ ]:


for i in range(2):
    
    if i == 0 :
        plt.figure(figsize = (25,6))
        plt.title('Validation Evaluation of XGBoost', y = 1.05, size = 15)
        plt.plot(y_val.reset_index()['isFraud'], '.', label = 'Real Validation Value', color = 'blue')
        plt.plot(prediction_val_LGB, '.', label = 'Predicted Validation Value', color = 'red')
        plt.legend()
        plt.show()
    
    else :
        plt.figure(figsize = (25,6))    
        plt.title('Error Evaluation of LightGBM', y = 1.05, size = 15)
        plt.plot(y_val.reset_index()['isFraud'] - prediction_val_LGB, '.', label = 'Error', color = 'green')
        plt.legend()
        plt.show()


# ### 5) Submission to Score Board

# In[ ]:


submission['isFraud'] = np.nan
submission.head()


# In[ ]:


submission['isFraud'] = (0.5 * prediction_XGB) + (0.5 * prediction_LGB)
#submission['isFraud'] = prediction_LGB
submission.head()


# In[ ]:


submission[submission['isFraud'] > 0.1]


# In[ ]:


submission.to_csv('sample_submission_after_Feature_Engineering11.csv', index = False)

