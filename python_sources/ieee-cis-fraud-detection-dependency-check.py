#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, timedelta
import gc
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import warnings
warnings.filterwarnings("ignore")

CAT_FCOLS = ['card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2']
C_FCOLS = [f'C{i}' for i in range(1, 15)]
D_FCOLS = [f'D{i}' for i in range(1, 16)]
V_FCOLS = [f'V{i}' for i in range(1, 340)] 
FLOAT64_TCOLS = CAT_FCOLS + C_FCOLS + D_FCOLS + V_FCOLS
FLOAT64_ICOLS = [f'id_0{i}' for i in range(1, 10)] + ['id_10', 'id_11', 'id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24', 'id_25', 'id_26', 'id_32']


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', dtype=dict.fromkeys(FLOAT64_ICOLS, np.float32))\ndf_test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', dtype=dict.fromkeys(FLOAT64_ICOLS, np.float32))\ndf_train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', dtype=dict.fromkeys(FLOAT64_TCOLS, np.float32))\ndf_test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', dtype=dict.fromkeys(FLOAT64_TCOLS, np.float32))\nX_train = pd.merge(df_train_transaction, df_train_identity, how='left', on='TransactionID')\nX_test = pd.merge(df_test_transaction, df_test_identity, how='left', on='TransactionID')\n\nprint('Number of Training Examples = {}'.format(df_train_transaction.shape[0]))\nprint('Number of Test Examples = {}\\n'.format(df_test_transaction.shape[0]))\nprint('Number of Training Examples with Identity = {}'.format(df_train_identity.shape[0]))\nprint('Number of Test Examples with Identity = {}\\n'.format(df_test_identity.shape[0]))\nprint('Training X Shape = {}'.format(X_train.shape))\nprint('Training y Shape = {}'.format(X_train['isFraud'].shape))\nprint('Test X Shape = {}\\n'.format(X_test.shape))\nprint('Training Set Memory Usage = {:.2f} MB'.format(X_train.memory_usage().sum() / 1024**2))\nprint('Test Set Memory Usage = {:.2f} MB\\n'.format(X_test.memory_usage().sum() / 1024**2))\n\ndel df_train_identity, df_test_identity, df_train_transaction, df_test_transaction\ngc.collect()")


# Grazder's [Filling card NaNs](https://www.kaggle.com/grazder/filling-card-nans) kernel inspired me to create this helper function. It basically checks the value counts of two given variables and outputs how many different values can dependent_var get for every independent variable value. This is one way to understand causality between two vectors which can't be seen by pearson correlation.
# 
# This function can be used to reveal connection between features and for imputation. There isn't any standard threshold for deciding dependent/not dependent, so if you have a hunch just use that information. There are some examples below.

# In[ ]:


def check_dependency(independent_var, dependent_var):
    
    independent_uniques = []
    temp_df = pd.concat([X_train[[independent_var, dependent_var]], X_test[[independent_var, dependent_var]]])
    
    for value in temp_df[independent_var].unique():
        independent_uniques.append(temp_df[temp_df[independent_var] == value][dependent_var].value_counts().shape[0])

    values = pd.Series(data=independent_uniques, index=temp_df[independent_var].unique())
    
    N = len(values)
    N_dependent = len(values[values == 1])
    N_notdependent = len(values[values > 1])
    N_null = len(values[values == 0])
        
    print(f'In {independent_var}, there are {N} unique values')
    print(f'{N_dependent}/{N} have one unique {dependent_var} value')
    print(f'{N_notdependent}/{N} have more than one unique {dependent_var} values')
    print(f'{N_null}/{N} have only missing {dependent_var} values\n')


# In[ ]:


check_dependency('card1', 'card2')


# In[ ]:


check_dependency('card1', 'card3')


# In[ ]:


check_dependency('card1', 'card4')


# In[ ]:


check_dependency('card1', 'card5')


# In[ ]:


check_dependency('card1', 'card6')


# In[ ]:


check_dependency('card1', 'addr2')


# In[ ]:


check_dependency('card1', 'P_emaildomain')


# In[ ]:


check_dependency('card1', 'R_emaildomain')


# In[ ]:


check_dependency('P_emaildomain', 'R_emaildomain')


# In[ ]:


check_dependency('addr1', 'P_emaildomain')


# In[ ]:


check_dependency('dist1', 'C3')


# How to use this function?
# * Found connection between **R_emaildomain** and **C5**
# * Checking what are the values can C5 take
# * Filling the NaNs

# In[ ]:


check_dependency('R_emaildomain', 'C5')


# In[ ]:


X_test[~X_test['R_emaildomain'].isnull()]['C5'].value_counts()


# In[ ]:


X_test = X_test['C5'].fillna(0)

