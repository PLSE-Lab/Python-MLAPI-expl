#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA


# # Data description
# Threads
# 1. https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203

# # Config

# In[ ]:


TARGET = 'isFraud'
PREPROCESS_BLACKLIST = {TARGET, 'TransactionID', 'TransactionDT'}
FORCE_KEEP = {'dist1'}
GROUP_CARDINALITY_MAX = 5
CATEGORICAL_FEATURES = {'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                       'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23',
       'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
       'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37',
       'id_38', 'DeviceType', 'DeviceInfo'}

PCA_PREFIX = '_pc_'
PCA_N_COMPONENTS = 19
PCA_FEATURES = set([f'{PCA_PREFIX}{i}' for i in range(PCA_N_COMPONENTS)])
pca_input = set()
for i in range(1, 340):
    pca_input.add(f'V{i}')


# In[ ]:


# Characters such as empty strings '' or numpy.inf are considered NA values
pd.set_option('use_inf_as_na', True)
pd.set_option('display.max_columns', 500)


# In[ ]:


def is_categorical(df, col):
    return col in CATEGORICAL_FEATURES or pd.api.types.is_string_dtype(df[col])


# In[ ]:


get_ipython().run_cell_magic('time', '', "folder_path = '../input/ieee-fraud-detection'\ntrain_identity = pd.read_csv(f'{folder_path}/train_identity.csv')\ntrain_transaction = pd.read_csv(f'{folder_path}/train_transaction.csv')\ntest_identity = pd.read_csv(f'{folder_path}/test_identity.csv')\ntest_transaction = pd.read_csv(f'{folder_path}/test_transaction.csv')\n\n# let's combine the data and work with the whole dataset\ntrain = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')\ntest = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')\ndel train_identity, train_transaction, test_identity, test_transaction")


# # Set data type for categorical variables
# * Use string instead of `category` dtype. This allows string manipulations later. 
# * Imput missing values with UNKNOWN to allow group-by aggregations later.

# In[ ]:


get_ipython().run_cell_magic('time', '', "def handle_categorical_variables(df, columns):\n    df[columns] = df[columns].astype(str)\n    #df[columns] = df[columns].fillna(imput_value)\n    #df[columns] = df[columns].replace('nan', imput)\n    return df\n\n\ncols = list(CATEGORICAL_FEATURES)\ntrain = handle_categorical_variables(train, cols)\ntest = handle_categorical_variables(test, cols)")


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.columns.values


# In[ ]:


train.head()


# # Drop columns that have only one value

# In[ ]:


one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
one_value_cols == one_value_cols_test


# # Purchaser address
# Concatenate "addr1" and "addr2" columns.

# In[ ]:


def combine_address_columns(df):
    cols = ['addr1', 'addr2']
    df['addr'] = df[cols].apply(','.join, axis=1)
    df = df.drop(columns=cols)
    return df


train = combine_address_columns(train)
test = combine_address_columns(test)


# # Handle missing data

# In[ ]:


def _missing_data(df):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data['Total'] > 0]


missing = _missing_data(train)
missing.head()


# In[ ]:


tmp = missing.filter(items=['dist1'], axis=0)
tmp.head()


# In[ ]:


cols = missing[missing['Percent'] > 0.5].index.values
cols = [col for col in cols if col not in FORCE_KEEP]
print(f'drop {len(cols)} columns={cols}')


# In[ ]:


train = train.drop(columns=cols)
test = test.drop(columns=cols)
print(f'train={train.shape}, test={test.shape}')


# In[ ]:


def imput_mode(col, train, test):
    imput = train[col].mode()[0]
    train[col].fillna(imput, inplace = True)
    test[col].fillna(imput, inplace = True)
    return imput


def imput_median(col, train, test):
    imput = train[col].median()
    train[col].fillna(imput, inplace = True)
    test[col].fillna(imput, inplace = True)
    return imput
    

cols = [col for col in train if col not in PREPROCESS_BLACKLIST and not is_categorical(train, col)]
for i, col in enumerate(cols):
    imput = imput_median(col=col, train=train, test=test)
    print(f'{i + 1}. "{col}" imput median={imput}')


# In[ ]:


# sanity check to ensure no missing data
missing = _missing_data(train)
print(f'[{len(missing) == 0}] train has no missing data')


# In[ ]:


missing = _missing_data(test)
print(f'[{len(missing) == 0}] test has no missing data')


# # Imbalanced class problem

# In[ ]:


train[TARGET].value_counts()


# # Test set occurs after train set on the timeline
# Use hold-out validation instead of k-fold. Temporal split.

# In[ ]:


train['TransactionDT'].describe().apply(lambda x: format(x, 'f'))


# In[ ]:


test['TransactionDT'].describe().apply(lambda x: format(x, 'f'))


# # Create ratios to group statistics

# In[ ]:


CATEGORICAL_FEATURES.add('addr')
CATEGORICAL_FEATURES.remove('addr1')
CATEGORICAL_FEATURES.remove('addr2')
cmap = {}
for cf in CATEGORICAL_FEATURES:
    cmap[cf] = len(train[cf].unique())

print(f'cmap={cmap}')


# In[ ]:


groups = []
for k, v in cmap.items():
    if v <= GROUP_CARDINALITY_MAX:  # max theshold for number of distinct values
        groups.append([k])
        
print(f'{len(groups)} groups={groups}')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def ratio_to_group(train, test, target_columns, group_columns, statistic):\n    for t in target_columns:\n        d = train.groupby(group_columns)[t].transform(statistic)\n        col = f\'{t}_to_{"_".join(group_columns)}_{statistic}\'\n        train[col] = train[t] / d\n        test[col] = test[t] / d\n        train[col] = train[col].fillna(0)\n        test[col] = test[col].fillna(0)\n    return train, test\n\ncols = [\'TransactionAmt\', \'dist1\']\nstatistics = [\'mean\', \'std\']\nfor g in groups:\n    for s in statistics:\n        train, test = ratio_to_group(train, test, target_columns=cols, group_columns=g, statistic=s)\n\n\ncols = train.columns.values\nprint(f\'{len(cols)} columns={cols}\')')


# # Handle categorical features
# Frequency encoding.

# In[ ]:


def encode(df, col, encoder):
    df[col] = df[col].map(encoder).fillna(0)
    assert df[col].isnull().sum() == 0

def freq_encode(col):
    encoder = dict(train[col].value_counts(normalize=True))
    encode(train, col, encoder)
    encode(test, col, encoder)


cols = [col for col in train if col not in PREPROCESS_BLACKLIST and is_categorical(train, col)]
fe_cols = set(cols)
print(f'Frequency encode {len(cols)} columns={cols}')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for col in cols:\n    freq_encode(col)\n\ntrain.head()')


# # Zero-mean, unit-variance normalization

# In[ ]:


get_ipython().run_cell_magic('time', '', "cols = set(train.columns.values) - PREPROCESS_BLACKLIST - fe_cols - PCA_FEATURES\ncols = list(cols)\nprint(f'transform {len(cols)} columns={cols}')\npt = PowerTransformer()\npt.fit(train[cols]) \ntrain[cols] = pt.transform(train[cols])\ntest[cols] = pt.transform(test[cols])\n# imput zero for any numerical errors\ntrain = train.fillna(0)\ntest = test.fillna(0)\ntrain.head()")


# # Dimensionality reduction: PCA
# Do standardization before PCA

# In[ ]:


get_ipython().run_cell_magic('time', '', "def _pca_features(dfs, cols, n_components, prefix):\n    pca = PCA(n_components=n_components)\n    pca.fit(dfs[0][cols])\n    res = []\n    for df in dfs:\n        pcs = pd.DataFrame(pca.transform(df[cols]))\n        pcs.rename(columns=lambda x: str(prefix)+str(x), inplace=True)\n        df = pd.concat([df, pcs], axis=1)\n        #df.drop(columns=cols, inplace=True)\n        res.append(df)\n    return res\n\n\ncols = pca_input & set(train.columns.values)\ncols = list(cols)\nprint(f'PCA {len(cols)} columns={cols}')\ndfs = _pca_features(dfs=[train, test], cols=cols, n_components=PCA_N_COMPONENTS, prefix=PCA_PREFIX)\ntrain, test = dfs[0], dfs[1]\ntrain.head()")


# In[ ]:


test.head()


# # Sort train by time to allow Temporal split later

# In[ ]:


train = train.sort_values(by=['TransactionDT'])
#val_size = int(0.01 * len(train))
#val = train[-val_size:]
#val['isFraud'].value_counts()
#del val


# # Save outputs
# Drop unused columns. Cast to smallest data types possible.

# In[ ]:


cols = ['TransactionID', 'TransactionDT']
train = train.drop(columns=cols)
test = test.drop(columns=cols)
cols = train.columns.values
print(f'{len(cols)} columns={cols}')


# In[ ]:


cdt_map = {TARGET: 'uint8'}
for col in test.columns:
    cdt_map[col] = 'float32'

train = train.astype(cdt_map)
del cdt_map[TARGET]
test = test.astype(cdt_map)
train.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
print(os.listdir("."))

