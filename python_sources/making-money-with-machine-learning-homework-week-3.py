#!/usr/bin/env python
# coding: utf-8

# I began by following the solution outlined here: https://youtu.be/e9XzWiy-Lgk for the classification. I used this kernel: https://www.kaggle.com/cyberia/heatmaps-contingency-tabs as guidance for visualization.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Image, display
from tqdm import tqdm_notebook as tqdm

import seaborn as sns
import warnings
warnings.simplefilter("ignore")
plt.style.use('ggplot')

import sklearn


# In[ ]:


def create_col_name(base_str, start_int, end_int):
    return [base_str + str(i) for i in range(start_int, end_int+1)]


# In[ ]:


create_col_name('card', 1, 6)


# In[ ]:


cat_cols = ((['ProductCD']) + create_col_name('card', 1, 6) + ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + create_col_name('M', 1, 9) + ['DeviceType', 'DeviceInfo'] + create_col_name('id_', 12, 38))

id_cols = ['TransactionID', 'TransactionDT']

target = 'isFraud'


# In[ ]:


type_map = {c: str for c in cat_cols + id_cols}


# In[ ]:


df_train_id = {c: str for c in cat_cols + id_cols}


# # Loading the training data

# In[ ]:


df_train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', dtype=type_map)
df_train_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', dtype=type_map)


# In[ ]:


df_train_id.shape, df_train_trans.shape


# In[ ]:


df_train_id.head()


# We merge the datasets into one, using TransactionID as the private key. Using a left join because we want every transaction included.

# In[ ]:


df_train = df_train_trans.merge(df_train_id, on='TransactionID', how='left')


# We will extract all of the names for the numeric columns.

# In[ ]:


numeric_cols = [col for col in df_train.columns.tolist() if col not in cat_cols + id_cols + [target]]


# Sanity check that the resulting dataframe has the same shape (i.e. the same number of rows).

# In[ ]:


assert(df_train.shape[0]==df_train_trans.shape[0])


# In[ ]:


df_train.head()


# # Loading the test data

# In[ ]:


df_test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', dtype=type_map)
df_test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', dtype=type_map)


# Repeat the process of transforming the data.

# In[ ]:


df_test = df_test_trans.merge(df_test_id, on='TransactionID', how='left')
df_test.head()


# # Visually exploring the dataset

# In[ ]:


sns.set(rc={'figure.figsize':(15,12)})
sns.heatmap(pd.crosstab(df_train.isFraud, df_train.ProductCD), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(df_train.isFraud, df_train.card4), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(df_train.P_emaildomain, df_train.isFraud), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(df_train.R_emaildomain, df_train.isFraud), annot = True, fmt = "d")


# # Modeling
# 
# We're using catboost, which can natively handle categorical columns. It is useful for when there are a lot of categorical features, as we have here. Also, we cannot easily determine which features are most important in this dataset, so it would be difficult for us to reduce the number of features we need our model to take into account.

# In[ ]:


from catboost import Pool, CatBoostClassifier, cv


# In[ ]:


#combine categorical and numeric columns
features = cat_cols + numeric_cols


# In[ ]:


df_train.loc[:,cat_cols] = df_train[cat_cols].fillna('<UNK>')


# In[ ]:


df_test.loc[:,cat_cols] = df_test[cat_cols].fillna('<UNK>')


# catboost requires that we inform it as to which columns are categorical, which we previously determined.

# In[ ]:


train_data = Pool(
    data = df_train[features],
    label = df_train[target],
    cat_features = cat_cols,
)


# In[ ]:


test_data = Pool(
    data=df_test[features],
    cat_features = cat_cols,
)


# We'll provide some initial parameters, which we may tweak later.

# In[ ]:


params = {
    'iterations' : 50,
    #learning_rate = 0.1
    'custom_metric': 'AUC',
    'loss_function': 'CrossEntropy'
}


# In[ ]:


cv_results = cv(train_data, params, fold_count = 3, plot = True, verbose = False)


# In[ ]:


model = CatBoostClassifier(**params)
model.fit(train_data, plot=True, verbose=False)


# In[ ]:


y_test_hat = model.predict_proba(test_data)[:,1]


# In[ ]:


df_test['isFraud'] = y_test_hat


# In[ ]:


df_test[['TransactionID', 'isFraud']].to_csv('../out/submission_v1.csv', index = False)

