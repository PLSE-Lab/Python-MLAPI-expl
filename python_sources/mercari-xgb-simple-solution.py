#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from scipy import sparse as ssp
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


import lightgbm as lgb
import xgboost as xgb


import time
import re
import collections
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# **Reading data**

# In[ ]:


df_train = pd.read_csv('../input/train.tsv', sep='\t')

Y_train=np.log1p(df_train.price);
Id_train=df_train.train_id
X_train=df_train[['name','item_condition_id','category_name','brand_name','shipping','item_description']];

df_test = pd.read_csv('../input/test.tsv', sep='\t')

X_test=df_test[['name','item_condition_id','category_name','brand_name','shipping','item_description']];
del df_train


# Basic visualization

# In[ ]:


X_train.info()
X_test.info()


# **Features extraction from basic data**

# In[ ]:


def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    


# In[ ]:


def convert_features(X):
    X['category_name'].fillna('other  / other / other', inplace=True)
    X['item_description'].fillna('No description yet', inplace=True)
    X['general_cat'], X['subcat_1'], X['subcat_2'] =     zip(*X['category_name'].apply(lambda x: split_cat(x)))
    del X['category_name']
    X.brand_name.replace(np.nan,value='Noname',inplace=True)
    Y=X
    return Y

X_train_converted=convert_features(X_train.copy())
X_test_converted=convert_features(X_test.copy())

X_train_converted.head()

## One hot encoding
ready_features=["item_condition_id","shipping"]
cat_features=['brand_name','general_cat','subcat_1','subcat_2']
train_len=len(X_train_converted)
all_data_categorials=pd.concat([X_train_converted[cat_features],X_test_converted[cat_features]],axis=0)

## Categorials to integer to reduce memory usage
for column in cat_features:
    all_data_categorials[column]  = pd.Categorical(all_data_categorials[column]).codes

all_data_dummies=all_data_categorials;
del all_data_categorials;


#for column in cat_features:
#    temp=pd.get_dummies(pd.Series(all_data_dummies[column]),sparse=True)
#    all_data_dummies=pd.concat([all_data_dummies,temp],axis=1)
#    all_data_dummies=all_data_dummies.drop([column],axis=1)
    



# **Gathering all features together**

# In[ ]:



X_train_dummies = all_data_dummies[:train_len]

X_test_dummies =  all_data_dummies[train_len:]

X_train_processed=result = pd.concat([X_train_converted[ready_features], X_train_dummies], axis=1)
X_test_processed=pd.concat([X_test_converted[ready_features],X_test_dummies], axis=1)
 
X_train_processed.info()
X_test_processed.info()

X_train_processed.head()


# **XGB training on selected features**

# train / validation split

# In[ ]:


train_X, valid_X, train_y, valid_y = train_test_split(X_train_processed, Y_train, test_size = 0.15, random_state = 201) 
print('data split done with success')


# xgb training

# In[ ]:


data_train = xgb.DMatrix(train_X, label=train_y)
data_valid = xgb.DMatrix(valid_X, label=valid_y)

 
watchlist = [(data_train, 'train'), (data_valid, 'valid')]

xgb_params = {'min_child_weight': 20,
              'eta': 0.013,
              'colsample_bytree': 0.45,
              'max_depth': 16,
            'subsample': 0.88,
              'lambda': 2.07,
              'nthread': 4,
              'booster' :
              'gbtree',
              'silent': 1,
            'eval_metric': 'rmse',
              'objective': 'reg:linear'}

model_xgb = xgb.train(xgb_params, data_train, 2000, watchlist, early_stopping_rounds=20, verbose_eval=10)


# Even without any usage of description the score is not so bad. To be improved...

# **XGB test generation **

# In[ ]:


data_test = xgb.DMatrix(X_test_processed)
test_predicitons = model_xgb.predict(data_test)


# **XGB test submission**

# In[ ]:


df_test["price"] = np.expm1(test_predicitons)
df_test[["test_id", "price"]].to_csv("submission_XGB.csv", index = False)


# **Training stage**
