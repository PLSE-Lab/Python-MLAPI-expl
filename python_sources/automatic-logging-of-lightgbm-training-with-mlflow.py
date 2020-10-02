#!/usr/bin/env python
# coding: utf-8

# # Install MLflow

# In[ ]:


get_ipython().system('pip install --quiet mlflow')


# # Import libraries

# In[ ]:


import warnings

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import mlflow
import mlflow.lightgbm

warnings.filterwarnings("ignore")


# # Prepare training and validation data

# In[ ]:


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val)


# # Train a model

# In[ ]:


bst_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'seed': 42,
}

train_params = {
    'num_boost_round': 30,
    'verbose_eval': 5,
    'early_stopping_rounds': 5,
}


mlflow.lightgbm.autolog()  # Enable auto logging.

with mlflow.start_run():
    model = lgb.train(
        bst_params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'valid'],
        **train_params,
    )
    
    # Do something with the trained model.
    # ...


# Unfortunately, it's impossible to display the logging result on a notebook. The GIF attached below shows how it looks like on the MLflow UI.

# ![demo](https://user-images.githubusercontent.com/17039389/77157368-8aae7c00-6ae4-11ea-8ac0-da0ce0c6ca79.gif)
