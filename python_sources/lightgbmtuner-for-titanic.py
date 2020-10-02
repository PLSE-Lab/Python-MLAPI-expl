#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install optuna==0.19.0')


# In[ ]:


import optuna 
optuna.__version__


# In[ ]:


get_ipython().run_line_magic('ls', '../input/titanic/')


# In[ ]:


import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb_origin
import optuna.integration.lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

input_path = Path("../input/titanic/")
train_path = input_path / "train.csv"
test_path = input_path / "test.csv"
submit_path = input_path / "gender_submission.csv"


# In[ ]:


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# In[ ]:


# Reference
# https://www.kaggle.com/currypurin/titanic-lightgbm
drop_cols = ['PassengerId', 'Name', 'Cabin', 'Ticket']
train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


def preprocess(df):
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df = pd.get_dummies(df, columns=['Embarked'])
    return df


# In[ ]:


train = preprocess(train)
test = preprocess(test)


# In[ ]:


target_col = 'Survived'
target = train[target_col]  
train.drop(columns=[target_col], inplace=True)  


# In[ ]:


n_splits = 4
kf = KFold(
        n_splits=n_splits,
        random_state=0)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))


# In[ ]:


for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(train)):
    print('Fold {}/{}'.format(fold_idx + 1, n_splits))
    trn_data = lgb_origin.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb_origin.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])
    
    # LightGBMTuner
    # Reference: https://gist.github.com/smly/367c53e855cdaeea35736f32876b7416
    best_params = {}
    tuning_history = []

    params = {
                'objective': 'binary',
                'metric': 'auc',
            }

    lgb.train(
        params,
        trn_data,
        num_boost_round=10000,
        valid_sets=[trn_data, val_data],
        early_stopping_rounds=100,
        verbose_eval=200,
        best_params=best_params,
        tuning_history=tuning_history)
    
    pd.DataFrame(tuning_history).to_csv('./tuning_history.csv')
    
    best_params['learning_rate'] = 0.05
    
    # origin LightGBM Model
    model = lgb_origin.train(
            best_params,
            trn_data,
            num_boost_round=20000,
            valid_names=['train', 'valid'],
            valid_sets=[trn_data, val_data],
            early_stopping_rounds=1000,
            verbose_eval=1000)
    
    oof[val_idx] = model.predict(train.iloc[val_idx], num_iteration=model.best_iteration)
    predictions += model.predict(test, num_iteration=model.best_iteration) / n_splits
    print("AUC: {}".format(roc_auc_score(target[val_idx], oof[val_idx])))


# In[ ]:


submit = pd.read_csv(submit_path)


# In[ ]:


submit


# In[ ]:


predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0


# In[ ]:


submit["Survived"] = predictions.astype(int)


# In[ ]:


submit


# In[ ]:


submit.to_csv("submission.csv", index=False)


# In[ ]:




