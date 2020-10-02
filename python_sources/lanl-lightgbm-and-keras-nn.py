#!/usr/bin/env python
# coding: utf-8

# <pre>Inspired by Anton Enns's Kernel
# https://www.kaggle.com/tocha4/lanl-master-s-approach</pre>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from tqdm import tqdm_notebook
import datetime
import time
import random
from joblib import Parallel, delayed

from catboost import Pool, CatBoostRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV, SelectFromModel


# In[ ]:


# This is NN Model creation
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras.callbacks import *


# In[ ]:


print(os.listdir('../input/'))
train_X_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_X_features_865.csv")
train_X_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_X_features_865.csv")
y_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_y.csv", index_col=False,  header=None)
y_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_y.csv", index_col=False,  header=None)


# In[ ]:


train_X = pd.concat([train_X_0, train_X_1], axis=0)
train_X = train_X.reset_index(drop=True)
print(train_X.shape)
train_X.head()


# In[ ]:


y = pd.concat([y_0, y_1], axis=0)
y = y.reset_index(drop=True)
y[0].shape


# In[ ]:


train_y = pd.Series(y[0].values)


# In[ ]:


test_X = pd.read_csv("../input/lanl-master-s-features-creating-0/test_X_features_10.csv")
# del X["seg_id"], test_X["seg_id"]


# In[ ]:


scaler = MinMaxScaler()
train_columns = train_X.columns

train_X[train_columns] = scaler.fit_transform(train_X[train_columns])
test_X[train_columns] = scaler.transform(test_X[train_columns])


# <pre>CAT BOOST Algorithm</pre>

# In[ ]:


train_columns = train_X.columns
n_fold = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=n_fold, shuffle = True, random_state=42)\n\noof = np.zeros(len(train_X))\ntrain_score = []\nfold_idxs = []\n# if PREDICTION: \npredictions = np.zeros(len(test_X))\n\nfeature_importance_df = pd.DataFrame()\n#run model\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):\n    strLog = "fold {}".format(fold_)\n    print(strLog)\n    fold_idxs.append(val_idx)\n\n    X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]\n    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]\n\n    model = CatBoostRegressor(n_estimators=25000, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered", task_type="GPU")\n    model.fit(X_tr, \n              y_tr, \n              eval_set=[(X_val, y_val)], \n#               eval_metric=\'mae\',\n              verbose=2500, \n              early_stopping_rounds=500)\n    oof[val_idx] = model.predict(X_val)\n\n    #feature importance\n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["Feature"] = train_columns\n    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    #predictions\n#     if PREDICTION:\n\n    predictions += model.predict(test_X[train_columns]) / folds.n_splits\n    train_score.append(model.best_score_[\'learn\']["MAE"])\n\ncv_score = mean_absolute_error(train_y, oof)\nprint(f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")')


# In[ ]:


today = str(datetime.date.today())
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

submission["time_to_failure"] = predictions
submission.to_csv(f'CatBoost_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv', index=False)
submission.head()


# <pre>Keras Neural Network</pre>

# In[ ]:


def create_model(input_dim=10):
    model = Sequential()
    model.add(Dense(256, activation="relu",input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))
 
    opt = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(
        loss='mae',
        optimizer=opt,
    )
    return model

patience = 50
call_ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# n_fold = 5\nfolds = KFold(n_splits=n_fold, shuffle = True, random_state=42)\n\nNN_oof = np.zeros(len(train_X))\ntrain_score = []\nfold_idxs = []\n\nNN_predictions = np.zeros(len(test_X))\n\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):\n    strLog = "fold {}".format(fold_)\n    print(strLog)\n    fold_idxs.append(val_idx)\n    \n    X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]\n    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]\n    model = create_model(train_X.shape[-1])\n    model.fit(X_tr, y_tr, epochs=500, batch_size=32, verbose=0, callbacks=[call_ES,], validation_data=[X_val, y_val]) #\n    \n    NN_oof[val_idx] = model.predict(X_val)[:,0]\n    \n    NN_predictions += model.predict(test_X[train_columns])[:,0] / folds.n_splits\n    history = model.history.history\n    tr_loss = history["loss"]\n    val_loss = history["val_loss"]\n    print(f"loss: {tr_loss[-patience]:.3f} | val_loss: {val_loss[-patience]:.3f} | diff: {val_loss[-patience]-tr_loss[-patience]:.3f}")\n    train_score.append(tr_loss[-patience])\n#     break\n    \ncv_score = mean_absolute_error(train_y, NN_oof)\nprint(f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")')


# In[ ]:


today = str(datetime.date.today())
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

submission["time_to_failure"] = NN_predictions
submission.to_csv(f'NN_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv', index=False)
submission.head()


# # Final Submission

# In[ ]:


Scirpus_prediction = pd.read_csv("../input/andrews-new-script-plus-a-genetic-program-model/gpI.csv")
Scirpus_prediction.head()


# In[ ]:


today = str(datetime.date.today())
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

submission["time_to_failure"] = (predictions+NN_predictions+Scirpus_prediction.time_to_failure.values)/3
submission.to_csv(f'FINAL_{today}_submission.csv', index=False)
submission.head()

