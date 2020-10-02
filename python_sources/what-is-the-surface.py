#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this kernel I tried to make easy and fast way to get fine perfomance. 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# # The Main Info
# As we can see we have *train set *and *test set* almost same shape and In this case we see that classes imbalanced enough. 

# In[ ]:


print("Files in the input folder:")
print(os.listdir("../input"))
train = pd.read_csv('../input/X_train.csv')
test = pd.read_csv('../input/X_test.csv')
y = pd.read_csv('../input/y_train.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print("\nX_train shape: {}, X_test shape: {}".format(train.shape, test.shape))
print("y_train shape: {}".format(y.shape))
y["surface"].value_counts().plot(kind='barh')


# # Feature Engineering
# This part was taken from [Surface Recognition Baseline kernel](http://www.kaggle.com/jsaguiar/surface-recognition-baseline).

# In[ ]:


def feature_extraction(raw_frame):
    frame = pd.DataFrame()
    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']
    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']
    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']
    
    for col in tqdm(raw_frame.columns[3:]):
        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()
        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()
        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()
        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()
        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']
        
        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
    return frame


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = feature_extraction(train)\ntest_df = feature_extraction(test)')


# # Data preparing
# Here we are encoding labels to numeric values and split into validation and train set. For all of this we use **LabelEncoder** and **train_test_split** from **Scikit-learn**.

# In[ ]:


Y = y["surface"].values
lbe = LabelEncoder().fit(Y)
Y = lbe.transform(Y)
X = train_df.values


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
inp = Pool(X_train, y_train)


# # Model and Prediction
# In this case we use **CatBoostClassifier** which evalute by simple **Accuracy** optimized via **MultiClass loss function** with **early stopping**.
# (about all CatBoost metrics you can read [here](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#loss-functions))

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = CatBoostClassifier(\n    loss_function=\'MultiClass\',\n    eval_metric=\'Accuracy\',\n    learning_rate=0.03,\n    task_type="GPU",\n    iterations=100000,\n    random_seed=42,\n    od_type=\'Iter\',\n    early_stopping_rounds=400,\n    verbose=0\n)\nmodel.fit(inp, eval_set=(X_val, y_val))\nprint(\'Validation: \', model.get_best_score()[\'validation_0\'])')


# In[ ]:


pred = lbe.inverse_transform(model.predict(test_df.values).reshape(-1).astype(int))
sub.surface = pred
sub.to_csv('submission.csv', index=False)

