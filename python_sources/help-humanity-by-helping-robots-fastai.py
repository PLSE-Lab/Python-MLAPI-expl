#!/usr/bin/env python
# coding: utf-8

# In[436]:


import os
import time
import numpy as np
import pandas as pd
from seaborn import countplot,lineplot, barplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix


from fastai import *
from fastai.tabular import *
from fastai.basic_data import DataBunch
from tqdm import tqdm_notebook

from bayes_opt import BayesianOptimization
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# In[437]:


tr = pd.read_csv('../input/X_train.csv')
te = pd.read_csv('../input/X_test.csv')
target = pd.read_csv('../input/y_train.csv')
ss = pd.read_csv('../input/sample_submission.csv')


# In[438]:


tr.head()


# In[439]:


tr.shape, te.shape


# In[440]:


countplot(y = 'surface', data = target)
plt.show()


# We need to classify on which surface our robot is standing.
# 
# So, its a simple classification task. Multi-class to be specific.

# In[441]:


len(tr.measurement_number.value_counts())


# What's that?
# Each series has 128 measurements. 

# In[442]:


tr.shape[0] / 128, te.shape[0] / 128


# So, we have 3810 train series, and 3816 test series.
# Let's engineer some features!

# ## Feature Engineering

# In[443]:


def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


# In[444]:


def fe_step0 (actual):
    
    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html
    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html
    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
        
    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)
    actual['mod_quat'] = (actual['norm_quat'])**0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']
    
    return actual

def fe_step1 (actual):
    """Quaternions to Euler Angles"""
    
    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual


# In[445]:


get_ipython().run_cell_magic('time', '', 'tr = fe_step0(tr)\nte = fe_step0(te)\n\ntr = fe_step1(tr)\nte = fe_step1(te)')


# In[446]:


def fe(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +
                             data['angular_velocity_Z'])** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +
                             data['linear_acceleration_Z'])**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 +
                             data['orientation_Z'])**0.5
   
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number', 'group_id']:
            continue
        if col in ['surface']:
            df[col] = data
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df


# In[447]:


get_ipython().run_cell_magic('time', '', 'tr = fe(tr)\nte = fe(te)\ntr.head()')


# In[448]:


tr.shape, te.shape


# In[449]:


tr.head()


# In[450]:


tr = tr.merge(target, on='series_id', how='inner')
tr = tr.drop(['group_id', 'series_id'], axis=1)
tr.shape, te.shape


# In[451]:


le = LabelEncoder()
tr['surface'] = le.fit_transform(tr['surface'])


# In[452]:


tr.fillna(0, inplace = True)
te.fillna(0, inplace = True)


# In[453]:


tr.replace(-np.inf, 0, inplace = True)
tr.replace(np.inf, 0, inplace = True)
te.replace(-np.inf, 0, inplace = True)
te.replace(np.inf, 0, inplace = True)


# In[454]:


tr.head()


# ## fastai model

# In[455]:


features = tr.drop('surface', axis=1).columns.values


# In[456]:


BATCH_SIZE = 64
random.seed(2019)
valid_idx = random.sample(list(tr.index.values), int(len(tr)*0.05))


# In[457]:


def get_data_learner(train_df, train_features, valid_idx, 
                     lr=0.02, epochs=1, layers=[512, 512, 256], ps=[0.2, 0.2, 0.2], name='learner'):
    data = TabularDataBunch.from_df(path='.', df=train_df, 
                                    dep_var='surface', 
                                    valid_idx=valid_idx, 
                                    cat_names=[], 
                                    cont_names=train_features, 
                                    bs=BATCH_SIZE,
                                    procs=[Normalize],
                                    test_df=te)
    learner = tabular_learner(data, layers=layers, ps=ps, metrics=[accuracy], use_bn=True)
    return learner, data


# In[458]:


learner, data = get_data_learner(tr, features, np.array(valid_idx))


# In[459]:


learner.fit_one_cycle(5, 1e-2)


# In[460]:


learner.lr_find()
learner.recorder.plot()


# In[461]:


learner.fit_one_cycle(5, 1e-3)


# In[462]:


learner.lr_find()
learner.recorder.plot()


# In[463]:


learner.fit_one_cycle(5, 1e-4)


# In[464]:


learner.lr_find()
learner.recorder.plot()


# In[465]:


learner.fit_one_cycle(5, 5e-5)


# In[466]:


val_predictions = np.squeeze(to_np(learner.get_preds(DatasetType.Valid)[0])).argmax(axis=1)


# In[467]:


predictions = np.squeeze(to_np(learner.get_preds(DatasetType.Test)[0])).argmax(axis=1)
te['surface'] = predictions
te['surface'] = le.inverse_transform(predictions.round().astype(np.int32))


# In[468]:


te[['surface']].to_csv(f'submission_fastai.csv')
te[['surface']].head()


# In[469]:


te['surface'].value_counts()


# ## Confusion matrix

# In[470]:


import itertools

def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):
    cm = confusion_matrix(truth, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', size=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()


# In[471]:


plot_confusion_matrix(tr['surface'].iloc[valid_idx], val_predictions, le.classes_)


# In[ ]:




