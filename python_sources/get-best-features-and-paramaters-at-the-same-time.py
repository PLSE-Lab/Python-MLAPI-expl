#!/usr/bin/env python
# coding: utf-8

# 
# Thank you!  
# [Robust, Lasso, Patches with RFE & GS](https://www.kaggle.com/featureblind/robust-lasso-patches-with-rfe-gs)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import os.path
import itertools
from itertools import chain

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cluster, datasets, mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D, GlobalAveragePooling2D,     GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras


# In[ ]:


from PIL import Image
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm


# ### load data

# In[ ]:


ls -la ../input


# In[ ]:


src_dir = '../input'
train_csv = pd.read_csv(os.path.join(src_dir, 'train.csv'))
print(train_csv.shape)
train_csv.head()


# In[ ]:


x_train0 = train_csv.iloc[:,2:].values
print(x_train0.shape)
x_train0


# In[ ]:


y_train0 = train_csv.target.values
print(y_train0.shape)
y_train0


# In[ ]:


y_cat_train0 = to_categorical(y_train0)
print(y_cat_train0.shape)
y_cat_train0[:5]


# In[ ]:


test_csv = pd.read_csv(os.path.join(src_dir, 'test.csv'))
print(test_csv.shape)
test_csv.head()


# In[ ]:


x_test = test_csv.iloc[:,1:].values
print(x_test.shape)
x_test


# In[ ]:


sample_submission_csv = pd.read_csv(os.path.join(src_dir, 'sample_submission.csv'))
print(sample_submission_csv.shape)
sample_submission_csv.head()


# ### Create model

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# In[ ]:


# some heuristic settings
rfe_min_features = 12
rfe_step = 15
rfe_cv = 20
sss_n_splits = 20
sss_test_size = 0.35
grid_search_cv = 20
noise_std = 0.01
r2_threshold = 0.185
random_seed = 213

np.random.seed(random_seed)


# In[ ]:


def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5


# In[ ]:


x = RobustScaler().fit_transform(np.concatenate((x_train0, x_test), axis=0))
x_train0 = x[:250]
x_test = x[250:]


# In[ ]:


x_train0 += np.random.normal(0, noise_std, x_train0.shape)


# In[ ]:


robust_roc_auc = make_scorer(scoring_roc_auc)

# define model and its parameters
# model = Lasso(alpha=0.031, tol=0.01, random_state=random_seed, selection='random')

param_grid = {
            'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
            'tol'   : [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]
        }
# param_grid = {
#             'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
#         }

# define recursive elimination feature selector
# feature_selector = RFECV(model,
#                          min_features_to_select=rfe_min_features,
#                          scoring=robust_roc_auc,
#                          step=rfe_step,
#                          verbose=0,
#                          cv=rfe_cv,
#                          n_jobs=-1)


# In[ ]:


class RFECV_wr(RFECV):
    
    def __init__(self, alpha=1.0, tol=0.0001, # for lasso
                       step=1, min_features_to_select=1, cv='warn',
                       scoring=None, verbose=0, n_jobs=None):
        estimator = Lasso(alpha=0.031, tol=0.001,
                          random_state=random_seed, selection='random')
        super().__init__(estimator, step=step,
                         min_features_to_select=min_features_to_select,
                         cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    
    def set_params(self, **params):
        if 'alpha' in params:
            self.estimator.set_params(alpha=params['alpha'])
        if 'tol' in params:
            self.estimator.set_params(tol=params['tol'])
        return self

feature_selector2 = RFECV_wr(min_features_to_select=rfe_min_features,
                             scoring=robust_roc_auc,
                             step=rfe_step,
                             verbose=0,
                             cv=rfe_cv,
                             n_jobs=-1)


# In[ ]:


feature_selector2.get_params()


# In[ ]:


param_range = np.logspace(-1.7, -1.5, 10)
param_range


# In[ ]:


scorer = make_scorer(roc_auc_score)
scorer
cv_splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.35, random_state=0)
cv_splitter.get_n_splits(x_train0, y_train0)


# In[ ]:


train_scores, test_scores = validation_curve(
    feature_selector2, x_train0, y_train0,
    param_name="alpha", param_range=param_range,
    cv=cv_splitter, scoring=scorer, n_jobs=None, verbose=0)


# In[ ]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
train_scores_mean


# In[ ]:


test_scores_mean


# In[ ]:


plt.title("Validation Curve")
plt.xlabel("param")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")


# In[ ]:


print("counter | val_mse  |  val_mae  |  val_roc  |  val_r2    |  alpha     | feature_count ")
print("-------------------------------------------------------------------------------------")

importances = np.zeros((300,))
predictions = pd.DataFrame()
counter = 0
for train_index, val_index in StratifiedShuffleSplit(n_splits=sss_n_splits, test_size=sss_test_size, random_state=random_seed).split(x_train0, y_train0):
    X, val_X = x_train0[train_index], x_train0[val_index]
    y, val_y = y_train0[train_index], y_train0[val_index]
    
    # get the best features and the best paramaters at the same time
    grid_search = GridSearchCV(feature_selector2,
                               param_grid=param_grid,
                               verbose=0,
                               n_jobs=None,
                               scoring=robust_roc_auc,
                               cv=grid_search_cv)
    
    grid_search.fit(X, y)
    
    # score our fitted model on validation data
    val_y_pred = grid_search.best_estimator_.predict(val_X)
    val_mse = mean_squared_error(val_y, val_y_pred)
    val_mae = mean_absolute_error(val_y, val_y_pred)
    val_roc = roc_auc_score(val_y, val_y_pred)
    val_cos = cosine_similarity(val_y.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_dst = euclidean_distances(val_y.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_r2  = r2_score(val_y, val_y_pred)
    
    # if model did well on validation, save its prediction on test data, using only important features
    # r2_threshold (0.185) is a heuristic threshold for r2 error
    # you can use any other metric/metric combination that works for you
    if val_r2 > r2_threshold:
        message = '<-- OK'
        prediction = grid_search.best_estimator_.predict(x_test)
        predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
        importances += grid_search.best_estimator_.support_.astype(int)
    else:
        message = '<-- skipping'

    print("{0:2}      | {1:.4f}   |  {2:.4f}   |  {3:.4f}   |  {4:.4f}    |  {5:.4f}    |  {6:3}         {7}  ".format(
        counter,
        val_mse,
        val_mae,
        val_roc,
        val_r2,
        grid_search.best_estimator_.estimator_.get_params()['alpha'],
        grid_search.best_estimator_.n_features_,
        message))
    
    counter += 1


# In[ ]:


grid_search.best_estimator_.estimator_.get_params()


# In[ ]:


mean_pred = pd.DataFrame(predictions.mean(axis=1))
mean_pred.index += 250
mean_pred.columns = ['target']
mean_pred.to_csv('submission.csv', index_label='id', index=True)
mean_pred.head()


# In[ ]:


importances


# In[ ]:


print(importances.shape)
sns.distplot(importances)


# In[ ]:





# In[ ]:





# In[ ]:




