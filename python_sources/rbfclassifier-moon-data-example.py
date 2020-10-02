#!/usr/bin/env python
# coding: utf-8

# * [RBFClassifier scikit-learn API](https://github.com/darecophoenixx/wordroid.sblo.jp/wiki/RBFClassifier-RBFRegressor)
# * moon

# In[ ]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3
from keras_ex.gkernel.sklearn import (
    RBFRegressor, RBFClassifier,
    make_model_gkernel1,
    make_model_gkernel2,
    make_model_gkernel3,
    make_model_out,
    make_model
)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import f1_score, classification_report, confusion_matrix, make_scorer
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras import backend as K
import tensorflow as tf


# In[ ]:


from sklearn.datasets import make_moons

n_samples = 2500
X1, y1 = make_moons(n_samples=n_samples, noise=.15, random_state=0)
df = pd.DataFrame(X1)
df.columns = ["col1", "col2"]
df['cls'] = y1

sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, size=8, markers='o')


# In[ ]:


n_samples = 500
X2, y2 = make_moons(n_samples=n_samples, noise=.15, random_state=0)
X2[:,0] += 1
X2[:,1] += 1
df = pd.DataFrame(X2)
df.columns = ["col1", "col2"]
df['cls'] = y2

sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, size=8, markers='o')


# In[ ]:


X = np.r_[X1,X2]
y = np.concatenate([y1, y2])

df = pd.DataFrame(X)
df.columns = ["col1", "col2"]
df['cls'] = y

sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, size=8, markers='o')


# In[ ]:


X.shape, X.min(), X.max()


# In[ ]:


y.shape


# In[ ]:


N = y.shape[0]
index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = y[index[index % 2 != 0]]
xtest = X[index[index % 2 == 0],:]
yans = y[index[index % 2 == 0]]


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, validation_curve
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, Lasso
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import joblib

from lightgbm import LGBMClassifier


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def f1_scorer(estimator, X, y):
    pred = estimator.predict(X)
    s = f1_score(y, pred, average='macro')
    return s


# In[ ]:


clf = RBFClassifier(
    num_lm=200,
    lm_select_from_x=True
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hst = clf.fit(xtrain, ytrain, epochs=200, batch_size=300, verbose=0)')


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].set_title('loss')
ax[0].plot(list(range(len(hst["loss"]))), hst["loss"], label="Train loss")
ax[1].set_title('accuracy')
ax[1].plot(list(range(len(hst["loss"]))), hst["accuracy"], label="accuracy")
ax[2].set_title('learning rate')
ax[2].plot(list(range(len(hst["loss"]))), hst["lr"], label="learning rate")
ax[0].legend()
ax[1].legend()
ax[2].legend()


# In[ ]:


pred = clf.predict(xtrain)


# In[ ]:


print('F1_SCORE :', f1_score(ytrain, pred, average='macro'))
print(classification_report(ytrain, pred))
print(confusion_matrix(ytrain, pred))


# In[ ]:


pred_test = clf.predict(xtest)
pred_test.shape


# In[ ]:


print('F1_SCORE :', f1_score(yans, pred_test, average='macro'))
print(classification_report(yans, pred_test))
print(confusion_matrix(yans, pred_test))


# In[ ]:


'''get landmarks'''
lm = clf.current_lm()
lm.shape


# In[ ]:


from matplotlib.colors import ListedColormap

h = .01
x_min, x_max = xtrain[:, 0].min() - .1, xtrain[:, 0].max() + .1
y_min, y_max = xtrain[:, 1].min() - .1, xtrain[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

#cm = plt.cm.coolwarm
cm = plt.cm.nipy_spectral
cm_bright = ListedColormap(['#FFFF00', '#00FFFF'])
y_pred = y_pred[:,1].reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(xtrain[:,0], xtrain[:,1], c=ytrain, cmap=cm_bright, edgecolors='k')
plt.scatter(lm[:,0], lm[:,1], c='red', s=64, marker='s', edgecolors='w')


# In[ ]:





# ## SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


get_ipython().run_cell_magic('time', '', "svc = SVC(gamma='scale', probability=True)\nsvc.fit(xtrain, ytrain)")


# In[ ]:


pred = svc.predict(xtrain)


# In[ ]:


print('F1_SCORE :', f1_score(ytrain, pred, average='macro'))
print(classification_report(ytrain, pred))
print(confusion_matrix(ytrain, pred))


# In[ ]:


pred_test = svc.predict(xtest)
pred_test.shape


# In[ ]:


print('F1_SCORE :', f1_score(yans, pred_test, average='macro'))
print(classification_report(yans, pred_test))
print(confusion_matrix(yans, pred_test))


# In[ ]:


h = .01
x_min, x_max = xtrain[:, 0].min() - .1, xtrain[:, 0].max() + .1
y_min, y_max = xtrain[:, 1].min() - .1, xtrain[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = svc.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

#cm = plt.cm.coolwarm
cm = plt.cm.nipy_spectral
cm_bright = ListedColormap(['#FFFF00', '#00FFFF'])
y_pred = y_pred[:,1].reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(xtrain[:,0], xtrain[:,1], c=ytrain, cmap=cm_bright, edgecolors='k')


# In[ ]:





# In[ ]:





# ## LogisticRegression

# In[ ]:


cv_splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.35, random_state=0)
cv_splitter.get_n_splits(xtrain, ytrain)


# In[ ]:


param_name = "C"
param_range = np.logspace(-3, 3, 7)
param_range


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_scores, test_scores = validation_curve(\n    LogisticRegression(penalty='l1', multi_class='multinomial', solver='saga', max_iter=10000),\n    xtrain, ytrain,\n    param_name=param_name, param_range=param_range,\n    cv=cv_splitter, n_jobs=1, verbose=2, scoring=f1_scorer)")


# In[ ]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
np.c_[param_range, train_scores_mean, test_scores_mean]


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


'''test data'''
est = LogisticRegression(penalty='l1', multi_class='multinomial', solver='saga', max_iter=10000, C=1.0)
est.fit(xtrain, ytrain)


# In[ ]:


pred = est.predict(xtrain)


# In[ ]:


print('F1_SCORE :', f1_score(ytrain, pred, average='macro'))
print(classification_report(ytrain, pred))
print(confusion_matrix(ytrain, pred))


# In[ ]:


pred_test = est.predict(xtest)


# In[ ]:


print('F1_SCORE :', f1_score(yans, pred_test, average='macro'))
print(classification_report(yans, pred_test))
print(confusion_matrix(yans, pred_test))


# In[ ]:


h = .01
x_min, x_max = xtrain[:, 0].min() - .1, xtrain[:, 0].max() + .1
y_min, y_max = xtrain[:, 1].min() - .1, xtrain[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = est.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FFFF00', '#00FFFF'])
y_pred = y_pred[:,1].reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(xtrain[:,0], xtrain[:,1], c=ytrain, cmap=cm_bright, edgecolors='k')


# In[ ]:





# ## LGBMClassifier

# In[ ]:


cv_splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.35, random_state=0)
cv_splitter.get_n_splits(xtrain, ytrain)


# In[ ]:


param_name = "reg_alpha"
param_range = np.logspace(-3, 3, 7)
param_range


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_scores, test_scores = validation_curve(\n    LGBMClassifier(\n        min_child_samples=20,\n        reg_alpha=0.1\n    ),\n    xtrain, ytrain,\n    param_name=param_name, param_range=param_range,\n    cv=cv_splitter, n_jobs=1, verbose=2, scoring=f1_scorer)')


# In[ ]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
np.c_[param_range, train_scores_mean, test_scores_mean]


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


'''test data'''
est = LGBMClassifier(
        min_child_samples=20,
        reg_alpha=0.1
    )
est.fit(xtrain, ytrain)


# In[ ]:


pred = est.predict(xtrain)


# In[ ]:


print('F1_SCORE :', f1_score(ytrain, pred, average='macro'))
print(classification_report(ytrain, pred))
print(confusion_matrix(ytrain, pred))


# In[ ]:


pred_test = est.predict(xtest)


# In[ ]:


print('F1_SCORE :', f1_score(yans, pred_test, average='macro'))
print(classification_report(yans, pred_test))
print(confusion_matrix(yans, pred_test))


# In[ ]:


h = .01
x_min, x_max = xtrain[:, 0].min() - .1, xtrain[:, 0].max() + .1
y_min, y_max = xtrain[:, 1].min() - .1, xtrain[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = est.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FFFF00', '#00FFFF'])
y_pred = y_pred[:,1].reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(xtrain[:,0], xtrain[:,1], c=ytrain, cmap=cm_bright, edgecolors='k')


# In[ ]:





# In[ ]:





# In[ ]:




