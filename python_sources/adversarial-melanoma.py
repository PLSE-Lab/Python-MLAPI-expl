#!/usr/bin/env python
# coding: utf-8

# So far it looks like there is not much of a gap between the local validation and the LB socres. However, there **is** a consistent gap in the experiemtns that have been run so far, and there is plenty of evidence that the training and validation datasets are statistically different.
# 
# In this notebook we'll try to assess the degree of difference using adversarial validation approach. For our features we'll just use the pixel-level data from the 32x32 resized images. This data has thus far been surprisingly informative about the images in this dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics

from sklearn import preprocessing
import gc


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')
test = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')


# In[ ]:


target = np.hstack([np.zeros(train.shape[0],), np.ones(test.shape[0],)])
train_test = np.vstack([train, test])
print(train_test.shape)
print(target.shape)


# In[ ]:


del train, test
gc.collect()
train_test = train_test.reshape((train_test.shape[0], 32*32*3))


# In[ ]:


train_test.shape


# In[ ]:


train, test, y_train, y_test = model_selection.train_test_split(train_test, target, test_size=0.33, random_state=42, shuffle=True)
del target, train_test
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=y_train)
test = lgb.Dataset(test, label=y_test)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.05,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 56,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 500
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# The adversarial validation AUC of 0.65 is not terrible, but is definitely significant. 
# 
# We'll now take a look at the feature importances, although since we are just using the pixel-level data they may not be that insightful.

# In[ ]:


features = [f'c_{i}' for i in range(3072)]


# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(25))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:




