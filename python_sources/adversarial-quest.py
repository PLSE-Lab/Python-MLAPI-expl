#!/usr/bin/env python
# coding: utf-8

# In this notebook we'll try to use adversarial validation in order to see how similar/different the train and test sets are. Since we don't have access to the final train dataset, this exercise is meant more to inform the modeling process than to use the information from the train set for the final submission. All the features for this kernel, as well as the references to previous kernels that created them, can be found here: https://www.kaggle.com/tunguz/distilbert-use-features-just-the-features

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import shap
import os
print(os.listdir("../input"))
from sklearn import preprocessing
import xgboost as xgb
import gc


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = np.load('../input/distilbert-use-features-just-the-features/X_train.npy')\ntest = np.load('../input/distilbert-use-features-just-the-features/X_test.npy')")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


np.zeros((6079,))


# In[ ]:


train_test = np.vstack([train, test])
target = np.hstack([np.zeros((6079,)), np.ones((476,))])
del train, test
gc.collect()


# In[ ]:


train, test, train_y, test_y = model_selection.train_test_split(train_test, target, test_size=0.33, random_state=42, shuffle=True)
del train_test, target
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)
gc.collect()


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 20, 
         'objective':'binary',
         'max_depth': 2,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.5,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 500
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=1000, early_stopping_rounds = 1000)


# That's not too shabby - AUC of 0.61 between train and test sets indicates high variability between distinct features. Unfortunately most of our featues come from the transformers embedding space(s), so it will not be easy to interpret whcih ones are responsible for what. Nonetheless, let's try to take a quick look.

# In[ ]:


features = ['feature_'+str(x) for x in range(3142)]

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# So it seams that "feature_717" is the most distinct feature between the train and test sets, with a long tail of features that have a relatively low value in this regard. 

# In[ ]:




