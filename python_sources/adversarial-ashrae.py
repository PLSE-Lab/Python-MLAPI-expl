#!/usr/bin/env python
# coding: utf-8

# In this notebook we'll try to use adversarial validation in order to see how similar/different the train and test sets are. Since we have to merge several datasets in order to get "trainable" data, we'll rely on other kernesl in which this has already been don. This starter version of Adversarial Validation will rely on my [Simple Linear Regression Benchmark kernel](https://www.kaggle.com/tunguz/simple-linear-regression-benchmark/) for train and test sets. 

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


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/simple-linear-regression-benchmark/new_train.csv')\ntest = pd.read_csv('../input/simple-linear-regression-benchmark/new_test.csv')")


# In[ ]:


train.shape


# In[ ]:


columns = train.columns
columns


# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


test.shape


# In[ ]:


train = train.sample(frac=0.1, random_state=3)
test = test.sample(frac=0.05, random_state=3)

#train = train.sample(frac=0.2, random_state=3)
#test = test.sample(frac=0.1, random_state=3)
gc.collect()


# In[ ]:


train['target'] = 0
test['target'] = 1


# In[ ]:


train_test = pd.concat([train, test], axis =0)



#imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp.fit(train_test[columns])
#train_test[columns] = imp.transform(train_test[columns])
target = train_test['target'].values


del train, test
gc.collect()


# In[ ]:


train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
del train_test
gc.collect()


# In[ ]:


train['target']


# In[ ]:


#train_y = train['target'].values.reshape(-1, 1)
#test_y = test['target'].values.reshape(-1,1)
train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)
gc.collect()


# In[ ]:


#train = train.values
#test = test.values
gc.collect()


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 2,
         'learning_rate': 0.2,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 50
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# In[ ]:


#clf = LogisticRegression()
#clf.fit(train, train_y)


# In[ ]:


#preds[:,0].shape


# In[ ]:


#test_y.flatten().shape


# In[ ]:


#roc_auc_score(test_y.flatten(), preds[:,1])


# AUC of 0.52 is actually pretty decent. Seems that at least at this level of analysis there is not much difference between the train and test sets. 

# AUC of 0.56 reveals some dicrepancy between the train and test sets. Let's take a look at the feature importances:

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# Looks like Dew Temperature is the most distinct feature between the train and test datasets.

# In[ ]:




