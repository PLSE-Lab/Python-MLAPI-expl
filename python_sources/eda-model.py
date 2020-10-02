#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. This notebook is still **very** raw - the first few EDA sections have been decently explored/formatted, but the latter predictive sections are completely uncommented. I hope to work on those as my, very limited, time permits.
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import operator
#import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb


# Now let us look at the input folder. Here we find all the relevant files for this competition.

# In[ ]:


print(os.listdir("../input"))


# We see that the input folder only contains three files ```train.csv```, ```test.csv```, and ```sample_submission.csv```. It seems that for this competition we don't have to do any complicated combination and mergers of files.
# 
# Now let's import and take a glimpse at these files.

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


train_df.info()


# In[ ]:


train_df.isnull().values.sum(axis=0)


# In[ ]:


train_df_describe = train_df.describe()
train_df_describe


# We see that the train dataset is fairly small - just under 10,000 rows. It has 143 features, including the Id and Target. Total number of features that can be used for training is 141, which includes 8 float-valued, 130 integer-valued, and 4 object valued, which need to be converted to numerical values. We also see that there are 5 features with missing data, including 3 that are dominated by missing values.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


test_df.shape


# In[ ]:


test_df.info()


# In[ ]:


test_df.isnull().values.sum(axis=0)


# In[ ]:


test_df_describe = test_df.describe()
test_df_describe


# The test set is almost 2.5 times larger than the train set. It also has 5 features with missing data.
# 
# Now, let's take a look at the target. We want to see the distribution of target values in the train set.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train_df.Target.values, bins=4)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


np.unique(train_df.Target.values)


# We see that tehre are only 4 numerical values in for the target. Value 4 seems to dominate, with about 60% of all values. 

# In[ ]:


pd.value_counts(train_df.Target)


# Now we want to subset the features, so that they don't include Id and target.

# In[ ]:


columns_to_use = train_df.columns[1:-1]


# In[ ]:


columns_to_use


# We'll set up the new variable ```y``` that will be our target variable for training.

# In[ ]:


y = train_df['Target'].values-1


# Next, we'll combine the train and test sets, so we can consistenly label encode all the categorical features.

# In[ ]:


train_test_df = pd.concat([train_df[columns_to_use], test_df[columns_to_use]], axis=0)
cols = [f_ for f_ in train_test_df.columns if train_test_df[f_].dtype == 'object']

for col in cols:
    le = LabelEncoder()
    le.fit(train_test_df[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
del le


# Now is the time to build our first model. We'll use make a simple LGBM model.

# In[ ]:


train = lgb.Dataset(train_df[columns_to_use].astype('float'),y ,feature_name = "auto")


# In[ ]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 6,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': len(np.unique(y)),
}


# In[ ]:


clf = lgb.train(params,
        train,
        num_boost_round = 500,
        verbose_eval=True)


# In[ ]:


preds1 = clf.predict(test_df[columns_to_use])


# In[ ]:


xgb_params = {
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.9,
        'colsample_bytree': 0.84,
        'objective': 'multi:softprob',
        'scale_pos_weight': 1,
        'eval_metric': 'merror',
        'silent': 1,
        'verbose': False,
        'num_class': 4,
        'seed': 44}
    
d_train = xgb.DMatrix(train_df[columns_to_use].values.astype('float'), y)
d_test = xgb.DMatrix(test_df[columns_to_use].values.astype('float'))
    
model = xgb.train(xgb_params, d_train, num_boost_round = 500, verbose_eval=100)
                        
xgb_pred = model.predict(d_test)


# In[ ]:


xgb_pred.shape


# In[ ]:


preds = 0.5*preds1 + 0.5*xgb_pred

preds = np.argmax(preds, axis = 1) +1
preds


# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()


# In[ ]:


sample_submission['Target'] = preds
sample_submission.to_csv('simple_lgbm_xgb_1.csv', index=False)
sample_submission.head()


# In[ ]:


np.mean(preds)


# To be continued ...

# In[ ]:




