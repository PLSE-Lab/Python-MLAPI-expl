#!/usr/bin/env python
# coding: utf-8

# # Regression model to predict missing EXT_SOURCE_x values
# The goal is to build a regression model to predict `EXT_SOURCE_x` values based on other features (as the external scoring agencies might have done to get those numbers).

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#set a fun plot style
#plt.xkcd()
import warnings
warnings.simplefilter(action='ignore')

import lightgbm as lgb

import os
PATH = "../input/"
print(os.listdir(PATH))


# ## Read in the data.

# In[22]:


application_train = pd.read_csv(PATH+'application_train.csv')


# ## Drop several less useful columns

# In[23]:


cols_2drop = [f_ 
              for f_ in application_train.columns 
              if 'FLAG_DOCUMENT' in f_ 
              or 'AVG' in f_
              or 'MODE' in f_
              or 'MEDI' in f_]
cols_2drop += ['SK_ID_CURR', 'TARGET']


# In[24]:


y=application_train['TARGET']
application_train.drop(cols_2drop, axis=1, inplace=True)


# ## Ordered/binary category encoding
# LightGBM can handle categorical features internally, however, we can add ordering for those features, where it is applicable and to simple 0/1 encoding for binary N/Y features

# In[25]:


encoding = {'NAME_CONTRACT_TYPE': {'Cash loans': 1, 'Revolving loans': 0},
            'FLAG_OWN_CAR': {'N': 0, 'Y': 1},
            'FLAG_OWN_REALTY': {'N': 0, 'Y': 1},
            'CODE_GENDER': {'M':0, 'F':1, 'XNA':np.nan},
            'NAME_EDUCATION_TYPE': {'Lower secondary': 0,
                                    'Secondary / secondary special': 1,
                                    'Incomplete higher': 2,
                                    'Higher education': 3,
                                    'Academic degree': 4}
           }


# In[26]:


for c_, map_ in encoding.items():
    application_train[c_] = application_train[c_].replace(map_)


# ## Convert 'object' types into 'category'

# In[27]:


cat_cols = application_train.select_dtypes(include='object')


# In[28]:


for c in cat_cols.columns:
    application_train[c] = application_train[c].astype('category')


# Just for info, let's look at the classes in the remaining categories

# In[29]:


for c in cat_cols.columns:
    print('======== {} =========='.format(c))
    print(cat_cols[c].value_counts())


# ## Look at the missing values

# In[30]:


# replace DAYS_EMPLOYED == 365243
application_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)


# In[31]:


null_train = application_train.isnull()


# In[12]:


plt.figure(figsize=(18,20))
sns.heatmap(null_train.T)#.iloc[:100000,:]


# ### Closer look on EXT_SOURCE missing values

# In[32]:


cols_ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

plt.figure(figsize=(18,6))
sns.heatmap(null_train.loc[:,cols_ext].T)


# Clearly, `EXT_SOURCE_2` has very small number of missing values, while `EXT_SOURCE_3` has a larger fraction, and `EXT_SOURCE_1` are missing even more often.

# ## What do EXT_SOURCE values correlate with

# In[33]:


corr_train = application_train.corr()


# In[34]:


plt.figure(figsize=(6,10))
sns.heatmap(corr_train[[f_ for f_ in application_train.columns if "EXT_" in f_]], cmap='BrBG', vmin=-1, vmax=1)
plt.savefig('ext_source_corr.png')


# `EXT_SOURCE_2` is clearly strongly correlated with the applicant age.

# ## Define a function for classifier fitting

# In[36]:


from sklearn.model_selection import KFold
folds = KFold(n_splits= 5, shuffle=True, random_state=101)


# In[51]:


from sklearn.metrics import roc_auc_score

def train_model(X_, y_, folds_):
    
    score = 0
    feat_imp  = np.zeros(X_.shape[1])
    oof_preds = np.zeros(X_.shape[0])
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(X_, y_)):
        X_train, y_train = X_.iloc[trn_idx], y_.iloc[trn_idx]
        X_val, y_val     = X_.iloc[val_idx], y_.iloc[val_idx]
    
        clf = lgb.LGBMClassifier(learning_rate=0.05,
                         num_leaves=10,
                         max_depth=-1, n_estimators=5000,
                         colsample_bytree=0.9, subsample=0.9,
                         reg_alpha=0.1, reg_lambda=0.1,
                         metric='None',
                         random_state=314, min_child_samples=100,
                         silent=True, n_jobs=4)
        # lightgbm training parameters for early stoping
        fit_params_={"early_stopping_rounds":50, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_val,y_val)],
            'eval_names': ['valid'],
            'verbose': 500,
            'categorical_feature': 'auto'}
        _ = clf.fit(X_train, y_train, **fit_params_)
        #
        feat_imp += clf.booster_.feature_importance('gain') / folds_.n_splits
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        
    score = roc_auc_score(y_, oof_preds)
    print('The final OOF score: {}'.format(score))
    
    feat_imp = pd.Series(feat_imp, index=application_train.columns)
    feat_imp.nlargest(10).plot(kind='barh', figsize=(8,10))
    
    return  score


# ## Do a basic modelbefore EXT_SOURCE imputation was done

# In[52]:


score_orig = train_model(application_train, y, folds)


# # Model fitting for imputation
# We will use LightGBM regression model - LightGBM allows to build very sophysticated models with a very short training time.

# In[43]:


def run_imputer(df, y_name='EXT_SOURCE_1', fobj='mse'):
    
    print('Impute {}'.format(y_name))
    
    notnans = df[y_name].notnull()
    nans = df[y_name].isnull()
    idx_nans = nans.nonzero()[0]
    
    if nans.sum() == 0:
        print('Nothin to impute. Do not run the LightGBMImputer')
        return
    
    train = df[notnans]
    test = df[nans]
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(train.drop(cols_ext, axis=1), train[y_name], test_size=0.20, random_state=314)
    print('Training size = {}'.format(train.shape))
    print('Test size = {}'.format(test.shape))
    
    # lightgbm training parameters for early stoping
    fit_params={"early_stopping_rounds":50, 
            "eval_metric" : ['mae', 'mse'], 
            "eval_set" : [(X_val,y_val)],
            'eval_names': ['valid'],
            'verbose': 500,
            'categorical_feature': 'auto'}
    
    #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
    clf1 = lgb.LGBMRegressor(objective=fobj, learning_rate=0.1,
                         num_leaves=10,
                         max_depth=-1, n_estimators=5000,
                         colsample_bytree=0.9, subsample=0.9,
                         reg_alpha=0.1, reg_lambda=0.1,
                         random_state=314, min_child_samples=100,
                         silent=True, n_jobs=4)
    clf1.fit(X_train, y_train, **fit_params)
    
    df[y_name][nans.nonzero()[0]] = clf1.predict(test.drop(cols_ext, axis=1))


# In[44]:


for c in cols_ext:
    run_imputer(application_train, y_name=c)


# ## Train again using imputed EXT_SOURCE data
# This allows us to avoid overtraining and we do not need to optimise the number of trees

# In[53]:


score_imp = train_model(application_train, y, folds)


# In[ ]:


print('Final comparison of ROC AUC scores: original = {}, with imputation = {:2f}'.format(score_orig, score_imp))


# As an outcome, precision of the classifier does not depend much on the advanced imputer and if anything gets even a bit worse
