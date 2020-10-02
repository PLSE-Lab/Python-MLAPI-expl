#!/usr/bin/env python
# coding: utf-8

# # Overview of different categorical treatments
# This is an overview of  of different ways one can treat categorical features in this competition. The following methods are compared:
# - OHE (one-hot-encoding);
# - label encoding with integers;
# - internal treatment in LightGBM, that implements a search for a next-to-optimal split strategy
# - target encoding using several approaches: 
#   - [category_encoders.TargetEncoder](https://github.com/scikit-learn-contrib/categorical-encoding), 
#   - **NEW** [xam.feature_extraction.SmoothTargetEncoder](https://github.com/MaxHalford/xam/blob/master/docs/feature-extraction.md#smooth-target-encoding) installing the `xam` package from github via the kernel interface,
#   - naive target-encoding with potential data leakage,
#   - target encoding with regularisation (all implementations have `transform` and `regularise` methods for validation/test and train samples, respectively):
#    - stratified KFold;
#    - expanding mean (a la catboost, I believe);
#    - **NEW** nested stratified KFold encoding, inspired by [this kernel by Eduardo Gomes](https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features). It was reworked to implement it as a transformer and treatment of priors are different.
#   
# I'm aware of several other options for targer-encoding implementation, but those are not included here:
#   - smoothing + regularisation noise, following [this kernel by Olivier](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features). But it requires tuning of noise and smoothing parameters.
#   
# 
# This kernel inherited ideas and SW solutions from other public kernels and in such cases I will post direct references to the original results, such that you can get some additional insights from the source.

# ## Import relevant libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
#plt.xkcd()
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
PATH = "../input/"
print(os.listdir(PATH))

# Any results you write to the current directory are saved as output.


# ## Read in the data reducing memory pattern for variables.
# The memory-footprint reduction was copied over from [this kernel](https://www.kaggle.com/gemartin/load-data-reduce-memory-usage)

# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# In[ ]:


application_train = import_data(PATH+'application_train.csv')
application_test = import_data(PATH+'application_test.csv')


# In[ ]:


# replace values that in fact are NANs
application_train['DAYS_EMPLOYED'] = (application_train['DAYS_EMPLOYED'].apply(lambda x: x if x != 365243 else np.nan))


# ## Data preprocessing
# The here we will not do anything fancy and will just drop all features, that are known to have a large fraction of missing values and that have little effect on the model performance

# In[ ]:


def feat_ext_source(df):
    medi_avg_mode = [f_ for f_ in df.columns if '_AVG' in f_ or '_MODE' in f_ or '_MEDI' in f_]
    df.drop(medi_avg_mode, axis=1, inplace=True)
    return df


# In[ ]:


application_train = feat_ext_source(application_train)
application_test  = feat_ext_source(application_test)


# ## Categorical encoding
# The function was taken from [this kernel](https://www.kaggle.com/sz8416/simple-intro-eda-baseline-model-with-gridsearch). It allows to do OneHotEncoding (OHE) keeping only those columns that are common to train and test samples. OHE is performed using `pd.get_dummies`, which allows to convert categorical features, while keeping numerical untouched.

# In[ ]:


# use this if you want to convert categorical features to dummies
def cat_to_dummy(train, test):
    train_d = pd.get_dummies(train, drop_first=False)
    test_d = pd.get_dummies(test, drop_first=False)
    # make sure that the number of features in train and test should be same
    for i in train_d.columns:
        if i not in test_d.columns:
            if i!='TARGET':
                train_d = train_d.drop(i, axis=1)
    for j in test_d.columns:
        if j not in train_d.columns:
            if j!='TARGET':
                test_d = test_d.drop(i, axis=1)
    print('Memory usage of train increases from {:.2f} to {:.2f} MB'.format(train.memory_usage().sum() / 1024**2, 
                                                                            train_d.memory_usage().sum() / 1024**2))
    print('Memory usage of test increases from {:.2f} to {:.2f} MB'.format(test.memory_usage().sum() / 1024**2, 
                                                                            test_d.memory_usage().sum() / 1024**2))
    return train_d, test_d

#application_train_ohe, application_test_ohe = cat_to_dummy(application_train, application_test)


# Add also int label encoding for a train+test pair implemented in the same manner

# In[ ]:


# use this if you want to convert categorical features to int labels
def cat_to_int(train, test):
    mem_orig_train = train.memory_usage().sum() / 1024**2
    mem_orig_test  = test .memory_usage().sum() / 1024**2
    categorical_feats = [ f for f in train.columns if train[f].dtype == 'object' or train[f].dtype.name == 'category' ]
    for f_ in categorical_feats:
        train[f_], indexer = pd.factorize(train[f_])
        test[f_] = indexer.get_indexer(test[f_])
    print('Memory usage of train increases from {:.2f} to {:.2f} MB'.format(mem_orig_train, 
                                                                            train.memory_usage().sum() / 1024**2))
    print('Memory usage of test increases from {:.2f} to {:.2f} MB'.format(mem_orig_test, 
                                                                            test.memory_usage().sum() / 1024**2))
    return categorical_feats, train, test

#categorical_feats, application_train_le, application_test_le = cat_to_int(application_train, application_test)


# Define also target encoding using KFold regularisation as well as expanding mean regularisation a la CatBoost. Both methods are explained in (and inspired by) this nice course on coursera: https://www.coursera.org/learn/competitive-data-science/home/welcome, in particular the two videos from week3: [mean encoding](https://www.coursera.org/learn/competitive-data-science/lecture/b5Gxv/concept-of-mean-encoding) and [encoding regularisation](https://www.coursera.org/learn/competitive-data-science/lecture/LGYQ2/regularization). 
# 
# These transformers were developed privately. So far i have them on kaggle only, but I plan to propagate them into a package on github later on. The final goal would be to have them as full transformers, i.e. to get rid of the `regularise` function, as it would not be compatible with pipelines. However, i did not find a way to reach it so far. Any ideas/suggestions are welcome!

# In[ ]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import TransformerMixin

class TargetEncoder_Base(TransformerMixin):
    '''
    The base class to do basic target encoding. It has no regularisation method
    '''
    def __init__(self, cat_cols=None, y_name='TARGET', tr_type='basic', random_state=0, prefix='ENC', prefix_sep='_'):
        self.cat_cols = cat_cols
        self.gb       = dict()
        self.y_name   = y_name
        self.tr_type = tr_type
        self.random_state = random_state
        self.prefix   = '{}{}'.format(prefix, prefix_sep)
        self.prior    = -1
        super(TargetEncoder_Base, self).__init__()
            
    def transform(self, X, **transform_params):
        X_ = X.copy(deep=True)
        for f_ in self.cat_cols:
            X_ [self.prefix + f_] = X_[f_].map(self.gb[f_]).astype(np.float32)
            X_ [self.prefix + f_].fillna(self.prior, inplace=True)
            del X_[f_]
        return X_
    
    def fit(self, X, y=None, **fit_params):
        self._prefit(X, y)
                
        #concatenate X and y to simplify usage (temporary object)
        XY = self._getXY(X, y)
        
        if self.tr_type == 'basic':
            # learn encodings from the full sample
            for f_ in self.cat_cols:
                self.gb[f_] = XY.groupby(f_)[self.y_name].mean()
        else:
            raise ValueError('Unknown value tr_type = {}'.format(self.tr_type))
        
        del XY   
        return self
    
    def _prefit(self, X, y=None):
        if y is None:
            raise RuntimeError('TargetEncoder_KFold needs y to learn the transform')
            
        # deduce categorical columns, if user did not speficy anything
        if self.cat_cols == None:
            self.cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        # make sure that we store the list of categorical columns as a list
        if not isinstance(self.cat_cols, list):
            try:
                self.cat_cols = self.cat_cols.tolist()
            except:
                RuntimeError('TargetEncoder_Base._prefit() fails to convert `cat_cols` into a list')
                
        #store the full sample mean for encoding of rare categories
        self.prior = y.mean()
        
    def _getXY(self, X, y):
        return pd.concat([X[self.cat_cols], y], axis=1)
    
    

class TargetEncoder_KFold(TargetEncoder_Base):
    '''
    Target encoding applying KFold regularisation
    following procedure outlined in https://www.coursera.org/learn/competitive-data-science/lecture/LGYQ2/regularization
    '''
    def __init__(self, cv=5, **kwargs):
        super(TargetEncoder_KFold, self).__init__(**kwargs)
        self.cv_n     = cv
        self.cv       = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
    
    def regularise(self, X, y=None, **transform_params):
        # a dataframe to store OOF target encodings
        oof = pd.DataFrame(np.zeros(shape=(X.shape[0], len(self.cat_cols))),
                           index=X.index,
                           columns=self.cat_cols)
        #concatenate X and y to simplify usage (temporary object)
        XY = self._getXY(X, y)
        
        # iterate over folds
        for trn_idx, val_idx in self.cv.split(X, y):
            trn = XY.iloc[trn_idx]
            val = XY.iloc[val_idx]
            # iterate over categorical features
            for f_ in self.cat_cols:
                # get target means for each class within category
                te = trn.groupby(f_)[self.y_name].mean()
                # encode the OOF partion
                oof.iloc[val_idx, oof.columns.get_loc(f_)] = val[f_].map(te).astype(np.float32)
        # do finla cosmetics and fill NAN
        oof = oof.add_prefix(self.prefix).fillna(self.prior)
        del XY
        
        X_ = X.drop(self.cat_cols, axis=1)
        return pd.concat([X_, oof], axis=1)


class TargetEncoder_ExpandingMean(TargetEncoder_Base):
    '''
    Target encoding applying expanding mean regularisation
    following procedure outlined in https://www.coursera.org/learn/competitive-data-science/lecture/LGYQ2/regularization
    '''
    def __init__(self, **kwargs):
        super(TargetEncoder_ExpandingMean, self).__init__(**kwargs)
    
    def regularise(self, X, y=None, **transform_params):
        X_ = X.copy(deep=True)
        
        # iterate over categorical features
        for f_ in self.cat_cols:
            gb = self._getXY(X_, y).groupby(f_)[self.y_name]
            # calculate expanding mean
            X_[self.prefix + f_] = ((gb.cumsum() - y) / gb.cumcount()).astype(np.float32).fillna(0)
            del gb
        
        X_.drop(self.cat_cols, axis=1, inplace=True)
        return X_


class TargetEncoder_NestedKFold(TargetEncoder_Base):
    '''
    That's a transformer-like implementation on the approach, 
    that is publicly available:
    R in https://www.kaggle.com/raddar/raddar-extratrees
    python in https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features
    '''
    def __init__(self, cv_inner=5, cv_outer=10, **kwargs):
        super(TargetEncoder_NestedKFold, self).__init__(**kwargs)
        self.cv_inner_n     = cv_inner
        self.cv_outer_n     = cv_outer
        self.cv_outer       = KFold(n_splits=cv_outer, shuffle=True, random_state=self.random_state)
        self.nan_tmp        = 'NAN_TMP'
    
    def regularise(self, X, y=None, **transform_params):
        # a dataframe to store OOF target encodings
        oof = pd.DataFrame(np.zeros(shape=(X.shape[0], len(self.cat_cols))),
                           index=X.index,
                           columns=self.cat_cols)
        #concatenate X and y to simplify usage (temporary object)
        XY = self._getXY(X, y)
        
        # iterate over folds
        for split, (trn_idx, val_idx) in enumerate(self.cv_outer.split(X, y)):
            # training k-1 folds are not needed, as encoding is stored in fit() already
            val = XY.iloc[val_idx]
            # iterate over categorical features
            for f_ in self.cat_cols:
                te = self.gb[f_ + '_' + str(split)]
                # encode the OOF partion
                oof.iloc[val_idx, oof.columns.get_loc(f_)] = val[f_].map(te).astype(np.float32)
        # do finla cosmetics and fill NAN
        oof = oof.add_prefix(self.prefix).fillna(self.prior)
        del XY
        
        X_ = X.drop(self.cat_cols, axis=1)
        return pd.concat([X_, oof], axis=1)
        
    def fit(self, X, y=None, **fit_params):
        self = super(TargetEncoder_NestedKFold, self).fit(X, y, **fit_params)
        
        #concatenate X and y to simplify usage (temporary object)
        XY = self._getXY(X, y)
        
        # A dictionary of all available classes
        cat_classes = dict()
        for f_ in self.cat_cols:
            cat_classes[f_] = X[f_].unique().tolist()
        
        for outer_split, (outer_trn_idx, outer_val_idx) in enumerate(self.cv_outer.split(X, y)):
            outer_trn = XY.iloc[outer_trn_idx]
            # validation subsample is not used
            
            # The final encoding for each categorical variable: 
            # 1 pd.Series per each outer fold storing averaged over inner folds encodings
            for f_ in self.cat_cols:
                self.gb[f_ + '_' + str(outer_split)] = pd.Series(np.zeros((len(cat_classes[f_]),)), 
                                                                 index=cat_classes[f_], 
                                                                 dtype=np.float32)
                
            # Convert categoricals to 'object', as self.nan_tmp is not in categories and fillna does not work
            outer_trn[self.cat_cols] = outer_trn[self.cat_cols].astype('object')
            # Fill NaN as self.nan_tmp, as groupby does not group on them
            outer_trn.fillna(self.nan_tmp, inplace=True)
            
            # Create a new cross-validator with a different random state per fold
            self.cv_inner = KFold(n_splits=self.cv_inner_n,
                                            shuffle=True,
                                            random_state=self.random_state+outer_split)
            # The target mean for the outer k-1 folds
            outer_prior = outer_trn[self.y_name].mean()
            
            for inner_split, (inner_trn_idx, inner_val_idx) in enumerate(self.cv_inner.split(outer_trn.drop(self.y_name, axis=1), 
                                                                                           outer_trn[self.y_name])):
                inner_trn = outer_trn.iloc[inner_trn_idx]
                # validation subsample is not used
                # The target mean for the inner k-1 folds
                inner_prior = float(inner_trn[self.y_name].mean())
                
                for f_ in self.cat_cols:
                    # get target means for each class within category ofr the k-1 folds of the inner CV loop 
                    # also change back from the temporary NaN naming to np.nan
                    te = inner_trn.groupby(f_)[self.y_name].mean().astype(np.float32).rename(index={self.nan_tmp:np.nan})
                    # add the inner-loop encoding for averaging
                    self.gb[f_ + '_' + str(outer_split)] += te/self.cv_inner_n
                    # also add inner-fold prior for the missing categories
                    for miss_f_ in [f__ for f__ in cat_classes[f_] if f__ not in te]:
                        self.gb[f_ + '_' + str(outer_split)].loc[miss_f_] += inner_prior/self.cv_inner_n
        del XY
        return self


# ### Split the full sample into train/test (80/20)
# Here `StratifiedShuffleSplit` is used instead of the usual `train_test_split`. In practice it gives the same result, but in this particular case it is beneficial in a few ways:
# - it allows to obtain *indices* of entries in the train/test split, which is very handy as we want to use the same split, but to aply different data pre-processing (i.e. different encodings of categorical features);
# - it allows to easily switch to a CV scheme, if one wants to.

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
rs = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=3)


# ### Define the main function to apply preprocessing, build and evaluate a lightgbm model

# In[ ]:


import lightgbm as lgb
import category_encoders
import time
import xam
# lightgbm early stopping
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : None,
            'eval_names': ['valid'],
            'verbose': False,
            'categorical_feature': 'auto'}

def fit_model(data_, trn_idx_, val_idx_, feats_, cat_enc_type='OHE'):
    trn_x, trn_y = data_[feats_].iloc[trn_idx_], data_['TARGET'].iloc[trn_idx_]
    val_x, val_y = data_[feats_].iloc[val_idx_], data_['TARGET'].iloc[val_idx_]
    cols_cat = data_.select_dtypes(include=['category', 'object']).columns.tolist()
    #if 'ExpandingMean' in cat_enc_type:
    #        print(trn_x[cols_cat].head(10))
    
    time_start = time.time()
    if cat_enc_type == 'OHE':
        trn_x, val_x = cat_to_dummy(trn_x, val_x)
    elif cat_enc_type == 'LabelEnc':
        categorical_feats, trn_x, val_x = cat_to_int(trn_x, val_x)
    elif  'TargetEnc' in cat_enc_type:
        te = None
        if 'category_encoders' in cat_enc_type:
            te = category_encoders.TargetEncoder(cols=cols_cat, smoothing=1)
            # category_encoders.TargetEncoder can not handle 'category' with empty empty class
            # See https://github.com/scikit-learn-contrib/categorical-encoding/issues/86
            for df in [trn_x, val_x]:
                for f_ in cols_cat:
                    df[f_]  = df[f_].astype('object')
        elif 'xam_Smooth' in cat_enc_type:
            te = xam.feature_extraction.SmoothTargetEncoder(prior_weight=1, columns=cols_cat, suffix='')
        elif 'KFold' in cat_enc_type:
            te = TargetEncoder_KFold(cv=5, cat_cols=None, random_state=1)
        elif 'ExpandingMean' in cat_enc_type:
            te = TargetEncoder_ExpandingMean(cat_cols=None, random_state=1)
        elif 'NestedKFold' in cat_enc_type:
            te = TargetEncoder_NestedKFold(cv_inner=5, cv_outer=5, cat_cols=None, random_state=1)
        elif 'NoRegularisation' in cat_enc_type:
            te = TargetEncoder_Base(cat_cols=None, random_state=1)
        te = te.fit(trn_x, trn_y)
        # transform the training set either simply or with regularisation
        if 'NoRegularisation' in cat_enc_type or 'category_encoders' in cat_enc_type or 'xam' in cat_enc_type:
            trn_x = te.transform(trn_x)
        else:
            trn_x = te.regularise(trn_x, trn_y)
        # transform the validation set WITHOUT regularisation (also applies to test set, if one adds that)
        val_x = te.transform(val_x)
        
        # explicit casting of remaining categorical columns (a feature of xam implementation)
        if 'xam' in cat_enc_type:
            for df in [trn_x, val_x]:
                for c in cols_cat:
                    df[c] = df[c].astype(np.float32)
    elif 'LGBM_internal' in cat_enc_type:
        pass
    else:
        raise ValueError('Unknown cat_enc_type value: ' + str(cat_enc_type))
    # set the transformed validation set as evaluation sample in lightgbm
    fit_params['eval_set'] = [(trn_x, trn_y), (val_x, val_y)]
    fit_params['eval_names']= ['train', 'valid']
    
    # FIT A MODEL
    # Model parameters were tuned in this kernel: https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
    # n_estimators is set to a "large value". 
    # The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
    clf = lgb.LGBMClassifier(max_depth=-1, n_jobs=4, n_estimators=5000, learning_rate=0.1, random_state=314, silent=True, metric='None')
    opt_parameters = {'colsample_bytree': 0.9234, 'min_child_samples': 399, 'min_child_weight': 0.1, 'num_leaves': 13, 'reg_alpha': 2, 'reg_lambda': 5, 'subsample': 0.855}
    clf.set_params(**opt_parameters)
    clf.fit(trn_x, trn_y, **fit_params)
    score_val = clf.best_score_['valid']['auc']
    score_trn = clf.best_score_['train']['auc']
    itr_trn = clf.best_iteration_
    print('{}: {:f}'.format(cat_enc_type, score_val))
    time_end = time.time()
    
    # cleanup to reduce memory footstep
    del trn_x, trn_y, val_x, val_y
    del clf
    gc.collect()
    
    return score_val, score_trn, itr_trn, time_end-time_start


# ### Evaluate models with different categorical treatment

# In[ ]:


feats = [f for f in application_train.columns if f not in ['SK_ID_CURR', 'TARGET']]

te_types = ['OHE', 'LabelEnc', 'LGBM_internal',
            'TargetEnc_category_encoders', 'TargetEnc_xam_Smooth', 'TargetEnc_NoRegularisation', 
            'TargetEnc_KFold', 'TargetEnc_ExpandingMean',
            'TargetEnc_NestedKFold']
perf = {}

for trn_idx, val_idx in rs.split(application_train, application_train['TARGET']):    
    for te_ in te_types:
        auc = fit_model(application_train, trn_idx, val_idx, feats, cat_enc_type=te_)
        perf[te_] = auc


# ## Visualise comparison of model performance

# In[ ]:


df_perf = pd.DataFrame(perf, index=['auc_valid', 'auc_train', 'ntrees', 'fit_time'])


# In[ ]:


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,14))
ax = ax.flatten()
fig.subplots_adjust(wspace=1.50)
for i,v in enumerate([('auc_valid', 'Validation ROC AUC', 'Blues'),
                      ('auc_train', 'training ROC AUC', 'Oranges'),
                      ('ntrees', 'Optimal number of trees', 'Greens'),
                      ('fit_time', 'Fit time', 'Purples')
                     ]):
    sns.heatmap(df_perf.loc[v[0],:].to_frame(), cmap=v[2], annot=True, xticklabels=False,fmt='.4g', ax=ax[i])#, vmin=0.75, vmax=0.8
    ax[i].set_title(v[1])
fig.savefig('Performance.png')


# # Observations
# - The difference in the final model performance is small. Variation of the random seed in `StratifiedShuffleSplit` changes the outcome. However, on the full sample joining all tables together i see a consistent improvement of O(0.001) on TargetEncoding with regularisation over the internal LightGBM method;
# - The TargetEncoding methods lead to smaller number of trees that are needed to achieve the same performance, thus training time is reduced (important for model training on the full dataset)
# - Nested and simple KFold regularisation on the training sample  perform surprisingly identical. There might be something overlooked in the implementation.
# - `category_encoders.TargetEncoder` performs the same as the basing target encoding with no regularisation.

# In[ ]:




