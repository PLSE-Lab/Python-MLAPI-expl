#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = '../input/avito-demand-prediction/'
textdata_path = '../input/adp-prepare-kfold-text/textdata.csv'
target_col = 'deal_probability'
os.listdir(DATA_DIR)


# In[20]:


usecols = ['user_id', #'item_id',
           'region', 'city', 'parent_category_name', 'category_name', 
           'param_1', 'param_2', 'param_3', 
           'activation_date',
           'title', 'description', 
           'price', 'item_seq_number', 
           'user_type', 
           'image_top_1', 'image']
eval_sets = pd.read_csv(textdata_path, usecols=['eval_set'])['eval_set'].values
train_num = (eval_sets!=10).sum()
eval_sets = eval_sets[:train_num]
train = pd.read_csv(DATA_DIR+'train.csv', usecols=usecols+[target_col])
test = pd.read_csv(DATA_DIR+'test.csv', usecols=usecols)


# In[21]:


def get_dow(df):
    f = lambda x:pd.to_datetime(x).dayofweek
    unq = df['activation_date'].unique().tolist()
    d = dict([u, f(u)] for u in unq)
    df['dow'] = df['activation_date'].map(d.get)
    return df
train = get_dow(train)
test = get_dow(test)
del train['activation_date'], test['activation_date']; gc.collect()


# In[22]:


len(set(train['user_id'].values.tolist()) & set(test['user_id'].values.tolist())), len(set(test['user_id'].values.tolist()))


# In[23]:


common_indexes = set(train['user_id'].values.tolist()) & set(test['user_id'].values.tolist())
common_indexes = list(common_indexes)
train['user_common'] = 0
test['user_common'] = 0
train = train.set_index('user_id')
test = test.set_index('user_id')
train.loc[common_indexes, 'user_common'] = 1
test.loc[common_indexes, 'user_common'] = 1
train = train.reset_index()
test = test.reset_index()
del common_indexes; gc.collect()


# In[24]:


train['user_id_common'] = train['user_id'].values
train.loc[train['user_common']==0, 'user_id_common'] = 'unknown'
test['user_id_common'] = test['user_id'].values
test.loc[test['user_common']==0, 'user_id_common'] = 'unknown'


# In[25]:


train.head(3).T


# In[26]:


train_num == len(train)


# In[27]:


y = train[target_col].values
del train[target_col]; gc.collect()
train_num = len(train)
df = pd.concat([train, test], ignore_index=True)
del train, test; gc.collect()


# In[28]:


df['image'].isnull().sum()


# In[29]:


df['image'] = (~df['image'].isnull()).astype('int8')


# In[30]:


del df['user_common']; gc.collect();


# In[31]:


df['image_top_1'].isnull().sum()


# In[32]:


df['image_top_1'].min(), df['image_top_1'].max()


# In[33]:


df.head(3).T


# In[34]:


df['price_bin'] = pd.cut(np.log1p(df['price']), 256, labels=np.arange(256))
df['price_bin'] = df['price_bin'].astype('float').fillna(-1)
df['price_bin'] = df['price_bin'].astype('int')


# In[35]:


df['item_seq_bin'] = pd.cut(np.log1p(df['item_seq_number']), 512, labels=np.arange(512))
df['item_seq_bin'] = df['item_seq_bin'].astype('float').fillna(-1)
df['item_seq_bin'] = df['item_seq_bin'].astype('int')


# In[36]:


enc_cols = ['user_id', 'user_id_common',
            'region', 'city', 'parent_category_name', 'category_name', 
            'param_1', 'param_2', 'param_3', 
            'user_type', #'price_bin',
            'image_top_1']

enc_dict = {}
for i, c in enumerate(enc_cols):
    print('label encoding', i, c)
    values, names = pd.factorize(df[c].fillna('unknown'))
    df[c] = values
    enc_dict[c] = pd.DataFrame(names.values, columns=['lbe'])
    #enc_dict[c].to_csv(c+'_enc.csv', index=False)


# In[37]:


df.head(3).T


# In[38]:


del df['title'], df['description']; gc.collect();
df.info()


# In[39]:


def reduce_memory(df):
    for c in df.columns:
        if df[c].dtype=='int':
            if df[c].min()<0:
                if df[c].abs().max()<2**7:
                    df[c] = df[c].astype('int8')
                elif df[c].abs().max()<2**15:
                    df[c] = df[c].astype('int16')
                elif df[c].abs().max()<2**31:
                    df[c] = df[c].astype('int32')
                else:
                    continue
            else:
                if df[c].max()<2**8:
                    df[c] = df[c].astype('uint8')
                elif df[c].max()<2**16:
                    df[c] = df[c].astype('uint16')
                elif df[c].max()<2**32:
                    df[c] = df[c].astype('uint32')
                else:
                    continue
    return df
df = reduce_memory(df)
print(df.info())


# In[41]:


cols = ['user_id',
        'user_id_common',
        'region',
        'city',
        'parent_category_name',
        'category_name',
        'param_1',
        'param_2',
        'param_3',
        'price', 'price_bin',
        'item_seq_number', 'item_seq_bin',
        'user_type',
        'image',
        'image_top_1',
        'dow']
df = df[cols]


# In[42]:


#df.to_csv('data_lbe.csv', index=False)


# In[43]:


df.shape


# In[44]:


train = df[:train_num].reset_index(drop=True)
test  = df[train_num:].reset_index(drop=True)
train[target_col] = y
train.head(2).T


# In[45]:


ts_folds = pd.read_csv(DATA_DIR+'train.csv', usecols=['activation_date'])
ts_folds.groupby('activation_date').size()


# In[46]:


ts_folds = ts_folds['activation_date'].values
ts_folds[ts_folds>'2017-03-28'] = '2017-03-28'


# In[47]:


pd.DataFrame(ts_folds).groupby(0).size()


# In[48]:


ts_valid_fold_list = sorted(np.unique(ts_folds))
ts_valid_fold_list


# In[49]:


# valid_fold = 0
# mask_val = eval_sets==valid_fold
# mask_tr  = ~mask_val
cross_valid_fold_list = list(range(10))
cross_valid_fold_list


# In[50]:


## https://zhuanlan.zhihu.com/p/26308272
class MeanEncoder(object):
    def __init__(self, 
                 categorical_features,
                 is_class_target, 
                 cross_valid_set, 
                 timeseries_valid_set=None, 
                 only_return_me_features=True,
                 verbose=True,
                 prior_weight_func=None):
        """MeanEncoder Class
        :param categorical_features: list of str, the name of the categorical columns to encode
        :param is_class_target: True for {classification} False for {regression}
        :param *_valid_set: 'tuple'-> ('list': valid_fold_names, 'list/np.array/series': valid_fold_values)
        :param prior_weight_func:
            a function that takes in the number of observations, and outputs prior weight
            when a dict is passed, the default exponential decay function will be used:
            k: the number of observations needed for the posterior to be weighted equally as the prior
            f: larger f --> smaller slope
        """
        self.categorical_features = categorical_features
        if is_class_target:
            self.target_values = []
        else:
            self.target_values = None
        self.is_class_target = is_class_target
        self.verbose = verbose
        self.only_return_me_features = only_return_me_features
        self.learned_stats = {}
        
        assert isinstance(cross_valid_set[0], list)
        if isinstance(cross_valid_set[1], list):
            cross_valid_set[1] = np.array(cross_valid_set[1]).copy()
        elif hasattr(cross_valid_set[1], 'values'):
            cross_valid_set[1] = cross_valid_set[1].values.copy()
        self.cv_names, self.cv_values = cross_valid_set
        
        if timeseries_valid_set is not None:
            assert isinstance(timeseries_valid_set[0], list)
            if isinstance(timeseries_valid_set[1], list):
                timeseries_valid_set[1] = np.array(timeseries_valid_set[1]).copy()
            elif hasattr(timeseries_valid_set[1], 'values'):
                timeseries_valid_set[1] = timeseries_valid_set[1].values.copy()
            self.tscv_names, self.tscv_values = timeseries_valid_set
        else:
            self.tscv_names, self.tscv_values = None, None
        
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', 
                                          dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            print('Set prior_weight_func to default: k=2, f=1')
            print('   lambda x: 1 / (1 + np.exp((x - 2) / 1))')
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))
        
    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_fn):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()
        col_avg_y = X_train.groupby(
            by=variable, sort=False, axis=0
        )['pred_temp'].agg(
            {'mean': 'mean', 'beta': 'size'}
        )
        col_avg_y['beta'] = prior_fn(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1-col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_valid = X_test.join(col_avg_y, on=variable).fillna(prior)[nf_name].values
        return nf_train, nf_valid, prior, col_avg_y
    
    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, (n_samples * n_features)
        :param y: pandas Series or numpy array: (n_samples,)
        return
        X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        new_feat_names = []
        if self.is_class_target:
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): []                                   for variable, target in product(
                                      self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                new_feat_names.append(nf_name)
                X_new.loc[:, nf_name] = np.nan
                if self.verbose:
                    gen = tqdm(enumerate(self.cv_names), ascii=True, desc=nf_name, total=len(self.cv_names))
                else:
                    gen = enumerate(self.cv_names)
                for cv_idx, cv_name in gen:
                    mask_valid = self.cv_values==cv_idx
                    mask_train = self.cv_values!=cv_idx
                    if self.tscv_names is None:
                        #print('cv', np.sum(mask_valid), np.sum(mask_train)) #debug
                        nf_train, nf_valid, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                            X_new.iloc[mask_train], y[mask_train], X_new.iloc[mask_valid], 
                            variable, target, self.prior_weight_func)
                        X_new.loc[mask_valid, nf_name] = nf_valid
                        self.learned_stats[nf_name].append((prior, col_avg_y))
                    else:
                        mask_expanding = False
                        tsgen = enumerate(self.tscv_names)
                        for tscv_idx, tscv_name in tsgen:
                            mask_current = self.tscv_values==tscv_name
                            mask_valid = (self.cv_values==cv_idx) & mask_current
                            mask_expanding |= mask_current
                            mask_train = (self.cv_values!=cv_idx) & mask_expanding
                            #print('cv+ts', np.sum(mask_valid), np.sum(mask_train)) #debug
                            nf_train, nf_valid, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                                X_new.iloc[mask_train], y[mask_train], X_new.iloc[mask_valid], 
                                variable, target, self.prior_weight_func)
                            X_new.loc[mask_valid, nf_name] = nf_valid
                        ####
                        self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                new_feat_names.append(nf_name)
                X_new.loc[:, nf_name] = np.nan
                if self.verbose:
                    gen = tqdm(enumerate(self.cv_names), ascii=True, desc=nf_name, total=len(self.cv_names))
                else:
                    gen = enumerate(self.cv_names)
                for cv_idx, cv_name in gen:
                    mask_valid = self.cv_values==cv_idx
                    mask_train = self.cv_values!=cv_idx
                    if self.tscv_names is None:
                        #print('cv', np.sum(mask_valid), np.sum(mask_train)) #debug
                        nf_train, nf_valid, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                            X_new.iloc[mask_train], y[mask_train], X_new.iloc[mask_valid], 
                            variable, None, self.prior_weight_func)
                        X_new.loc[mask_valid, nf_name] = nf_valid
                        self.learned_stats[nf_name].append((prior, col_avg_y))
                    else:
                        mask_expanding = False
                        tsgen = enumerate(self.tscv_names)
                        for tscv_idx, tscv_name in tsgen:
                            mask_current = self.tscv_values==tscv_name
                            mask_valid = (self.cv_values==cv_idx) & mask_current
                            mask_expanding |= mask_current
                            mask_train = (self.cv_values!=cv_idx) & mask_expanding
                            #print('cv+ts', np.sum(mask_valid), np.sum(mask_train)) #debug
                            nf_train, nf_valid, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                                X_new.iloc[mask_train], y[mask_train], X_new.iloc[mask_valid], 
                                variable, None, self.prior_weight_func)
                            X_new.loc[mask_valid, nf_name] = nf_valid
                        ####
                        self.learned_stats[nf_name].append((prior, col_avg_y))
        if self.only_return_me_features:
            X_new = X_new[new_feat_names]
        return X_new
    
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        return 
        :param X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        new_feat_names = []
        if self.is_class_target:
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                new_feat_names.append(nf_name)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior)[
                        nf_name]
                X_new[nf_name] /= len(self.cv_names)
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                new_feat_names.append(nf_name)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior)[
                        nf_name]
                X_new[nf_name] /= len(self.cv_names)
        if self.only_return_me_features:
            X_new = X_new[new_feat_names]
        return X_new


# In[51]:


# For Target Encoding
cat_cols = ['user_id',
            'user_id_common',
            'region',
            'city',
            'parent_category_name',
            'category_name',
            'param_1',
            'param_2',
            'param_3',
            'price_bin', #'price',
            'item_seq_number', 'item_seq_bin',
            'user_type',
            'image',
            'image_top_1',
            'dow']


# In[ ]:


def get_me_feats(k, f, with_ts=False, save=True):
    cv_set = (cross_valid_fold_list, eval_sets)
    ts_set = (ts_valid_fold_list, ts_folds) if with_ts else None
    prior_weight_func = dict(k=k, f=f)
    me = MeanEncoder(cat_cols, False, 
                     cv_set, 
                     ts_set, 
                     prior_weight_func=prior_weight_func)
    me_tr = me.fit_transform(train, y).fillna(-1).astype('float32')
    me_te = me.transform(test).fillna(-1).astype('float32')
    if save:
        fprefix = 'cvts' if with_ts else 'cv'
        fprefix = '_me_'+fprefix+'_k_{}_f_{}.csv'.format(k, f)
        me_tr.to_csv('train'+fprefix, index=False)
        me_te.to_csv('test'+fprefix, index=False)
    return me_tr, me_te


# In[ ]:


get_ipython().run_cell_magic('time', '', 'settings = [\n            (20, 2, True)\n           ]\nfor k, f, with_ts in settings:\n    me_tr, me_te = get_me_feats(k, f, with_ts)')

