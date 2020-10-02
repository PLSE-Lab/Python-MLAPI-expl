#!/usr/bin/env python
# coding: utf-8

# # LGBM Regression with assessment attempt features
# 
# This kernel is based on https://www.kaggle.com/artgor/quick-and-dirty-regression.

# In[ ]:


get_ipython().run_line_magic('ls', '-lh ../input/data-science-bowl-2019')


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import IPython
import gc


pd.set_option('display.max_columns', None)
sns.set()


# In[ ]:


def display(*dfs, head=True):
    """
    Display multiple dataframes
    """
    for df in dfs:
        IPython.display.display(df.head() if head else df)


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numeric_dtypes = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    mem_before = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        dtype = df[col].dtypes

        if dtype in numeric_dtypes:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(dtype)[:3] == 'int':
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
                    df[col] = df[col].astype(np.float132)
                else:
                    df[col] = df[col].astype(np.float64)

    mem_after = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(mem_after, 100 * (mem_before - mem_after) / mem_before))


# # Load data

# In[ ]:


DATA_DIR = '../input/data-science-bowl-2019'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
train = train[train['installation_id'].isin(labels['installation_id'].unique())]

test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
title_pred = test.groupby('installation_id').last().reset_index()[['installation_id', 'title']]
sbm = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

reduce_mem_usage(train)
reduce_mem_usage(test)

display(
    train,
    labels,
    test,
    labels,
    title_pred,
    sbm,
)


# In[ ]:


assert title_pred['installation_id'].equals(sbm['installation_id'])


# # Filter assessments

# In[ ]:


def filter_assessments(df):
    is_assessment = (
        (df['title'].eq('Bird Measurer (Assessment)') & df['event_code'].eq(4110)) |
        (~df['title'].eq('Bird Measurer (Assessment)') & df['event_code'].eq(4100)) &
        df['type'].eq('Assessment')
    )
    return df[is_assessment].reset_index(drop=True)


# # Extract attempt results

# In[ ]:


def extract_attempt(df):
    """
    Extract attempt result as boolean (true: correct, false: incorrect)
    """
    correct = df['event_data'].str.extract(r'"correct":([^,]+)', expand=False).eq('true').astype(int)
    return df.assign(correct=correct)


# # Calculate attempt stats

# In[ ]:


def with_name(func, name):
    func.__name__ = name
    return func


def accuracy_group(acc):
    if acc == 0:
        return 0
    elif acc == 1:
        return 3
    elif acc == 0.5:
        return 2
    else:
        return 1


def calc_attempt_stats(df):
    aggs = {
        'correct': [
            with_name(lambda s: (s == 1).sum(), 'num_correct'),
            with_name(lambda s: (s == 0).sum(), 'num_incorrect'),
            with_name(lambda s: s.size, 'attempts'),
            with_name(lambda s: s.mean(), 'accuracy'),
        ],

        'timestamp': [
            with_name(lambda s: s.iloc[-1], 'timestamp'),
        ],
    }
    
    # apply aggregation
    by = ['installation_id', 'title', 'game_session']
    stats = df.groupby(by, sort=False).agg(aggs).reset_index()

    # flatten multi-level columns
    stats.columns = [col[1] if (col[1] != '') else col[0] for col in stats.columns]

    # add accuracy group
    stats = stats.assign(accuracy_group=stats['accuracy'].map(accuracy_group).astype(np.int8))

    return stats


# # Expand attempt stats

# In[ ]:


from functools import reduce
from sklearn.preprocessing import OneHotEncoder


def add_prefix(l, prefix, sep='_'):
    """
    Add prefix to list of strings
    """
    return [prefix + sep + x for x in l]


def concat_dataframes(dfs, axis):
    """
    Concat arbitrary number of dataframes
    """
    return reduce(lambda l, r: pd.concat([l, r], axis=axis), dfs)


def expand_stats(df):
    """
    Calculate product of assessment stats and one-hot encoded title vector
    
    Input DataFrame:
            num_correct
    game_A            1 
    game_B            2
    
    Output DataFrame:
            game_A_num_correct  game_B_num_correct
    game_A                   1                   0                   
    game_B                   0                   2   
    """
    col = 'title'
    enc = OneHotEncoder().fit(df[[col]])
    enc_cols = enc.categories_[0]
    one_hot = enc.transform(df[[col]]).toarray().astype(np.int8) 

    dfs = [df]
    cols = ['num_correct', 'num_incorrect', 'attempts', 'accuracy', 'accuracy_group']

    for col in cols:
        prod = pd.DataFrame(df[[col]].values * one_hot,
                            columns=add_prefix(enc_cols, col))
        dfs.append(prod)
    
    return concat_dataframes(dfs, axis=1)


# # Calculate cumulative features

# In[ ]:


import re


def filter_cols_startswith(cols, s):
    return [c for c in cols if c.startswith(s)]


def calc_cum(df, is_test=False):

    def process_gdf(df):
        funcs = {
            'cumsum': ['num_correct', 'num_incorrect', 'attempts'],
            'cummean': ['accuracy'],  # note that this contains accuracy_group
        }

        dfs = []
        drop_cols = []
        for func, patterns in funcs.items():
            for pat in patterns:
                cols = filter_cols_startswith(df.columns, pat)
                drop_cols += cols

                # for test, it's not necessary to shift rows
                periods = int(not is_test)

                if func == 'cumsum':
                    cum = df[cols].cumsum().shift(periods)
                elif func == 'cummean':
                    cum = df[cols].expanding().mean().shift(periods)

                cum.columns = add_prefix(cols, func)
                dfs.append(cum)

        # keep accuracy_group for training
        drop_cols.remove('accuracy_group')

        return concat_dataframes([df.drop(drop_cols, axis=1)] + dfs, axis=1)
    
    return df.groupby('installation_id', sort=False).apply(process_gdf)


# # Assessment game stats

# In[ ]:


game_stats = labels.groupby('title').agg({
    'num_correct': with_name(lambda s: s.sum(), 'num_correct_sum'),
    'num_incorrect': with_name(lambda s: s.sum(), 'num_incorrect_sum'),
    'accuracy': with_name(lambda s: s.mean(), 'avg_acc'),
    'accuracy_group': [
        with_name(lambda s: s.mean(), 'avg_acg'),
        with_name(lambda s: s.value_counts().index[0], 'most_freq_acg'),
    ]
}).reset_index()

game_stats.columns = [col[1] if (col[1] != '') else col[0] for col in game_stats.columns]
game_stats


# # Apply functions

# In[ ]:


def apply_funcs(df, funcs, debug=True):
    applied = df
    for func in funcs:
        applied = func(applied)
        
        if debug:
            print(func.__name__, applied.shape)
            display(applied)

    return applied


# In[ ]:


# functions to apply on both train and test
funcs_common = [
    filter_assessments,
    extract_attempt,
    calc_attempt_stats,
    expand_stats,
]


# In[ ]:


funcs_train = [
    lambda df: calc_cum(df, is_test=False),
    lambda df: pd.merge(df, game_stats, on='title', how='left'),
]

final_train = apply_funcs(train, funcs_common + funcs_train)

del train
gc.collect()


# In[ ]:


funcs_test = [
    lambda df: calc_cum(df, is_test=True),
    lambda df: df.groupby('installation_id', sort=False).last().reset_index(),
    lambda df: pd.merge(title_pred, df.drop('title', axis=1), on='installation_id', how='left'),
    lambda df: pd.merge(df, game_stats, on='title', how='left'),
]

final_test = apply_funcs(test, funcs_common + funcs_test)

del test
gc.collect()


# In[ ]:


final_train


# # Prepare training data

# In[ ]:


# train
ins_ids_train = final_train['installation_id']  # keep installation_id for group k fold
X_train = final_train.select_dtypes('number').drop('accuracy_group', axis=1)
y_train = final_train['accuracy_group']

# test
ins_ids_test = final_test['installation_id']
X_test = final_test.select_dtypes('number').drop('accuracy_group', axis=1)


del final_train, final_test
gc.collect()

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)


# In[ ]:


assert X_train.columns.tolist() == X_test.columns.tolist()
assert sbm['installation_id'].equals(ins_ids_test)


# # Train model

# In[ ]:


bst_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'subsample': 0.75,
    'subsample_freq': 1,
    'learning_rate': 0.04,
    'feature_fraction': 0.9,
    'max_depth': 15,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'random_state': 42,
}

fit_params = {
    'num_boost_round': 10000,
    'verbose_eval': 100,
    'early_stopping_rounds': 100,
}


# In[ ]:


from numba import jit
from functools import partial
import scipy as sp


@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        return -qwk(y, X_p)

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

    def coefficients(self):
        return self.coef_['x']


def div_by_sum(x):
    return x / x.sum()


def print_divider(text):
    print('\n---------- {} ----------\n'.format(text))


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

fold = GroupKFold(n_splits=5)

fi_split = np.zeros(X_train.shape[1])
fi_gain = np.zeros(X_train.shape[1])
oof_pred = np.zeros(len(X_train))
pred_test = np.zeros(len(X_test))
coff_avg = np.zeros(3)


for fold_idx, (idx_trn, idx_val) in enumerate(fold.split(X_train, y_train, ins_ids_train)):
    print_divider(f'Fold: {fold_idx}')
    X_trn, X_val = X_train.iloc[idx_trn], X_train.iloc[idx_val]
    y_trn, y_val = y_train[idx_trn], y_train[idx_val]

    d_trn = lgb.Dataset(X_trn, y_trn)
    d_val = lgb.Dataset(X_val, y_val)

    model = lgb.train(bst_params, d_trn,
                      valid_sets=[d_trn, d_val],
                      **fit_params)

    fi_split += div_by_sum(model.feature_importance(importance_type='split')) / fold.n_splits
    fi_gain += div_by_sum(model.feature_importance(importance_type='gain')) / fold.n_splits

    pred_val = model.predict(X_val)
    pred_train = model.predict(X_trn)
    oof_pred[idx_val] = pred_val
    pred_test += model.predict(X_test) / fold.n_splits

    optr = OptimizedRounder()
    optr.fit(pred_train, y_trn)
    coff_avg += optr.coefficients() / fold.n_splits
    print('\nround coefficients:', optr.coefficients())
    
    del X_trn, y_trn, X_val, y_val
    gc.collect()


# In[ ]:


coff_avg


# # Feature Importance

# In[ ]:


def plot_feature_importance(features, fi, fi_type, limit=30):
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = np.argsort(fi)[-limit:]
    y = np.arange(len(idxs))
    ax.barh(y, fi[idxs], align='center', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(features[idxs])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance: {fi_type}')


# In[ ]:


features = np.array(model.feature_name())
plot_feature_importance(features, fi_split, 'split')
plot_feature_importance(features, fi_gain, 'gain')


# # Round prediction

# In[ ]:


oof_pred_round = optr.predict(oof_pred, coff_avg)
qwk(y_train, oof_pred_round)


# In[ ]:


pred_round = optr.predict(pred_test, coff_avg)
pred_round[:10]


# In[ ]:


sbm['accuracy_group'] = pred_round.astype(int)
sbm['accuracy_group'].value_counts(normalize=True)


# In[ ]:


assert sbm['accuracy_group'].notnull().all()
assert sbm['accuracy_group'].isin([0, 1, 2, 3]).all()


# ## Save submission

# In[ ]:


sbm.to_csv('submission.csv', index=False)


# In[ ]:




