#!/usr/bin/env python
# coding: utf-8

# ## TL;DR:
# 
# 1. Classic variant:
#    * Mem. usage decreased from 4867.46 Mb to 2452.37 Mb (49.6% reduction)
#    * Number of unique values: 3285120 -> 3216884 **(2.0% lost)**
# 2. My variant:
#    * Mem. usage decreased from 4847.46 Mb to 2515.03 Mb (48.1% reduction)
#    * Number of unique values: 3285120 -> 3285120 **(0.0% lost)**
# 3. My variant with optional object -> category conversion (read the **Objects -> categories** section before using!):
#    * Mem. usage decreased from 4847.46 Mb to **1086.85 Mb (77.6% reduction)**
#    * Number of unique values: 3285120 -> 3285120 **(0.0% lost)**
# 
# ## Rationale
# 
# It seems that many competition teams use the same `reduce_mem_usage` function (with modifications) e.g. https://www.kaggle.com/kyakovlev/ieee-data-minification .
# 
# Though I see a few major drawbacks in using it as is:
# 1. Such functions either:
#     1. Don't use minimal possible types for the sake of (imaginary) safety, and therefore use more memory than actually needed.
#     2. Use float16 but don't guarantee that you don't lose precision or unique values much (see **Issue 1** below)
# 2. None of them try to perform float to int conversion.
# 3. It's done only once and don't allow you to easily minify newly created features.
# 
# So my functons address all of these problems.
# They allow using really minimal amount of memory and guarantee not losing anything (precision, na values, unique values, etc.).
# And you can do minification on the fly for new columns: `df['a/b'] = sd(df['a']/df['b'])`.
# 
# Also my `sd` (stands for `safe downcast`) function is very flexible. If you consider you can allow to lose 0.1 precision when rounding but wanna save more memory, then no problem, just set `sd(col, max_loss_limit=0.1, avg_loss_limit=0.1)`.
# 
# ## Objects -> categories:
# 
# My functions can do object -> category conversion as well. But it's important to remember that if you do this for train and test separately they will have different internal representation (see **Issue 2** below) and may cause issues with ML algorithms if they mess with codes.
# In my code I use a concatenated dataset with 2-level indexes so it's not a problem. See the **Load dataset** section below.
# 
# If you want them to be converted, use `obj_to_cat=True` arg.
# In this case you'll get:
# * **Mem. usage decreased from 4847.46 Mb to 1086.85 Mb (77.6% reduction)**
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
import gc


# ## Issue 1

# In[ ]:


x = pd.Series([1111.5, 1111.9, 30001, 30007]) 
print(x)
print('nunique:', x.nunique(), '\n')

y = x.astype(np.float16)
print(y)
print('nunique:', y.nunique())


# ## Issue 2

# In[ ]:


x = pd.Series(['a', 'b', 'c'])
y = pd.Series(['a', 'c', 'a'])
print(x.astype('category').cat.codes)
print(y.astype('category').cat.codes)


# ## Functions

# In[ ]:


# safe downcast
def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
    """
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == 'float'
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ['float16', 'float32']

    if na_count <= na_loss_limit:
        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == 'int'):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
            return col_tmp

    # field can't be converted
    return col


def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)
        
        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')
        
        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if (na_count != new_na_count):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if (n_uniq != new_n_uniq):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df

# https://www.kaggle.com/kyakovlev/ieee-data-minification
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df


# In[ ]:


def get_stats(df):
    stats = pd.DataFrame(index=df.columns, columns=['na_count', 'n_unique', 'type', 'memory_usage'])
    for col in df.columns:
        stats.loc[col] = [df[col].isna().sum(), df[col].nunique(dropna=False), df[col].dtypes, df[col].memory_usage(deep=True, index=False) / 1024**2]
    stats.loc['Overall'] = [stats['na_count'].sum(), stats['n_unique'].sum(), None, df.memory_usage(deep=True).sum() / 1024**2]
    return stats

def print_header():
    print('col         conversion        dtype    na    uniq  size')
    print()
    
def print_values(name, conversion, col):
    template = '{:10}  {:16}  {:>7}  {:2}  {:6}  {:1.2f}MB'
    print(template.format(name, conversion, str(col.dtypes), col.isna().sum(), col.nunique(dropna=False), col.memory_usage(deep=True, index=False) / 1024 ** 2))


# ## Load dataset
# 
# The dataset is prepared in https://www.kaggle.com/alexeykupershtokh/concat-dataframes

# In[ ]:


tmp = pd.read_pickle('/kaggle/input/concat-dataframes/concat.pkl')


# In[ ]:


tmp.sample(20).sort_index()


# In[ ]:


tmp.loc['test'].sample(10).sort_index()


# In[ ]:


# cache a mini-dataset for examples
example = tmp[['card1', 'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14']].copy()


# ## Minify

# In[ ]:


stats = get_stats(tmp)
stats


# In[ ]:


tmp1 = reduce_mem_usage_sd(tmp.copy(), verbose=True)


# In[ ]:


stats1 = get_stats(tmp1)


# In[ ]:


tmp1.to_pickle('safe_memreduced_2.5gb.pkl')


# In[ ]:


tmp1 = reduce_mem_usage_sd(tmp1, verbose=True, obj_to_cat=True)


# In[ ]:


tmp1.to_pickle('safe_memreduced_1gb.pkl')


# In[ ]:


# don't copy, as the original df is not needed anymore
tmp2 = reduce_mem_usage(tmp, verbose=True)


# In[ ]:


overall_stats = pd.concat([stats.add_prefix('a_'), stats1.add_prefix('b_'), get_stats(tmp1).add_prefix('c_')], axis=1)


# In[ ]:


overall_stats


# ## Example 1: float conversion params

# Let's try to create a new feature Series

# In[ ]:


new_feature = (example.groupby('card1')['TransactionAmt'].transform('mean'))

print_header()
print_values('mean_amt', 'original', new_feature)


# Let's try to minify it with default settings

# In[ ]:


new_feature2 = sd(new_feature)

print_header()
print_values('mean_amt', 'default sd():', new_feature2)


# Oops, it didn't work. The reason is most likely in that the values are too dense (e.g. there could be values like 100.0001 and 100.0002). But as far as this feature is ordinal and we don't care about preserving all of the unique values, let's losen our minification rules.

# In[ ]:


new_feature3 = sd(new_feature, n_uniq_loss_limit=100)

print_header()
print_values('mean_amt', 'allow uniq loss:', new_feature3)


# You can see that we lost `11957 - 11882 = 75` unique values but saved 50% of memory.

# ## Example 2: automatic float to int conversion

# let's try frequency encoding

# In[ ]:


new_feature = (example.groupby('card1')['TransactionAmt'].transform('nunique'))
new_feature2 = sd(new_feature)

print_header()
print_values('nunique', 'original', new_feature)
print_values('nunique', 'default sd():', new_feature2)


# ## Example 3: C1-C14 column compression (lossy for 3 rows)

# In[ ]:


print_header()
for col in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14']:
    f = example[col]
    print_values(col, 'original', f)
    
    # here we try to use default setting (precise enough)
    f = sd(f)
    print_values(col, 'default sd():', f)

    # here we allow to fill up to 3 na fields with -99
    f = sd(f, na_loss_limit=3, fillna=-99)
    print_values(col, 'limited na loss:', f)
    print()


# In[ ]:




