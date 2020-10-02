#!/usr/bin/env python
# coding: utf-8

# This notebooks is inspired by 2nd Lecture of Fast.ai's ML course. In this notebook, we will be talking about some of the best practices (of removing features) which can save you alot of computation and at the same time increase your score. In particular:
# - Features importance
# - Removing redundant features
# - Removing temporal features
# 
# For all of these topics we will discuss the theory, code and interpretation as well. Hope you all will enjoy it. This is my second kernel, so please comment and let me know the things that you liked and things that you want me to improve. I would love you hear from you.
# 
# > **Note:** I highly recommend going through the [Part-1](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-1) as this notebook is basically the continuation of it. Alot of code written here is directly taken from Part-1.

# In[ ]:


import pandas as pd
import numpy as np
import re
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `fldname` of `df`."
    fld = df[fldname]
    fld_dtype = fld.dtype
    
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtypeType):
        fld_dtype = np.datetime64
        
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
         
    prefix = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: 
        df[prefix + n] = getattr(fld.dt, n.lower())
    df[prefix + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


def fix_missing(df, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing."""
    for name,col in df.items():
        if is_numeric_dtype(col):
            if pd.isnull(col).sum():
                df[name+'_na'] = pd.isnull(col)
                filler = na_dict[name] if name in na_dict else col.median()
                df[name] = col.fillna(filler)
                na_dict[name] = filler
    return na_dict


# In[ ]:


def numericalize(df, max_cat):
    """ Changes the column col from a categorical type to it's integer codes."""
    for name, col in df.items():
        if hasattr(col, 'cat') and (max_cat is None or len(col.cat.categories)>max_cat):
            df[name] = col.cat.codes+1
            
def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement."""
#     idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[-n:].copy()


# In[ ]:


def process_df(df_raw,y_fld=None, subset=None, na_dict={}, max_cat=None,):
    if subset: df = get_sample(df_raw,subset)
    else: df = df_raw.copy()
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
    df.drop(y_fld, axis=1, inplace=True)
    
    # Missing continuous values
    na_dict = fix_missing(df, na_dict)
    
    # Normalizing continuous variables
    means, stds = {}, {}
    for name,col in df.items():
        if is_numeric_dtype(col) and col.dtype not in ['bool', 'object']:
            means[name], stds[name] = col.mean(), col.std()
            df[name] = (col-means[name])/stds[name] 
    
    # categorical variables
    categorical = []
    for col in df.columns:
        if df[col].dtype == 'object' : categorical.append(col)  # pandas treat "str" as "object"
    for col in categorical: 
        df[col] = df[col].astype("category").cat.as_ordered()
        
    # converting categorical variables to integer codes.
    numericalize(df, max_cat) # features with cardinality more than "max_cat".
    
    df = pd.get_dummies(df, dummy_na=True) # one-hot encoding for features with cardinality lower than "max_cat".
    
    return df, y#, na_dict, means, stds


# In[ ]:


df_org = pd.read_csv('../input/train/Train.csv', low_memory=False); df_org.head()


# In[ ]:


df_org.SalePrice = np.log(df_org.SalePrice)


# In[ ]:


df = df_org.copy()


# In[ ]:


add_datepart(df, 'saledate')


# In[ ]:


df, y = process_df(df, 'SalePrice'); df.head().T


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# Everything until this point in the notebook was covered in the [previous notebook](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-1). Again, if you have not read the [Part-1](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-1), please go through it before moving any further.

# ### Feature Importance
# It's not normally enough to just to know that a model can make accurate predictions - we also want to know how it's making predictions. The most important way to see this is with feature importance.

# In[ ]:


def rf_feature_importance(m,df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[ ]:


fi = rf_feature_importance(m, X_train); fi[:10]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:30]);


# In[ ]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep) # taking only the important features


# In[ ]:


df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m) # removing less important features not just saved the computation but also gave better scores.


# In[ ]:


fi = rf_feature_importance(m, df_keep)
plot_fi(fi);


# # Removing redundant features

# One thing that makes this harder to interpret is that there seem to be some variables with very similar meanings. Let's try to remove redundent features.

# In[ ]:


from scipy.cluster import hierarchy as hc
import scipy


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# Let's try removing some of these related features to see if the model can be simplified without impacting the accuracy.

# In[ ]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# Here's our baseline.

# In[ ]:


get_oob(df_keep)


# Now we try removing each variable one at a time.

# In[ ]:


for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# It looks like we can try one from each group for removal. Let's see what that does.

# In[ ]:


to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))


# Looking good! Let's use this dataframe from here. We'll save the list of columns so we can reuse it later.

# In[ ]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


keep_cols = df_keep.columns
df_keep = df[keep_cols]


# And let's see how this model looks on the full dataset.

# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# # Extrapolation
# 
# Removing temporal features to better generalize the model for unseen data points from different timestamp.

# In[ ]:


df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y = process_df(df_ext, 'is_valid')


# Lets make a classifier which will classify training and validation sample

# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# okay! it means you can classify training and validation samples with 100% accuracy. Clearly, There are some very prominent features which makes it so easy and obvious that something is a training sample or a validation sample.

# In[ ]:


fi = rf_feature_importance(m, x); fi[:10]


# In[ ]:


feats=['SalesID', 'saleElapsed', 'MachineID']


# In[ ]:


(X_train[feats]/1000).describe()


# In[ ]:


(X_valid[feats]/1000).describe()


# And since we are not using any randomness to split our data, the differences between **mean** and **std** of [SalesID, saleElapsed, MachineID] in validation and train set is huge because they are from completely different time stamps. These differences make sure that we easily distinguish between trainning and validation sample, hence the high oob_score.
# 
# If we get rid of these features i.e we are getting rid of features that differentiate training and validation sample. Removing such features makes training and validation samples less distinct. Making sure that the model will generalize better by learning **actually** important features rather than learning from features like [SalesID, saleElapsed, MachineID] which introduced temporal information (make it difficult for the model to extrapolate).
# 
# Tree model find it very difficult to extraplolate. Intuitionaly, random forests are taking average of the nearest neighbors to make predictions. If we have temporal features or information in out training data; the model will leverage this temporal information, giving you better scores in validation set. But during the test set, the new data point belongs to different time stamp (completely new, something that model has not seen before), now the model dont have any neighbors to compare, to take average. The model tries it's best, ends up making a false prediction, giving you lower test scores. 
# 
# So, removing temporal features can help random forests to extrapolate (to some extend).

# In[ ]:


x.drop(feats, axis=1, inplace=True)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[ ]:


fi = rf_feature_importance(m, x); fi[:10]


# In[ ]:


feats=['SalesID', 'saleElapsed', 'MachineID', 'state', 'saleDay', 'saleDayofyear']


# Until this point, we worked with the df_ext and RF classifier.
# ___

# In[ ]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)


# In[ ]:


df_subs = df_keep.drop(['SalesID', 'MachineID'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


plot_fi(rf_feature_importance(m, X_train));


# In[ ]:


# np.save('tmp/subs_cols.npy', np.array(df_subs.columns)) # save the final columsn list


# # Tree interpreter
# Interpretation of the model is a very rare practice, I don't know why? It gives you a very clear idea of why your tree is making certain prediction, how the values change internally and how much is each feature contributing to the final prediction. If you have some domain knowledge you can better interprete the why or how is certain thing making an impact. 
# > Machine Learing + Domanin Knowledge = Great Results

# In[ ]:


from treeinterpreter import treeinterpreter as ti


# In[ ]:


df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)


# In[ ]:


row = X_valid.values[None,0]; row


# In[ ]:


prediction, bias, contributions = ti.predict(m, row)


# In[ ]:


prediction[0], bias[0]


# In[ ]:


idxs = np.argsort(contributions[0])


# In[ ]:


[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]


# In[ ]:


contributions[0].sum()


# # Our final model!

# In[ ]:


m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# Our score of 0.21229 is as good as the competition winner. Its not directly comparable because we are using different validation set but you can expect similar results. This  is so amazing, with no knowledge of the domain, using only our ML skills, we can easily win this kaggle competition. I highly recommend reading what the winner's did, their explanation and their code. You can find all of it in the [discussion tab](https://www.kaggle.com/c/bluebook-for-bulldozers/discussion).

# ## Summary
# 
# In this notebook we covered:
# - **Feature Importance** : How taking only the important features not only saved us from large computation but also gave us better scores
# - **Removing redundant features** : We use scipy package and created a **dendrogram**, then removed the features which represented the same imformation
# - **Extrapolation** : We remove temporal features to help the model generalize better for new data, by learning the features which are actually important
# - **Tree Interpretation** : Tree interpretation can help you answer question that was not possible before. Also opens new doors for domain experts to analysis
# 
# It was a great learning experience for me. Writing these notebooks was also a part of learning. I am really glad that I did it. I am planning to write a medium article as well.
# 
# Hope these [[Part-1](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-1) & [Part-2](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-2)] notebooks will help you to learn something new. Please comment to me know the things I need to work on.
