#!/usr/bin/env python
# coding: utf-8

# **Source: fast.ai course v3**
# https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-3-fa4065d8cb1e

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install fastai==0.7.0 --no-deps')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

import seaborn as sns

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
set_plot_sizes(12,14,16)


# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# **Difference between train and validation set**
# [Explanation here](https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-5-df45f0c99618)

# In[ ]:


train_cpy = train.copy()
test_cpy = test.copy()
train_cpy['is_valid'] = 0
test_cpy['is_valid'] = 1


# In[ ]:


train_test = pd.concat([train_cpy.drop('SalePrice', axis=1), test_cpy])
train_cats(train_test)
x, y, nas = proc_df(train_test, 'is_valid')


# In[ ]:


def get_train_valid_oob(x, y):
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-1, oob_score=True)
    m.fit(x, y)
    return m.oob_score_


# In[ ]:


get_train_valid_oob(x, y)


# The OOB score is very high, since we haven't removed Id column
# which clearly indicates whether row is from valid or not.

# In[ ]:


train_test.drop(["Id"], axis=1, inplace=True)


# In[ ]:


x, y, nas = proc_df(train_test, 'is_valid')
get_train_valid_oob(x, y)


# Now we can see that train and test set are hard to distinguish by the model,
# so we can assume similar distributions of each of the features in train and test set,
# and generally homogeneity of train and test set.
# That means a random sample from training set is appropriate for validation set.

# In[ ]:


train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

duplicates = [
    "GarageArea", # Twin with "GarageCars", but lower correlation
    "TotRmsAbvGrd", # same, with "GrLivArea"
    "1stFlrSF" # Twin with "TotalBsmtSF"
]

train.drop(duplicates, axis=1, inplace=True)
test.drop(duplicates, axis=1, inplace=True)


# In[ ]:


# Shuffle the dataset to get better train/valid split
train = train.sample(frac=1, random_state=0).reset_index(drop=True)


# In[ ]:


# Simple label encoding
train_cats(train)


# In[ ]:


df, y, nas = proc_df(train, 'SalePrice')


# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=0)
m.fit(df, y)
m.score(df,y)


# In[ ]:


# Train/valid split
def split_vals(a,n): return a[:n], a[n:]
n_valid = 300
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
raw_train, raw_valid = split_vals(train, n_trn)


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


# Simple RF model
m = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=-1, random_state=0, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# Feature importance - first try
def plot_fi(fi, title=None): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False, title=title)
fi = rf_feat_importance(m, df)
plot_fi(fi[:25], title="Feature importance - first glance")


# In[ ]:


# Remove unimportant columns
to_keep = fi[fi.imp > 0.001].cols
len(to_keep)


# In[ ]:


df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=-1, random_state=0, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# Feature importance again
fi = rf_feat_importance(m, df_keep)
plot_fi(fi[:25], title="Feature importance - after dropping unimportant")


# In[ ]:


# OneHot Encoding - all categorical vars
df_trn2, y_trn, nas = proc_df(train, 'SalePrice', max_n_cat=42)
X_train, X_valid = split_vals(df_trn2, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=100, max_features=0.27, n_jobs=-1, random_state=0, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25], title="Feature importance - OneHot Encoding")


# In[ ]:


to_keep = fi[fi.imp > 0.001].cols
df_keep2 = df_trn2[to_keep]
len(to_keep)


# In[ ]:


X_train, X_valid = split_vals(df_keep2, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=100, max_features=0.27, n_jobs=-1, random_state=0, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_keep2)
plot_fi(fi[:25], title="Feature importance - OneHot Encoding after dropping unimportant")


# In[ ]:


def get_preds(t): return t.predict(X_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:,0]), np.std(preds[:,0])


# [Tree variance](https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-4-a536f333b20d) by OverallQual

# In[ ]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.OverallQual.value_counts().plot.barh();


# In[ ]:


raw_train.OverallQual.value_counts().plot.barh()


# In[ ]:


sns.distplot(x.pred_std)


# In[ ]:


def mean_stdev_by_col(df, colname):
    flds = [colname, 'SalePrice', 'pred', 'pred_std']
    summ = x[flds].groupby(flds[0]).mean()
    summ['rel_std'] = (summ.pred_std/summ.pred)
    return summ

def plot_relative_stdev(df, colname):
    mean_std = mean_stdev_by_col(df, colname)
    sns.barplot(x=mean_std.index, y='rel_std', data=mean_std.sort_values(by='rel_std'))    


# In[ ]:


plt.figure(figsize=(12, 6))
plot_relative_stdev(x, "OverallQual")


# In[ ]:


plot_relative_stdev(x, "GarageCars")


# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


def draw_dendrogram(df, title=None):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(25,30))
    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
    plt.title(label=title, fontdict={'fontsize': 25})
    plt.show()


# In[ ]:


draw_dendrogram(df_keep, title='Dendrogram (numerical encodings)')


# In[ ]:


draw_dendrogram(df_keep2, title="OneHot encodings")


# In[ ]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=100, max_features=0.27, n_jobs=-1, random_state=0, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# In[ ]:


get_oob(df_keep)


# In[ ]:


drop_candidates = ["Exterior1st", "Exterior2nd", "FireplaceQu", "Fireplaces", "GarageYrBlt"]
for c in drop_candidates:
    print(c, get_oob(df_keep.drop(c, axis=1)))


# In[ ]:


# Two columns to drop: Exterior1st and FireplaceQu
drops = ["Exterior1st", "FireplaceQu"]
df1 = df_keep.copy()
# Add and drop vs drop - adding slightly better here - we do not lose information
df1["Exterior2nd"] += df1["Exterior1st"]
print(get_oob(df_keep.drop(drops, axis=1)), get_oob(df1.drop(drops, axis=1)))


# In[ ]:


# Drop collinear columns
df_keep["Exterior2nd"] += df_keep["Exterior1st"]
df_keep.drop(drops, axis=1, inplace=True)
get_oob(df_keep)


# In[ ]:


draw_dendrogram(df_keep, "After dropping Exterior1st and FireplaceQu")


# In[ ]:


df2 = df_keep.copy()
df2.ExterQual += df2.KitchenQual
get_oob(df2.drop("KitchenQual", axis=1))

