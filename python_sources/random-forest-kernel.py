#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip uninstall fastai -y')
get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip list | grep fast')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor
from pandas_summary import DataFrameSummary
from IPython.display import display

from sklearn import metrics


# In[ ]:


df_raw = pd.read_csv('../input/train/Train.csv', low_memory=False, parse_dates=['saledate'])


# In[ ]:


def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)


# In[ ]:


display_all(df_raw.tail().transpose())


# In[ ]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# In[ ]:


train_cats(df_raw)


# In[ ]:


df_raw.UsageBand.cat.categories


# In[ ]:


df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# In[ ]:


os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')


# In[ ]:


add_datepart(df_raw, 'saledate')


# In[ ]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)


# In[ ]:


m.score(df, y)


# In[ ]:


def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 12000 # Same as kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
              m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


n = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# Draw tree here
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# Bagging
# Training multiple trees with rows chosen at random
# So that each tree has a different insight on the data
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


# Our forest predicted 9.3, real value was 9.1
preds.shape


# In[ ]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])


# In[ ]:


# As we see adding more and more trees mean increasing the r_sqaured
# Let's try adding more trees
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# We observe that even though we are doubling the number of trees, the r_sqaured on validation set is still 0.7
# 
# So after a point adding more and more trees does not make sense. It will never get worse but it's not getting worse

# # Out-of-bag (OOB) score

# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# When we pass oob_score as True, the model takes all the rows (which were left out randomly) to create a validation dataset for each tree and then averages them to get the accuracy!

# # Reducing over fitting
# Instead of creating bags from a subset, why not give each tree access to the complete dataset

# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)


# In[ ]:


set_rf_samples(20000)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# # Tree building parameters

# In[ ]:


reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# Another way to reduce overfitting is restrict min_samples_leaf.
# We tell the tree to stop spliting when it has 3 leafs.
# 3, 5, 7 are good values to try with but for large datasets it can be hunderds or thousands

# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# Another idea is your trees should not be correlated with each other
# Therefore in addition to taking a samples of rows we also take samples of columns.
# This way we can reduce correaltion between columns and get a better result
# 
# Here we use max_features=0.5 
# Which mean randomly pick half of the columns

# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# # Confidence based on tree variance

# Now, we know that we take mean of predictions from different trees and that minimizes our error, but what if we come across a row which is new to the trees and most of the trees give wrong insights.
# 
# To tackle this problem, we take standard deviation of our predictions and see if our std. dev. is high, we know that this is a row that our forest has not seen before!

# In[ ]:


set_rf_samples(50000)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')
# mean, std. deviation
np.mean(preds[:, 0]), np.std(preds[:, 0])


# In[ ]:


def get_preds(t): return t.predict(X_valid)
# parallel_trees is a fast ai function that get help you run trees in parallel
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:, 0]), np.std(preds[:, 0])


# In[ ]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();


# In[ ]:


flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ


# In[ ]:


enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0, 11))


# In[ ]:


enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0, 11))


# In[ ]:


raw_valid.ProductSize.value_counts().plot.barh()


# In[ ]:


flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby('ProductSize').mean()
summ


# In[ ]:


(summ.pred/summ.pred_std).sort_values(ascending=False)


# # Feature Importance (Most important)

# In[ ]:


fi = rf_feat_importance(m, df_trn)
fi[:10]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10, 6), legend=False)


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:30])


# In[ ]:


to_keep = fi[fi.imp > 0.005].cols; len(to_keep)


# In[ ]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi)


# We notice here that importance for Coupler System reduced drastically after removing less important feature. That is because there must be some co relation with some features that spiked by the importance of Coupler System. 
# That is why I trust the new dataset better than the previous dataset.

# # One-hot encoding
# 
# max_n_cats=7 means one-hot enocde every category with number of cats less than 7.

# In[ ]:


df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25])


# In this case the model got worse, but gave us another insight that Enclosure_EROPS is the most importance thing, even more than YearMade!
# 
# Now you can know what EROPS mean and why is it so important.

# # Removing redundant variables
# One thing that makes it harder to interpret the data is if variables with very similar meanings exist in the dataset. So we try to remove redundant variables.

# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16, 12))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[ ]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# In[ ]:


get_oob(df_keep)


# In[ ]:


for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# In[ ]:


to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))


# So this looks good, let's use this dataframe from now on!

# In[ ]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


np.save('tmp/keep_cols.npy', np.array(df_keep.columns))


# In[ ]:


keep_cols = np.load('tmp/keep_cols.npy', allow_pickle=True)
df_keep = df_trn[keep_cols]


# And let's see how the model performs on full dataset.

# In[ ]:


reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40, max_features=0.5, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# # Partial Dependance

# In[ ]:


from pdpbox import pdp
from plotnine import *


# In[ ]:


set_rf_samples(50000)


# In[ ]:


df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'plot_fi')


# In[ ]:


plot_fi(rf_feat_importance(m, df_trn2)[:10])


# One important piece of information is how old was the Bulldozer which was sold.
# So we plot YearMade and saleElapsed on a scatter plot

# In[ ]:


df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.1, figsize=(10,8));


# Now we understand, there are data points that have year made in 1000s. We can assume that no Bulldozers were made in 1000
# So this may be a way to handle empty or unknown values

# In[ ]:


x_all = get_sample(df_raw[df_raw.YearMade>1960], 300)


# In[ ]:


ggplot(x_all, aes('YearMade', 'SalePrice')) + stat_smooth(se=True)


# In[ ]:


x = get_sample(X_train[X_train.YearMade>1930], 500)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'pdp.pdp_isolate')


# In[ ]:


def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, x.columns, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                       cluster=clusters is not None, n_cluster_centers=clusters)


# In[ ]:


plot_pdp('YearMade')


# In[ ]:


plot_pdp('YearMade', clusters=5)


# In[ ]:


feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, x.columns, feats)
pdp.pdp_interact_plot(p, feats)


# In[ ]:


plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')


# In[ ]:


df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear - df_raw.YearMade


# In[ ]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep))


# # Tree interpreter

# In[ ]:


get_ipython().system('pip install treeinterpreter')
from treeinterpreter import treeinterpreter as ti


# In[ ]:


df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)


# In[ ]:


row = X_valid.values[None, 0]; row


# In[ ]:


prediction, bias, contributions = ti.predict(m, row)


# In[ ]:


prediction[0], bias[0]


# In[ ]:


[o for o in zip(df_keep.columns, df_valid.iloc[0], contributions[0])]


# In[ ]:


contributions[0].sum()


# In[ ]:




