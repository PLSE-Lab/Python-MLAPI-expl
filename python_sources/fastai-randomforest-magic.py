#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai==0.7.0')


# In[ ]:


#It will automatically reload the latest module when you start again
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
#Used to display plots and graphs inside Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')
#The following are fastAi imports
from fastai.imports import * 
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Loading the Data

# In[ ]:


PATH = "../input/"


# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


df_raw = pd.read_csv(PATH+ 'train.csv', low_memory = False ,parse_dates=['YrSold'])


# ## Preprocessing

# In[ ]:


df_raw


# In[ ]:


#function to display all the data of the dataframe at one go
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


# In[ ]:


df_raw.SalePrice  = np.log(df_raw.SalePrice)


# In[ ]:


df_raw.YrSold.head(5)


# In[ ]:


def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(df_raw, 'YrSold')


# In[ ]:


df_raw.YrSoldYear.head(5)


# In[ ]:


train_cats(df_raw) #behind scenes everything will be converted into numbers


# In[ ]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[ ]:


df.columns


# In[ ]:


df.drop(columns=['Id'],inplace=True)


# ## Let's start creating basic model's 
# and slowly move towards complex models

# In[ ]:


m = RandomForestRegressor(n_jobs=-1) #njobs = -1 use all the resouces for running our model
m.fit(df, y)
m.score(df, y)


# In[ ]:


def split_vals(a,n) : return a[:n].copy(), a[n:].copy()


# In[ ]:


n_valid = 200 #same as kaggle's test size
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df, n_trn)
x_train, x_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
x_train.shape, y_train.shape, x_valid.shape


# In[ ]:


#the following code will print the score
def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(x_train), y_train), rmse(m.predict(x_valid), y_valid), m.score(x_train, y_train), m.score(x_valid, y_valid)]
    if hasattr(m ,'obb_score_'): res.append(m.obb_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(x_train,y_train)')
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df)
fi[:10]


# In[ ]:


fi.plot('cols', 'imp', figsize = (10,6), legend = False)


# In[ ]:


def plot_fi(fi) : return fi.plot('cols', 'imp', 'barh', figsize = (12,7), legend=False)


# In[ ]:


plot_fi(fi[:30])


# In[ ]:


to_keep = fi[fi.imp > 0.005].cols
len(to_keep)


# In[ ]:


df_keep = df[to_keep].copy()
x_train, x_valid = split_vals(df_keep, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi)


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.scatter(df_raw['OverallQual'], df_raw['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('YearMade')


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.scatter(df_raw['GrLivArea'], df_raw['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('YearMade')


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.scatter(df_raw['YearBuilt'], df_raw['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('YearBuilt')


# In[ ]:


from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
plt.figure(figsize=(15,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size = 16)
plt.show()


# In[ ]:


cols = df_keep.columns


# In[ ]:


m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
m.fit(x_train, y_train)
print_score(m)


# ## Prediction
# Now for this basic model, let's load the test dataset and try out predictions

# In[ ]:


test = pd.read_csv(PATH + 'test.csv'  ,low_memory = False ,parse_dates=['YrSold'])


# In[ ]:


test.head(5)


# In[ ]:


test.drop(columns=['Id'], inplace=True)


# In[ ]:


add_datepart(test, 'YrSold')


# In[ ]:


apply_cats(df=test, trn=df_raw)


# In[ ]:


test.info()


# In[ ]:


X_test, _, nas = proc_df(test, na_dict=nas)
nas


# In[ ]:


#these are the extra columns made by proc_df as they are unique only to test set, so we will drop it, i manually check them
X_test.drop(columns=['BsmtFinSF1_na', 'BsmtFinSF2_na', 'BsmtUnfSF_na', 'TotalBsmtSF_na', 'BsmtFullBath_na', 'BsmtHalfBath_na', 'GarageCars_na', 'GarageArea_na'], inplace=True)


# In[ ]:


x_test = X_test[cols]


# In[ ]:


x_test.columns


# In[ ]:


ypreds = m.predict(x_test)


# In[ ]:


ypreds


# In[ ]:


submit = pd.read_csv(PATH + "sample_submission.csv")
submit['SalePrice'] = ypreds*10000


# In[ ]:


submit.to_csv("submission.csv", index = False)


# End
# So this is the basic notebook which explore the first part of FastAI's machine learning course, we used here RandomForest as our model and trained the model on preprocessed data.
# 
# Upvote the kernel if you find it useful
# 
# Connect on Linkedin - https://www.linkedin.com/in/savannahar/
