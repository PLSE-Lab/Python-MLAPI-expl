#!/usr/bin/env python
# coding: utf-8

# # Predicting house prices

# #### I would greatly appreciate any help, comments and suggestions that any can give me! :)
# 
# #### Note this Kernel gives 0.14022. If I am not wrong, this is not too bad for a random forest (not boosted) and with minimal feature engineering. 

# In this Kernel, I am trying to tackle the 
# 'House Prices: Advanced Regression Techniques problem' using a random forest. My code is based on the fastai machine learning course code (https://course18.fast.ai/ml.html) with some edits.

# ## Loading packages and data

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd

from fastai.imports import *
#from fastai.structured import *
#from structured import *
from fastai.tabular import *
from fastai import *
#from fastai_structured import *


from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.cluster import hierarchy as hc

from fastai_structured import *


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


def display_small(df):
    with pd.option_context("display.max_rows", 10, "display.max_columns", 5): 
        display(df)


# In[ ]:


PATH = "../input/house-prices-advanced-regression-techniques/"
get_ipython().system('ls {PATH}')


# In[ ]:


## Training dataset
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False,
                     index_col='Id')
## Test dataset
df_test_raw = pd.read_csv(f'{PATH}test.csv', low_memory=False,
                          index_col='Id')


# ## Checking data and preprocessing

# Below, we are displaying the training and test data. This is just to spot any obvious problems. I used the display_all instead of display_small during the first run through.

# In[ ]:


display_small(df_raw.tail().T)


# In[ ]:


display_small(df_test_raw.tail().T)


# Now we display the description of the data. Again, I used the display_all instead of display_small during the first run through.

# In[ ]:


display_small(df_raw.describe(include='all').T)


# We then turn the categorical data into categorical values using the train_cats and apply_cats function

# In[ ]:


train_cats(df_raw)
apply_cats(df_test_raw,df_raw)


# Now we will go through the data and order the categorical variables. Actually, we don't: for some reason, this increased RMSE.

# In[ ]:


#df_raw.head()


# In[ ]:


#df_raw.MSZoning.cat.set_categories(['FV', 'RL', 'RM', 'RH', 'C (all)'], ordered=True, inplace=True)
#df_raw.LotShape.cat.set_categories(['Reg', 'R1', 'R2', 'R3'], ordered=True, inplace=True)
#df_raw.Utilities.cat.set_categories(['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], ordered=True, inplace=True)
#df_raw.LandSlope.cat.set_categories(['Gtl', 'Mod', 'Sev'], ordered=True, inplace=True)
#df_raw.ExterQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
#df_raw.ExterCond.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
#df_raw.BsmtQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.BsmtExposure.cat.set_categories(['Gd', 'Av', 'Mn', 'No', 'NA'], ordered=True, inplace=True)
#df_raw.BsmtFinType1.cat.set_categories(['GLD', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], ordered=True, inplace=True)
#df_raw.HeatingQC.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.Electrical.cat.set_categories(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', 'NA'], ordered=True, inplace=True)
#df_raw.KitchenQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.BsmtFinType1.cat.set_categories(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'], ordered=True, inplace=True)
#df_raw.FireplaceQu.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.GarageFinish.cat.set_categories(['Fin', 'RFn', 'Unf', 'NA'], ordered=True, inplace=True)
#df_raw.GarageQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.GarageCond.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.PavedDrive.cat.set_categories(['Y', 'P', 'N'], ordered=True, inplace=True)
#df_raw.PoolQC.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)
#df_raw.Fence.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ordered=True, inplace=True)


# From the description of the data on top, we know there are nans. We want to quickly see how much nans are in our data

# In[ ]:


## train set
(df_raw.isnull().sum().sort_values(ascending=False)/len(df_raw)).head(10)


# We now use proc_df to remove the nans and split the data into the input variables (df) and the dependent variables (y). Firstly, we log our SalePrice

# In[ ]:


df_raw['SalePrice'].head()


# In[ ]:


df_raw['SalePrice'] = np.log(df_raw['SalePrice']); df_raw['SalePrice'].head()


# In[ ]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# We now check the nans again to make sure they are all zeros (no nans left).

# In[ ]:


## train set
(df.isnull().sum().sort_values(ascending=False)/len(df)).head(5)


# The df dataframe now needs to be split into a training and validation set. We have set n_valid to be approximately 25% of the whole dataset.
# 

# In[ ]:


X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.25, random_state=42)


# "Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price"

# ## Base model

# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# ## Running some preliminary trees and diagrams

# In[ ]:


m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# Below is a function draw_tree taken from the package draw_tree. However, it wasn't working in the kaggle Kernel as it needed the imports to make it work. I just copied the source code from draw_tree

# In[ ]:


import IPython
import graphviz

def mydraw_tree(t, df, size=10, ratio=0.6, precision=0):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))


# In[ ]:


## Draw tree is fastai
mydraw_tree(m.estimators_[0], X_train, precision=3)


# Looks like OverallQual gives the biggest mse difference followed Living space. That makes sense to me intuitively.

# ## Testing number of estimators needed

# We are going to run 500 estimators and look at the diminishing returns to get a feel of the number of estimators we should use

# In[ ]:


m = RandomForestRegressor(n_estimators=300,n_jobs=-1,oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# Note: Currently the RMSE is at 0.139 (r2 at 0.888); which places us approximately a bit below the middle of the pack (July 2019) in Kaggle (assuming the test data gives the same number).

# Below, we are going to get the predictions of every tree in our 300 trees

# In[ ]:


## Each tree is found in m.estimators_

preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


preds.shape


# We plot the changes in the r2_score as we increase number of trees

# In[ ]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(300)]);


# It looks like the majority of the improvements are before 20 trees. However, there still seems to be visible (albeit small) improvements up to 100 trees or so.

# ## Improved model

# We try different min_samples_leaf, max_features and set_rf_samples(1460) to see what gives a good result. Note we should really have used a grid search instead of manually searching

# In[ ]:


param_grid = {
    'min_samples_leaf': [1, 5, 10, 15, 20, 25],
    'max_features': ['sqrt', 'log2', 0.2, 0.4, 0.6],
    'n_estimators': [300],
    'n_jobs': [-1],
    'random_state': [42]
}

m = RandomForestRegressor(n_estimators=300)

grid_search = GridSearchCV(m, param_grid=param_grid, cv=5, iid=False,
                           verbose=0, scoring='neg_mean_squared_error');
grid_search.fit(X_train, y_train);
#print(grid_search.cv_results_)


# In[ ]:


print(grid_search.best_score_)


# In[ ]:


print(grid_search.best_params_)


# In[ ]:


myscoredf = pd.DataFrame(grid_search.cv_results_)[['param_min_samples_leaf','param_max_features','mean_test_score']]; myscoredf.head(10)


# In[ ]:


myscoredf = myscoredf.pivot('param_min_samples_leaf','param_max_features','mean_test_score')


# In[ ]:


ax = sns.heatmap(myscoredf, annot=True, fmt=".5g", cmap=cm.coolwarm)


# In[ ]:


#set_rf_samples(800)
#reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1)
m.fit(X_train, y_train);
print_score(m)


# Now we have a RMSE of 0.138 (r2 of 0.871).

# ## Looking at feature importance

# Here we want to look at feature importance and remove some useless/redundant features 

# In[ ]:


fi = rf_feat_importance(m, df); fi[:10]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);


# Will set limit to appoximately >0.005. This pushes the features from 82 to 27

# In[ ]:


len(fi.cols)


# In[ ]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[ ]:


df_keep = df[to_keep].copy()
#X_train, X_valid = split_vals(df_keep, n_trn)
X_train,X_valid,_ ,_ = train_test_split(df_keep,y,test_size=0.25, random_state=42)


# We now redo the n_estimators=100 to see if we have any changes to the RMSE and r2_score 

# In[ ]:


m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1)
m.fit(X_train, y_train);


# In[ ]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# 1. OverallQual: Rates the overall material and finish of the house
# 2. GrLiveArea: Above grade (ground) living area square feet
# 3. YearBuilt: Year garage was built
# 4. ExterQual: Evaluates the quality of the material on the exterior. **(We should make sure this is sorted by quality) (Perhaps also make a overall quality average - difficult to judge though)**
# 5. GarageCars: Size of garage in car capacity **(We should probably make a total area feature)**
# 6. GarageArea: Size of garage in square feet **(We should probably make a total area feature)**
# 7. TotalBsmtSF: Total square feet of basement area **(We should probably make a total area feature)**
# 8. 1stFlrSF: First Floor square feet
# 9. GarageYrBlt: Year garage was built
# 10. LotArea: Lot size in square feet

# #### To do list:
# 1. Total area = TotalBsmtSF + GrLivArea + GarageArea + PoolArea
# 2. GoodLivArea = TotalBsmtSF + GrLivArea - LowQualFinSF
# (Got this idea from https://towardsdatascience.com/my-first-kaggle-competition-using-random-forests-to-predict-housing-prices-76efee28d42f)
# 2. Building age = YrSold - YearBuilt
# 3. Remod age = YrSold - YearRemodAdd

# ## Adding the our extra interaction features

# In[ ]:


df_raw2 = df_raw.copy(); df_raw2.head()


# Note that after trial and error, it seems that TotalArea only makes the predictions worse. Therefore we only go with three of the new features.

# In[ ]:


#df_raw2['TotalArea']   = (df_raw2['TotalBsmtSF'] + df_raw2['GrLivArea']
#                       + df_raw2['GarageArea'] + df_raw2['PoolArea'])
df_raw2['GoodLivArea'] = (df_raw2['TotalBsmtSF'] + df_raw2['GrLivArea']
                       - df_raw2['LowQualFinSF'])
df_raw2['BuildingAge'] = (df_raw2['YrSold'] - df_raw2['YearBuilt'])
df_raw2['RemodAge']    = (df_raw2['YrSold'] - df_raw2['YearRemodAdd'])
df_raw2.head()


# In[ ]:


df2,_ , nas = proc_df(df_raw2, 'SalePrice')
X_train,X_valid,_ ,_ = train_test_split(df2,y,test_size=0.25, random_state=42)


# In[ ]:


m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1)
m.fit(X_train, y_train);
print_score(m)


# This is an improvement.

# In[ ]:


fi2 = rf_feat_importance(m, df2); fi2[:10]


# In[ ]:


to_keep2 = fi2[fi2.imp>0.002].cols
df_keep2 = df2[to_keep2].copy()
X_train,X_valid,_ ,_ = train_test_split(df_keep2,y,test_size=0.25, random_state=42)


# In[ ]:


print(len(fi2), len(to_keep2))


# In[ ]:


plot_fi(fi2[:25]);


# As expected, TotalArea, GoodLivArea, Building Age and Remod Age are all very important.

# In[ ]:


m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1)
m.fit(X_train, y_train);
print_score(m)


# A small improvement.

# ## Removing redundant features

# We use a dendrogram which is part of hierarchical clustering.
# 
# This is where we look at every pair of objects and see which is the closest. We then delete them and replace with the midpoint of the pair.

# In[ ]:


## We use spearman's correlation
corr = np.round(scipy.stats.spearmanr(df_keep2).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep2.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[ ]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1, oob_score=True)
    x,_ ,_ ,_ = train_test_split(df_keep2,y,test_size=0.25,
                                       random_state=42)

    m.fit(x, y_train)
    return m.oob_score_


# In[ ]:


get_oob(df_keep2)


# In[ ]:


df_keep2.columns


# In[ ]:


for c in ('GrLivArea', 'GoodLivArea', 'FireplaceQu', 'Fireplaces',
          '1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars',
          'Exterior2nd', 'Exterior1st'):
    print(c, get_oob(df_keep2.drop(c, axis=1)))


# In[ ]:


to_drop = ['GrLivArea', 'FireplaceQu', 'TotalBsmtSF', 'GarageArea', 'Exterior1st']
get_oob(df_keep2.drop(to_drop, axis=1))


# Similar. So ignore this for final model

# ## Using 1-hot encoding to find important features

# This is important if one class in a categorical feature is especially important. We use all features again (instead of df_keep) to check the low contribution features as well.

# In[ ]:


df_hot,_ ,_ = proc_df(df_raw2, 'SalePrice', max_n_cat=5)
X_train,X_valid,_ ,_ = train_test_split(df_hot,y,test_size=0.25, random_state=42)

m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train);
print_score(m)


# RMSE at 0.136 (r2 at 0.894). No change.

# In[ ]:


fi = rf_feat_importance(m, df_hot)
plot_fi(fi[:25]);


# The only one that stands out is ExterQual_TA. Will not use one hot encoding.

# ## Final model

# In[ ]:


## We bootstrap from a different 20000 everytime
#set_rf_samples(1460)


# In[ ]:


df_test_raw2 = df_test_raw.copy()


# In[ ]:


#df_test_raw2['TotalArea']   = (df_test_raw2['TotalBsmtSF'] + df_test_raw2['GrLivArea']
#                       + df_test_raw2['GarageArea'] + df_test_raw2['PoolArea'])
df_test_raw2['GoodLivArea'] = (df_test_raw2['TotalBsmtSF'] + df_test_raw2['GrLivArea']
                       - df_test_raw2['LowQualFinSF'])
df_test_raw2['BuildingAge'] = (df_test_raw2['YrSold'] - df_test_raw2['YearBuilt'])
df_test_raw2['RemodAge']    = (df_test_raw2['YrSold'] - df_test_raw2['YearRemodAdd'])
df_test_raw2.head()


# In[ ]:


df_test, _, _ = proc_df(df_test_raw2, na_dict=nas)
df_test_keep = df_test[to_keep2].copy()


X_train,X_valid ,y_train,y_valid = train_test_split(df_keep2,y,test_size=0.01, random_state=42)
print(len(X_train), len(X_valid))


# In[ ]:


np.shape(X_train)


# In[ ]:


m = RandomForestRegressor(n_estimators=300,
                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                          max_features=grid_search.best_params_['max_features'],
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train);


# In[ ]:


y_test = np.exp(pd.Series(m.predict(df_test_keep))); y_test.head()


# In[ ]:


mysubmission = pd.DataFrame({'SalePrice': y_test.values}, index=df_test.index.values)
mysubmission.index.name = 'Id'
mysubmission.head(10)


# In[ ]:


pd.DataFrame.to_csv(mysubmission,'mysubmission.csv')

