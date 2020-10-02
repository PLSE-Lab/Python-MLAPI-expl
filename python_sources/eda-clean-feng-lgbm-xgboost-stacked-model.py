#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# *  [Introduction](#section1) 
# *  [Libraries](#section2) 
# *  [EDA](#section4)
#       - [Read the data in](#section5)
#           - [Description](#section6)
#           - [Target column](#section7)
#       - [Cleaning](#section8)
#           - [Missing values](#section9)
#               - [Train/Test](#section10)
#               - [Historical transactions](#section12)
#               - [New transactions](#section13)
#       - [Feature engineering](#section14)
#           - [Train/Test](#section15)
#           - [Historical + New transactions](#section18)
#           - [Aggregate function](#section40)   
# * [Data preparation](#section20)     
#    - [Merge](#section56)
#    - [Final Train/Test](#section57)
# * [Modeling and testing](#section21)
#     - [LightGBM](#section22)
#     - [Xgboost](#section23)
#     - [Summary of results](#section24)
#     - [Feature importance](#section25)
#     - [Ensembled model: averaged and stacked](#section26)
#         - [Model definition](#section27)
#         - [Predictions](#section28)
#         - [Results](#section29)
# * [Submission](#section30)
# * [References](#section31)
# 
# by @samaxtech
# 
# If you found this kernel helpful or would like to share your thoughts feel free to upvote and leave a comment! :)

# ---
# <a id='section1'></a>
# # Introduction
# As part of the Elo Merchant Category Recommendation Kaggle competition, this kernel aims to predict a merchant loyalty score for a certain credict card holder.

# <a id='section2'></a>
# # Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import math
import datetime

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from mlxtend.regressor import StackingCVRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')


# In[ ]:


Image(url= "https://upload.wikimedia.org/wikipedia/commons/1/10/Logo-ELO-NEG-Black.png")


# ---
# <a id='section4'></a>
# # EDA 

# <a id='section5'></a>
# ## Read in the data

# In[ ]:


# Historical and new transactions data
hist_trans = pd.read_csv('../input/historical_transactions.csv')
new_trans = pd.read_csv('../input/new_merchant_transactions.csv')

# Train and Test data
train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])

train_idx = train.shape[0]
test_idx = test.shape[0]

print("--------------------------")
print("Train shape: ", train.shape)
print("Test shape: ", test.shape)
print("--------------------------")
print("Historical transactions shape: ", hist_trans.shape)
print("New transactions shape: ", new_trans.shape)


# <a id='section6'></a>
# ### Description

# In[ ]:


print("----------------------------------------------------------------")
print("Train")
print("----------------------------------------------------------------")
print(train.info())
print("\n----------------------------------------------------------------")
print("Test")
print("----------------------------------------------------------------")
print(train.info())
print("\n----------------------------------------------------------------")
print("Historical transactions")
print("----------------------------------------------------------------")
print(hist_trans.info())
print("\n----------------------------------------------------------------")
print("New transactions")
print("----------------------------------------------------------------")
print(new_trans.info())


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


hist_trans.head()


# In[ ]:


new_trans.head()


# <a id='section7'></a>
# ### Target column

# In[ ]:


print("Target description:\n\n", train['target'].describe())
print("\n--------------------------------------------------------------------------------------------")
print("\nTarget values:\n\n", train['target'].value_counts())


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
ax1, ax2 = axes.flatten()

# Distribution
sns.distplot(train['target'], ax=ax1, color='Green')

# Sorted correlations with target
sorted_corrs = train.corr()['target'].sort_values(ascending=False)
sns.heatmap(train[sorted_corrs.index].corr(), ax=ax2)

ax1.set_title('Target Distribution')
ax2.set_title('Correlations')
plt.show()
del sorted_corrs


# There seem to be 2207 values around -33 for the target column, which follows a normal distribution. Let's take that into account. Also, 'feature_3' correlates better than 'feature_2' and 'feature_1' with 'target'.
# 
# Let's confirm the number of values under -30.

# In[ ]:


under_30 = train.loc[train['target'] < -30, 'target'].count()
print("Under -30:", under_30, "values.")


# <a id='section8'></a>
# # Cleaning
# <a id='section9'></a>
# ## Missing

# In[ ]:


print("MISSING VALUES BEFORE CLEANING\n")
print("--------------------------------------------------\nTrain:\n--------------------------------------------------\n", train.isnull().sum())
print("\n--------------------------------------------------\nTest:\n--------------------------------------------------\n", test.isnull().sum())
print("\n--------------------------------------------------\nHistorical transactions:\n--------------------------------------------------\n", hist_trans.isnull().sum())
print("\n--------------------------------------------------\nNew transactions:\n--------------------------------------------------\n", new_trans.isnull().sum())


# <a id='section10'></a>
# ### Train/Test
# There's no null values for train. However, there seem to be one observation with a missing 'first_active_month' in test.

# In[ ]:


test_missing = test[test.isnull()['first_active_month']]
idx_test_missing = test_missing.index
test_missing


# We can look for all the observations that match the same 'feature_1', 'feature_2' and 'feature_3' values as that, and replace it with the 'first_active_month' that corresponds to their mode.

# In[ ]:


same_category = test[(test['feature_1'] == 5) & (test['feature_2'] == 2) & (test['feature_3'] == 1)]
test.loc[idx_test_missing, 'first_active_month'] = same_category['first_active_month'].mode()[0]

del same_category
test.iloc[11578]


# <a id='section12'></a>
# ### Historical transactions

# Similarly, in this case let's drop the missing rows for 'category_3' and 'merchant_id' (less than 1% of total).

# In[ ]:


hist_trans.dropna(subset=['category_3', 'merchant_id'], inplace=True)


# For 'category_2', since it has about 10% of missing values, let's replace them with the rounded average value (since values for this column include [1.0, 2.0, 3.0, 4.0, 5.0]), as seen below.

# In[ ]:


hist_trans['category_2'].describe()


# In[ ]:


hist_trans['category_2'].fillna((math.floor(hist_trans['category_2'].mean())), inplace=True)


# <a id='section13'></a>
# ### New transactions
# Once more, since there are missing values in 'category_3', 'merchant_id', 'category_2' and they add up to no more than about 5%, let's  drop the corresponding rows.

# In[ ]:


new_trans.dropna(inplace=True)


# Lastly, let's confirm no null values are present after cleaning.

# In[ ]:


print("MISSING VALUES AFTER CLEANING\n")
print("--------------------------------------------------\nTrain:\n--------------------------------------------------\n", train.isnull().sum())
print("\n--------------------------------------------------\nTest:\n--------------------------------------------------\n", test.isnull().sum())
#print("\n--------------------------------------------------\nMerchant:\n--------------------------------------------------\n", merchants.isnull().sum())
print("\n--------------------------------------------------\nHistorical transactions:\n--------------------------------------------------\n", hist_trans.isnull().sum())
print("\n--------------------------------------------------\nNew transactions:\n--------------------------------------------------\n", new_trans.isnull().sum())


# <a id='section14'></a>
# # Feature engineering

# In[ ]:


# Merge train and test for data processing
data = pd.concat([train, test], ignore_index=True)

# Check shapes match
print("Train ({}) + Test ({}) observations: {}".format(train.shape[0], test.shape[0], train.shape[0] + test.shape[0]))
print("Merged shape:", data.shape)

del train
del test


# <a id='section15'></a>
# ## Train/Test

# In[ ]:


# Year and month, separately
data['year'] = data['first_active_month'].dt.year
data['month'] = data['first_active_month'].dt.month

# Elapsed time, until the latest date on the dataset
data['elapsed_time'] = (datetime.date(2018, 2, 1) - data['first_active_month'].dt.date).dt.days

# Categorical features: 'feature_1', 'feature_2' and 'feature_3'
cont = 1
for col in ['feature_1', 'feature_2', 'feature_3']:
    dummy_col = pd.get_dummies(data[col], prefix='f{}'.format(cont))
    data = pd.concat([data, dummy_col], axis=1)
    data.drop(col, axis=1, inplace=True)
    cont += 1
    
data.head()


# <a id='section18'></a>
# ### Historical + New transactions
# Let's create a column called 'new' on 'hist_trans' and 'new_trans' such that, before concatening them, they have the age reference:
# 
# - 1: New
# - 0: Historical

# In[ ]:


new_trans['new'] = 1
hist_trans['new'] = 0

# Concatenate new_trans and hist_trans
trans_data = pd.concat([new_trans, hist_trans])

del new_trans
del hist_trans


# More preprocessing: 'category_1', 'category_2' and 'category_3'.

# In[ ]:


# Change Yes/No for 0/1 in 'authorized_flag' and 'category_1'
yes_no_dict = {'Y':1, 'N':0}
trans_data['authorized_flag'] = trans_data['authorized_flag'].map(yes_no_dict)
trans_data['category_1'] = trans_data['category_1'].map(yes_no_dict)

# Create five different cols for 'category_2'
dummy_col = pd.get_dummies(trans_data['category_2'], prefix='category_2')
trans_data = pd.concat([trans_data, dummy_col], axis=1)
trans_data.drop('category_2', axis=1, inplace=True)
    
# Create three different cols for categorical A/B/C in 'category_3'
dummy_col = pd.get_dummies(trans_data['category_3'], prefix='cat3')
trans_data = pd.concat([trans_data, dummy_col], axis=1)
trans_data.drop('category_3', axis=1, inplace=True)

trans_data.head()


# ---
# <a id='section40'></a>
# ### Aggregate function
# Aggregate function, grouped by 'card_id': min, max, mean, median, std, sum, nunique, range. 
# 
# Added:
# - Count on 'installments' and 'purchase_amount'.
# - Mode() on 'new' column (previously created).
# - Mean on new trans_data's category_2 dummy columns.
# - Mean on trans_data's category_4.
# - Mean on 'cat3_A', 'cat3_B' and 'cat3_C' (old 'category_3').
# - Mean on merchants' new dummy columns.

# In[ ]:


def aggregate_historical_transactions(trans_data):
    
    trans_data.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans_data['purchase_date']).astype(np.int64)*1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'cat3_A': ['mean'],
        'cat3_B': ['mean'],
        'cat3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['count', 'sum', 'median', 'max', 'min', 'std'],
        'installments': ['count', 'sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max'],
        'new':[lambda x:x.value_counts().index[0]] # Mode
        }
    
    agg_history = trans_data.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)

    df = (trans_data.groupby('card_id').size().reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

trans_data = aggregate_historical_transactions(trans_data)
trans_data.head()


# ---
# <a id='section20'></a>
# # Data preparation
# <a id='section56'></a>
# ## Merge

# In[ ]:


# Merch data (train + test) with trans_data (historical + new transactions)
processed_data = pd.merge(data, trans_data, on='card_id', how='left')
del data
del trans_data
print(processed_data.shape)
processed_data.head()


# <a id='section57'></a>
# ## Final Train/Test 

# In[ ]:


# Train and Test
train = processed_data[:train_idx]
test = processed_data[train_idx:]

del processed_data

# There are some nan values after feature eng in 'purchase_amount_std' and 'installments_std'
cols = ['purchase_amount_std', 'installments_std']

for col in cols:
    train[col].fillna((train[col].value_counts().index[0]), inplace=True)
    test[col].fillna((test[col].value_counts().index[0]), inplace=True)

target = train['target']

cols_2_remove = ['target', 'card_id', 'first_active_month']
for col in cols_2_remove:  
    del train[col]
    del test[col] 

# Check on shapes
print("--------------------------")
print("Train shape: ", train.shape)
print("Test shape: ", test.shape)
print("--------------------------")


# ---
# <a id='section21'></a>
# # Modeling and testing

# <a id='section22'></a>
# ## LightGBM

# In[ ]:


lgb_params = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 2000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 2000)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    

del fold_importance_df_lgb
del trn_data
del val_data

print(np.sqrt(mean_squared_error(oof_lgb, target)))


# <a id='section23'></a>
# ## Xgboost

# In[ ]:


train.rename(index=str, columns={"new_<lambda>": "new_mode"}, inplace=True)
test.rename(index=str, columns={"new_<lambda>": "new_mode"}, inplace=True)

xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 2000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=200)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits

del trn_data
del val_data
del watchlist

np.sqrt(mean_squared_error(oof_xgb, target))


# <a id='section24'></a>
# ## Summary of results

# In[ ]:


print("-----------------\nScores on train\n-----------------")
print('lgb:', np.sqrt(mean_squared_error(oof_lgb, target)))
print('xgb:', np.sqrt(mean_squared_error(oof_xgb, target)))

total_sum = 0.5*oof_lgb + 0.5*oof_xgb

print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))


# <a id='section25'></a>
# ## Feature importance

# In[ ]:


cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
del feature_importance_df_lgb


# ---
# <a id='section26'></a>
# # Ensembled model: averaged and stacked
# The follwing tests lgbm, xgb, catboost, random forest, decision tree, knn, ridge and lasso models individual performance, and compared for averaged and stacked models.
# <a id='section27'></a>
# ## Model definition

# In[ ]:


# Model definition
train_y = target

# Same lgbm and xgb models as before
lgbm_model = LGBMRegressor(
                objective="regression", metric="rmse", 
                max_depth=7, min_child_samples=20, 
                reg_alpha= 1, reg_lambda=1,
                num_leaves=64, learning_rate=0.001, 
                subsample=0.8, colsample_bytree=0.8, 
                verbosity=-1
)

xgb_model = XGBRegressor(
                eta=0.001, max_depth=7, 
                subsample=0.8, colsample_bytree=0.8, 
                objective='reg:linear', eval_metric='rmse', 
                silent=True
)


# Test catboost, random forest, decision tree, knn, ridge and lasso models individual performance, for averaged and stacked model
catboost_model = CatBoostRegressor(iterations=150)
rf_model = RandomForestRegressor(n_estimators=25, min_samples_leaf=25, min_samples_split=25)
tree_model = DecisionTreeRegressor(min_samples_leaf=25, min_samples_split=25)
knn_model = KNeighborsRegressor(n_neighbors=25, weights='distance')
ridge_model = Ridge(alpha=75.0)
lasso_model = Lasso(alpha=0.75)

# ------------------------------------------------------------------------------------------------
# Average regressor
class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors
        self.predictions = None

    def fit(self, X, y):
        for regr in self.regressors:
            regr.fit(X, y)
        return self

    def predict(self, X):
        self.predictions = np.column_stack([regr.predict(X) for regr in self.regressors])
        return np.mean(self.predictions, axis=1)
    
# Averaged & stacked models 
averaged_model = AveragingRegressor([catboost_model, xgb_model, rf_model, lgbm_model])


stacked_model = StackingCVRegressor(
    regressors=[catboost_model, xgb_model, rf_model, lgbm_model],
    meta_regressor=Ridge()
)

# Test performance
def rmse_fun(predicted, actual):
    return np.sqrt(np.mean(np.square(predicted - actual)))

rmse = make_scorer(rmse_fun, greater_is_better=False)

models = [
     ('CatBoost', catboost_model),
     ('XGBoost', xgb_model),
     ('LightGBM', lgbm_model),
     ('DecisionTree', tree_model),
     ('RandomForest', rf_model),
     ('Ridge', ridge_model),
     ('Lasso', lasso_model),
     ('KNN', knn_model),
     ('Averaged', averaged_model),
     ('Stacked', stacked_model),
]


scores = [
    -1.0 * cross_val_score(model, train.values, train_y.values, scoring=rmse).mean()
    for _,model in models
]


# In[ ]:


dataz = pd.DataFrame({ 'Model': [name for name, _ in models], 'Error (RMSE)': scores })
dataz.plot(x='Model', kind='bar')
plt.savefig('stacked_scores.png')


# <a id='section28'></a>
# ## Results

# In[ ]:


dataz


# <a id='section29'></a>
# ## Predictions

# In[ ]:


# Stacked model predictions (best score)
stacked_model.fit(train.values, target.values)    
predictions_stacked = stacked_model.predict(test.values)


# <a id='section30'></a>
# # Submission

# In[ ]:


# LightGBM/Xgboost
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df.to_csv("submission_lgbxgboost.csv", index=False)

# Stacked
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = predictions_stacked
sub_df.to_csv("submission_stacked.csv", index=False)


# ---
# <a id='section31'></a>
# # References
# Special thanks to the following references:
# - https://www.kaggle.com/mjbahmani/a-data-science-framework-for-elo (@mjbahmani)
# - https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you (@youhanlee)
# - https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737 (@peterhurford)
# - https://www.kaggle.com/eikedehling/comparing-models-xgb-lgb-rf-and-stacking (@eikedehling)

# In[ ]:




