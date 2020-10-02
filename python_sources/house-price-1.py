#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
import warnings
import xgboost
import lightgbm as lgb
import seaborn as sns
import datetime
import os
print(os.listdir("../input"))


# In[ ]:


warnings.filterwarnings("ignore")


# ## Import CSV

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Data Cleaning

# In[ ]:


obj_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols.remove('SalePrice')
num_cols.remove('Id')


# In[ ]:


def impute_object(df, obj_cols):
    df_copy = df.copy()
    # impute object null to Na/No/None if stated in description
    df_copy['Alley'].fillna('None', inplace = True)
    df_copy['MasVnrType'].fillna('None', inplace = True)
    df_copy['BsmtQual'].fillna('None', inplace = True)
    df_copy['BsmtCond'].fillna('NA', inplace = True)
    df_copy['BsmtExposure'].fillna('No', inplace = True)
    df_copy['BsmtFinType1'].fillna('NA', inplace = True)
    df_copy['BsmtFinType2'].fillna('NA', inplace = True)
    df_copy['FireplaceQu'].fillna('NA', inplace = True)
    df_copy['GarageType'].fillna('NA', inplace = True)
    df_copy['GarageFinish'].fillna('NA', inplace = True)
    df_copy['GarageQual'].fillna('NA', inplace = True)
    df_copy['GarageCond'].fillna('NA', inplace = True)
    df_copy['PoolQC'].fillna('NA', inplace = True)
    df_copy['Fence'].fillna('NA', inplace = True)
    df_copy['MiscFeature'].fillna('NA', inplace = True)

    # impute object null to mode value for not stated in description
    object_imputer = SimpleImputer(strategy='most_frequent', fill_value='missing_value')
    for col in obj_cols:
        df_copy[col] = object_imputer.fit_transform(df_copy[col].values.reshape(-1,1))
    return df_copy


# In[ ]:


train = impute_object(train, obj_cols)
test = impute_object(test, obj_cols)


# In[ ]:


def impute_numeric(df,num_cols):
    df_copy = df.copy()
    # impute numeric value to median
    imputer = Imputer(strategy = 'median')
    for col in num_cols:
        df_copy[col] = imputer.fit_transform(df_copy[col].values.reshape(-1,1))
    return df_copy


# In[ ]:


train = impute_numeric(train, num_cols)
test = impute_numeric(test, num_cols)


# ## EDA

# ### Basement Features

# In[ ]:


##Basement Quality affect SalePrice linearly
train[['BsmtQual','SalePrice']].groupby('BsmtQual').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


##People prefer no basement rather than poor quality basement
train[['BsmtCond','SalePrice']].groupby('BsmtCond').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


train[['BsmtExposure','SalePrice']].groupby('BsmtExposure').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


agg_bsmt = train[['BsmtQual','BsmtCond','BsmtExposure','SalePrice']]            .groupby(['BsmtCond','BsmtQual','BsmtExposure']).median()            .sort_values('SalePrice', ascending=False)


# In[ ]:


agg_bsmt['rank'] = agg_bsmt['SalePrice'].rank(ascending = True) 


# In[ ]:


clf = tree.DecisionTreeClassifier()


# In[ ]:


agg_bsmt['BsmtQual'] = agg_bsmt.index.get_level_values('BsmtQual')
agg_bsmt['BsmtCond'] = agg_bsmt.index.get_level_values('BsmtCond')
agg_bsmt['BsmtExposure'] = agg_bsmt.index.get_level_values('BsmtExposure')


# In[ ]:


one_hot_bsmt = pd.get_dummies(agg_bsmt[['BsmtQual','BsmtCond','BsmtExposure']])
test_bsmt = pd.get_dummies(test[['BsmtQual','BsmtCond','BsmtExposure']]) 


# In[ ]:


bsmt_cols = [one_hot_bsmt.columns.values.tolist()[i] for i in [one_hot_bsmt.columns.values.tolist().index(x) 
                                          for x in test_bsmt.columns.values.tolist()]]


# In[ ]:


clf = clf.fit(one_hot_bsmt[bsmt_cols], agg_bsmt['rank'])


# ### Garage Features

# In[ ]:


garage_col = train.columns.values.tolist() 
garage_col = [s for s in garage_col if any(xs in s for xs in ['Garage','SalePrice'])]


# In[ ]:


garage_df = train[garage_col].copy()


# In[ ]:


garage_df[['GarageType','SalePrice']].groupby('GarageType').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


garage_df[['GarageFinish','SalePrice']].groupby('GarageFinish').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


garage_df[['GarageQual','SalePrice']].groupby('GarageQual').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


garage_df[['GarageCond','SalePrice']].groupby('GarageCond').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


agg_garage = garage_df[['GarageType','GarageFinish','GarageQual','GarageCond','SalePrice']]            .groupby(['GarageType','GarageFinish','GarageQual','GarageCond']).median()            .sort_values('SalePrice', ascending=False)


# In[ ]:


agg_garage['rank'] = agg_garage['SalePrice'].rank(ascending = True) 
agg_garage['rank'] = agg_garage['rank'].astype(int)


# In[ ]:


clf_garage = tree.DecisionTreeClassifier()
agg_garage['GarageType'] = agg_garage.index.get_level_values('GarageType')
agg_garage['GarageFinish'] = agg_garage.index.get_level_values('GarageFinish')
agg_garage['GarageQual'] = agg_garage.index.get_level_values('GarageQual')
agg_garage['GarageCond'] = agg_garage.index.get_level_values('GarageCond')

one_hot_garage = pd.get_dummies(agg_garage[['GarageType','GarageFinish','GarageQual','GarageCond']])
test_garage = pd.get_dummies(test[['GarageType','GarageFinish','GarageQual','GarageCond']]) 


# In[ ]:


garage_cols = [one_hot_garage.columns.values.tolist()[i] for i in [one_hot_garage.columns.values.tolist().index(x) 
                                          for x in test_garage.columns.values.tolist()]]


# In[ ]:


clf_garage = clf_garage.fit(one_hot_garage[garage_cols], agg_garage['rank'])


# ### Exterior Features

# In[ ]:


exter_col = train.columns.values.tolist() 
exter_col = [s for s in exter_col if any(xs in s for xs in ['Exter','SalePrice'])]


# In[ ]:


exter_df = train[exter_col].copy()


# In[ ]:


exter_df[['Exterior1st','SalePrice']].groupby('Exterior1st').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


exter_df[['Exterior2nd','SalePrice']].groupby('Exterior2nd').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


exter_df[['ExterQual','SalePrice']].groupby('ExterQual').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


exter_df[['ExterCond','SalePrice']].groupby('ExterCond').median().sort_values('SalePrice', ascending=False).plot.bar()


# In[ ]:


agg_exter = exter_df.groupby(['Exterior1st','Exterior2nd','ExterQual','ExterCond']).median()            .sort_values('SalePrice', ascending=False)


# In[ ]:


agg_exter['rank'] = agg_exter['SalePrice'].rank(ascending = True) 
agg_exter['rank'] = agg_exter['rank'].astype(int)


# In[ ]:


clf_exter = tree.DecisionTreeClassifier()
agg_exter['Exterior1st'] = agg_exter.index.get_level_values('Exterior1st')
agg_exter['Exterior2nd'] = agg_exter.index.get_level_values('Exterior2nd')
agg_exter['ExterQual'] = agg_exter.index.get_level_values('ExterQual')
agg_exter['ExterCond'] = agg_exter.index.get_level_values('ExterCond')

one_hot_exter = pd.get_dummies(agg_exter[['Exterior1st','Exterior2nd','ExterQual','ExterCond']])
test_exter = pd.get_dummies(test[['Exterior1st','Exterior2nd','ExterQual','ExterCond']])


# In[ ]:


exter_cols = [one_hot_exter.columns.values.tolist()[i] for i in [one_hot_exter.columns.values.tolist().index(x) 
                                          for x in test_exter.columns.values.tolist()]]


# In[ ]:


clf_exter = clf_exter.fit(one_hot_exter[exter_cols], agg_exter['rank'])


# ## Label Encode

# In[ ]:


def encode_by_rank(df, clf_model, column):
    df_dummies = pd.get_dummies(df)
    df_dummies['rank'] = clf_model.predict(df_dummies[column])
    return df_dummies['rank']


# In[ ]:


train['rank_bsmt'] = encode_by_rank(train, clf, bsmt_cols)
train['rank_garage'] = encode_by_rank(train, clf_garage, garage_cols)
train['rank_exter'] = encode_by_rank(train, clf_exter, exter_cols)


# In[ ]:


test['rank_bsmt'] = encode_by_rank(test, clf, bsmt_cols)
test['rank_garage'] = encode_by_rank(test, clf_garage, garage_cols)
test['rank_exter'] = encode_by_rank(test, clf_exter, exter_cols)


# ## Feature Scaling

# ### Skewness Check

# In[ ]:


skew_cols = train[num_cols].skew().sort_values(ascending = False)
skew_cols = skew_cols[abs(skew_cols) > 2].index.values.tolist()


# ### Log Transform Skew Data (skewness > 2)

# In[ ]:


def log_transform(df, column):
    return np.log(df[column] + 1)


# In[ ]:


train_processed = train.copy()
test_processed = test.copy()


# In[ ]:


for skew_col in skew_cols:
    train_processed[skew_col] = log_transform(train_processed, skew_col)
    test_processed[skew_col] = log_transform(test_processed, skew_col)


# ## Feature Selection

# In[ ]:


train_processed = pd.get_dummies(train_processed)
test_processed = pd.get_dummies(test_processed)


# In[ ]:


correlation = train_processed.corr()['SalePrice'].sort_values()


# ## Label and Feature Separation

# In[ ]:


label = train['SalePrice']


# In[ ]:


feat_cols_train = train_processed.columns.values.tolist()
feat_cols_test = test_processed.columns.values.tolist()


# In[ ]:


feat_cols = [feat_cols_train[i] for i in [feat_cols_train.index(x) for x in feat_cols_test]]
feat_cols.remove('Id')


# In[ ]:


features = train_processed[feat_cols]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)


# In[ ]:


gbm = lgb.LGBMRegressor(boosting_type='gbdt', 
                        num_leaves=31, 
                        max_depth=-1, 
                        learning_rate=0.01, 
                        n_estimators=1000, 
                        max_bin=255, 
                        subsample_for_bin=50000, 
                        objective='regression', 
                        min_split_gain=0, 
                        min_child_weight=3,
                        min_child_samples=10, 
                        subsample=1, 
                        subsample_freq=1, 
                        colsample_bytree=1, 
                        reg_alpha=0.1, 
                        reg_lambda=0, 
                        seed=17,
                        silent=False, 
                        nthread=-1)


# In[ ]:


gbm.fit(X_train, y_train, 
            eval_metric='rmse',
            eval_set=[(X_test, y_test)],
            verbose = True)


# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importances_,X_train.columns)), columns=['Value','Feature'])
# take only top 80 % features
feature_imp['cum_pct'] = feature_imp['Value'].cumsum()/feature_imp['Value'].sum()


# In[ ]:


top_feats = feature_imp.loc[feature_imp['cum_pct'] >= 0.2]['Feature'].values.tolist()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.loc[feature_imp['cum_pct'] >= 0.2].sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features[top_feats], label, test_size=0.2, random_state=42)


# ## Linear Regression

# In[ ]:


linreg = LinearRegression()


# In[ ]:


linreg.fit(X_train, y_train)


# In[ ]:


price_prediction = abs(linreg.predict(X_test))


# ## Random Forest Regression

# In[ ]:


forest_reg = RandomForestRegressor()


# In[ ]:


forest_reg.fit(X_train, y_train)


# In[ ]:


f_price_prediction = forest_reg.predict(X_test)


# ## XGBoost

# In[ ]:


xgb = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.01,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[ ]:


xgb.fit(X_train, y_train)


# In[ ]:


xgb_predict = xgb.predict(X_test)


# ## Performance Benchmarking

# In[ ]:


def cross_validate(model, features, label, n_cv):
    scores = cross_val_score(model, features, label,
                             scoring="neg_mean_squared_error", cv=n_cv)
    rmse_scores = np.sqrt(-scores)
    
    return rmse_scores


# In[ ]:


models = {'linear' : linreg, 'random forest': forest_reg, 'xgboost': xgb}


# In[ ]:


def model_benchmark(models, X_test, y_test, n_cv):
    performance_benchmark = pd.DataFrame(columns=['model','rmsle'])
    for model in models:
        temp_model = [model]*n_cv
        temp_performance = cross_validate(models[model], X_test, y_test, n_cv)
        temp_df = pd.DataFrame({'model': temp_model, 'rmsle': temp_performance})
        performance_benchmark = pd.merge(performance_benchmark, temp_df, how = 'outer')
    sns.boxplot(x="model", y="rmsle", data=performance_benchmark)
    plt.title('Performance Benchmark')


# In[ ]:


model_benchmark(models, X_test, y_test, 10)


# ## Fine Tuning

# In[ ]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'reg_alpha': [0, 0.25, 0.5, 0.75],
        'reg_lambda': [0.45, 0.75 , 1]
        }


# In[ ]:


folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)


# In[ ]:


random_search = RandomizedSearchCV(xgb, 
                                   param_distributions=params, 
                                   n_iter=param_comb, 
                                   scoring='neg_mean_squared_error', 
                                   n_jobs=1, 
                                   cv=skf.split(features[top_feats], label), 
                                   verbose=3, 
                                   random_state=1001 )


# In[ ]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[ ]:


start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(features[top_feats], label)
timer(start_time) # timing ends here for "start_time" variable


# In[ ]:


print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)

results = pd.DataFrame(random_search.cv_results_)


# In[ ]:


results


# ## Export Model

# In[ ]:


joblib.dump(clf, 'bsmt_clf.pkl')
joblib.dump(linreg, 'lin_reg_1.pkl')
joblib.dump(forest_reg, 'forest_reg_1.pkl')
joblib.dump(xgb, 'xgb_1.pkl')
joblib.dump(random_search.best_estimator_, 'xgb_best.pkl')


# ## Generate Ouput CSV for Kaggle

# In[ ]:


def predict_export_csv(model, feature, index, file_name):
    temp_output = model.predict(feature)
    temp_output_df = pd.DataFrame({'Id': index, 'SalePrice': temp_output })
    temp_output_df.to_csv('{}.csv'.format(file_name), sep=',', encoding='utf-8', index=False)


# In[ ]:


predict_export_csv(random_search, test_processed[top_feats], test['Id'], 'house_prediction')


# In[ ]:




