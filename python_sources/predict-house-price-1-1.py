#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler, MaxAbsScaler, PowerTransformer, RobustScaler, Normalizer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, validation_curve, learning_curve

from sklearn.pipeline import make_pipeline

import datetime


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df.head(5)


# In[ ]:


test_df.head(5)


# In[ ]:


df_na = train_df.isna().sum()
df_na = df_na[df_na > 0].sort_values(ascending=False)
df_na


# In[ ]:


df_na_test = test_df.isna().sum()
df_na_test = df_na_test[df_na_test > 0].sort_values(ascending=False)
df_na_test


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(15,7))
sns.barplot(x=df_na.index, y=df_na, ax=ax)
plt.xticks(rotation=90)


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(15,7))
sns.barplot(x=df_na_test.index, y=df_na_test, ax=ax)
plt.xticks(rotation=90)


# We can remove poolQC as there is very small amount of values present in that to predict                                                                                 

# In[ ]:


train_df.drop(['PoolQC'], axis=1, inplace=True)
test_df.drop(['PoolQC'], axis=1, inplace=True)


# In[ ]:


lst_none = ['MasVnrType', 'MiscFeature']


# In[ ]:


train_df[lst_none] = train_df[lst_none].fillna('None')
test_df[lst_none] = test_df[lst_none].fillna('None')


# In[ ]:


train_df_num = train_df.select_dtypes(exclude='object')
test_df_num = test_df.select_dtypes(exclude='object')


# In[ ]:


train_df_num = train_df_num.loc[:, ~ train_df_num.columns.isin(['Id', 'SalePrice'])]
test_df_num = test_df_num.loc[:, ~ test_df_num.columns.isin(['Id'])]


# In[ ]:


train_df_num.fillna(np.nan, inplace=True)
test_df_num.fillna(np.nan, inplace=True)


# In[ ]:


imp = SimpleImputer(missing_values=np.nan, strategy='median')


# In[ ]:


train_df[train_df_num.columns] = pd.DataFrame(imp.fit_transform(train_df_num), columns=train_df_num.columns)
test_df[test_df_num.columns] = pd.DataFrame(imp.fit_transform(test_df_num), columns=test_df_num.columns)


# In[ ]:


lst_na = ['Alley', 'Fence', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual']


# In[ ]:


train_df[lst_na] = train_df[lst_na].fillna('NA')
test_df[lst_na] = test_df[lst_na].fillna('NA')


# In[ ]:


train_df_cat = train_df.select_dtypes(include='object')
test_df_cat = test_df.select_dtypes(include='object')


# In[ ]:


train_df[train_df_cat.columns]= train_df_cat.fillna(train_df_cat.mode().iloc[0])
test_df[test_df_cat.columns] = test_df_cat.fillna(test_df_cat.mode().iloc[0])


# In[ ]:


df_na = train_df.isna().sum()
df_na = df_na[df_na > 0].sort_values(ascending=False)
df_na


# In[ ]:


df_na_test = test_df.isna().sum()
df_na_test = df_na_test[df_na_test > 0].sort_values(ascending=False)
df_na_test


# In[ ]:


def get_corr(df, threshhold):
    corr_mat = df.corr()
    target_corr_mat = abs(corr_mat['SalePrice'])
    target_corr_mat = target_corr_mat[target_corr_mat > threshhold]
    corr_mat = train_df.loc[:, target_corr_mat.index].corr()
    return corr_mat


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(20,20))
sns.heatmap(get_corr(train_df,0.2), annot=True, ax=ax)


# In[ ]:


train_df['TotBath'] = train_df['BsmtFullBath'] + train_df['BsmtHalfBath'] + train_df['FullBath'] + train_df['HalfBath']
test_df['TotBath'] = test_df['BsmtFullBath'] + test_df['BsmtHalfBath'] + test_df['FullBath'] + test_df['HalfBath']


# In[ ]:


train_df['Yr_old'] = datetime.datetime.now().year - train_df['YearBuilt']
test_df['Yr_old'] = datetime.datetime.now().year - test_df['YearBuilt']


# In[ ]:


train_df['Yr_last_remod'] = datetime.datetime.now().year -  train_df['YearRemodAdd']
test_df['Yr_last_remod'] = datetime.datetime.now().year -  test_df['YearRemodAdd']


# In[ ]:


train_df['Yr_garage_old'] = datetime.datetime.now().year -  train_df['GarageYrBlt']
test_df['Yr_garage_old'] = datetime.datetime.now().year -  test_df['GarageYrBlt']


# In[ ]:


train_df.drop(['GarageArea', 'FullBath' , 'HalfBath' ,'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YearRemodAdd', 'BedroomAbvGr', 'GarageYrBlt'], axis=1,inplace=True)
test_df.drop(['GarageArea', 'FullBath' , 'HalfBath' ,'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YearRemodAdd', 'BedroomAbvGr', 'GarageYrBlt'], axis=1,inplace=True)


# In[ ]:


corr_features = abs(get_corr(train_df,0.2))['SalePrice'].sort_values(ascending=False)
corr_features


# In[ ]:


target_feature = train_df['SalePrice']
test_id = test_df['Id']


# In[ ]:


train_df.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)


# In[ ]:


filtered_num_features = corr_features.index[~ corr_features.index.isin(['SalePrice'])]
filtered_num_features


# In[ ]:


filtered_cat_features = train_df_cat.columns
filtered_cat_features


# In[ ]:


condition_map = { 'Ex': 5, 'Gd' :  4, 'TA' :  3, 'Fa' :  2, 'Po' :  1, 'NA': 0 }
finishing_map = { 'Fin' : 3, 'RFn' : 2, 'Unf' : 1, 'NA' : 0}
func_map = { 'Typ' : 7, 'Min1' : 6, 'Min2' : 5, 'Mod'  : 4, 'Maj1' : 3, 'Maj2' : 2, 'Sev'  : 1, 'Sal'  : 0 }
fin_type_map = {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'NA' : 0 }
exposure_map = { 'Gd' : 4, 'Av' : 3, 'Mn' : 2, 'No' : 1, 'NA' : 0 }
house_style_map = { '1Story' : 1, '1.5Unf' :  2, '1.5Fin' : 3, '2Story' : 4, '2.5Unf' : 5, '2.5Fin' : 6, 'SFoyer' : 7, 'SLvl' : 8 }
slope_map = {'Gtl'  : 1, 'Mod'  : 2, 'Sev' :  3}


# In[ ]:


ordinal_cat_map = { 'ExterQual' : condition_map , 
                'ExterCond' : condition_map , 
                'BsmtQual'  : condition_map , 
                'BsmtCond'  : condition_map ,
                'KitchenQual' : condition_map,
                'FireplaceQu' : condition_map, 
                'GarageQual'  : condition_map, 
                'GarageCond'  : condition_map,
                'GarageFinish' : finishing_map,
                'Functional'  : func_map,
                'BsmtFinType2' : fin_type_map,
                'BsmtFinType1' : fin_type_map,
                'BsmtExposure' : exposure_map,                
                'ExterCond' : condition_map,
                'ExterQual' : condition_map,
                'HouseStyle' : house_style_map
              }


# In[ ]:


def apply_map(df,mappings):
    for key, val in mappings.items():
        df.loc[:, key] = df[key].map(val) 


# In[ ]:


apply_map(train_df,ordinal_cat_map)
apply_map(test_df,ordinal_cat_map)


# In[ ]:


nominal_col =  train_df.columns[~ train_df.columns.isin(filtered_num_features.tolist() + list(ordinal_cat_map.keys()))]
# process columns, apply LabelEncoder to categorical features
for c in nominal_col:
    lbl = LabelEncoder() 
    lbl.fit(list(train_df[c].values))
    train_df[c] = lbl.transform(list(train_df[c].values))
    lbl.fit(list(test_df[c].values))
    test_df[c] = lbl.transform(list(test_df[c].values))


# In[ ]:


train_df.head(10)


# In[ ]:


test_df.head(10)


# In[ ]:


sns.distplot(target_feature)


# In[ ]:


pt_y = PowerTransformer().fit(target_feature.to_frame())
target_feature = pt_y.transform(target_feature.to_frame())


# In[ ]:


sns.distplot(target_feature)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df.values, target_feature.ravel(), test_size=0.3, random_state=0)


# In[ ]:


model_linear = make_pipeline(PowerTransformer(), LinearRegression())
model_lasso = make_pipeline(PowerTransformer(), Lasso(alpha=0.0005, random_state=1))
model_ENet = make_pipeline(PowerTransformer(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
model_krr = make_pipeline(PowerTransformer(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
model_svr = make_pipeline(PowerTransformer(), SVR(gamma='scale', C=1.0, epsilon=0.2))
model_adaboost = AdaBoostRegressor()
model_gradient_boosting = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)
model_random_forest = RandomForestRegressor()
model_extra_tree  = ExtraTreesRegressor(n_jobs=1, random_state=0),
model_mlp = MLPRegressor(max_iter=500)


# In[ ]:


model_dict = {
    'Linear' : model_linear,
    'lasso' : model_lasso,
    'Enet' : model_ENet,
    'KRR' : model_krr,
    'SVR' : model_svr,
    'AdaBoost' : model_adaboost,
    'Gradient Boosting' : model_gradient_boosting,
    'Random Forest' : model_random_forest,
    'MLP' : model_mlp
}


# In[ ]:


def calc_cv_scores(models):
    score_df = pd.DataFrame()
    for key, model in models.items():
        score_df[key] = cross_val_score(model, X_train, y_train,cv=5)
    return score_df


# In[ ]:


score_df = calc_cv_scores(model_dict)


# In[ ]:


score_df


# In[ ]:


score_df.loc[:,['Random Forest', 'Gradient Boosting']].mean()


# In[ ]:


score_df.loc[:,['Random Forest', 'Gradient Boosting']].plot()


# In[ ]:


final_fitted_model = model_dict.get('Gradient Boosting').fit(X_train, y_train)


# In[ ]:


imp_series = pd.Series(final_fitted_model.feature_importances_, index=train_df.columns)
pred_contributiors = imp_series.loc[~(imp_series == 0)]


# In[ ]:


pred_contributiors.nlargest(20).plot(kind='barh')


# In[ ]:


test_error = mean_squared_error(y_test, final_fitted_model.predict(X_test))


# In[ ]:


test_error


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


title = "Learning Curves (Gradient Boosting)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator =  model_dict.get('Gradient Boosting')
plot_learning_curve(estimator, title, train_df.values, target_feature.ravel(), ylim=(0.7, 1.01), cv=5, n_jobs=4, train_sizes=[50, 100, 500, 700, 1000])


# In[ ]:


test_data_predition = final_fitted_model.predict(test_df)


# In[ ]:


test_id.shape, test_data_predition.shape


# In[ ]:


pred_df = pd.DataFrame(test_data_predition, columns=["SalePrice"])
final_predictions = pt_y.inverse_transform(pred_df)


# In[ ]:


final_predictions.ravel()


# In[ ]:


my_submission = pd.DataFrame({'Id' : test_id, 'SalePrice' : final_predictions.ravel()})
my_submission.to_csv('my_submission.csv', index=False)

