#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques
# 
# Target: Predict the sales price for each house.
# 
# Source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques#description

# In[ ]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from ml_metrics import rmse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import learning_curve
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from functools import partial
from sklearn.metrics import make_scorer
import lightgbm as lgb
import catboost as ctb


# In[ ]:


np.random.seed(2018)


# In[ ]:


# display data
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 1500)


# In[ ]:


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_test['SalePrice'] = 0


# ### Data description
#     
#    - SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
#    - MSSubClass: The building class
#    - MSZoning: The general zoning classification
#    - LotFrontage: Linear feet of street connected to property
#    - LotArea: Lot size in square feet
#    - Street: Type of road access
#    - Alley: Type of alley access
#    - LotShape: General shape of property
#    - LandContour: Flatness of the property
#    - Utilities: Type of utilities available
#    - LotConfig: Lot configuration
#    - LandSlope: Slope of property
#    - Neighborhood: Physical locations within Ames city limits
#    - Condition1: Proximity to main road or railroad
#    - Condition2: Proximity to main road or railroad (if a second is present)
#    - BldgType: Type of dwelling
#    - HouseStyle: Style of dwelling
#    - OverallQual: Overall material and finish quality
#    - OverallCond: Overall condition rating
#    - YearBuilt: Original construction date
#    - YearRemodAdd: Remodel date
#    - RoofStyle: Type of roof
#    - RoofMatl: Roof material
#    - Exterior1st: Exterior covering on house
#    - Exterior2nd: Exterior covering on house (if more than one material)
#    - MasVnrType: Masonry veneer type
#    - MasVnrArea: Masonry veneer area in square feet
#    - ExterQual: Exterior material quality
#    - ExterCond: Present condition of the material on the exterior
#    - Foundation: Type of foundation
#    - BsmtQual: Height of the basement
#    - BsmtCond: General condition of the basement
#    - BsmtExposure: Walkout or garden level basement walls
#    - BsmtFinType1: Quality of basement finished area
#    - BsmtFinSF1: Type 1 finished square feet
#    - BsmtFinType2: Quality of second finished area (if present)
#    - BsmtFinSF2: Type 2 finished square feet
#    - BsmtUnfSF: Unfinished square feet of basement area
#    - TotalBsmtSF: Total square feet of basement area
#    - Heating: Type of heating
#    - HeatingQC: Heating quality and condition
#    - CentralAir: Central air conditioning
#    - Electrical: Electrical system
#    - 1stFlrSF: First Floor square feet
#    - 2ndFlrSF: Second floor square feet
#    - LowQualFinSF: Low quality finished square feet (all floors)
#    - GrLivArea: Above grade (ground) living area square feet
#    - BsmtFullBath: Basement full bathrooms
#    - BsmtHalfBath: Basement half bathrooms
#    - FullBath: Full bathrooms above grade
#    - HalfBath: Half baths above grade
#    - Bedroom: Number of bedrooms above basement level
#    - Kitchen: Number of kitchens
#    - KitchenQual: Kitchen quality
#    - TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#    - Functional: Home functionality rating
#    - Fireplaces: Number of fireplaces
#    - FireplaceQu: Fireplace quality
#    - GarageType: Garage location
#    - GarageYrBlt: Year garage was built
#    - GarageFinish: Interior finish of the garage
#    - GarageCars: Size of garage in car capacity
#    - GarageArea: Size of garage in square feet
#    - GarageQual: Garage quality
#    - GarageCond: Garage condition
#    - PavedDrive: Paved driveway
#    - WoodDeckSF: Wood deck area in square feet
#    - OpenPorchSF: Open porch area in square feet
#    - EnclosedPorch: Enclosed porch area in square feet
#    - 3SsnPorch: Three season porch area in square feet
#    - ScreenPorch: Screen porch area in square feet
#    - PoolArea: Pool area in square feet
#    - PoolQC: Pool quality
#    - Fence: Fence quality
#    - MiscFeature: Miscellaneous feature not covered in other categories
#    - MiscVal: $Value of miscellaneous feature
#    - MoSold: Month Sold
#    - YrSold: Year Sold
#    - SaleType: Type of sale
#    - SaleCondition: Condition of sale
# 

# ### Info about data
# 
# I used following function in orger to get some important info about data:
# 
#     - info(): some help information
#     - shape: method, which describe shape of dataset
#     - describe(): compte descriptive statistics
#     - nunique():  number of unique elements
#     - isnull(): search missing value
#     - sample(n): return n random row
#     - corr(): compute correlation between columns

# In[ ]:


df_train.head(5)


# In[ ]:


df_train.info()

Columns: 81
Object: 1460
Dtype: float64(3), int64(35), object(43)
NaN: Yes
Target variale: SalePrice
Problem: Regression
Memory require: 924.0 KB
Categorial variable: Yes
Amount variable: Yes
Ordinal variable: Yes
Date: Yes (year)
# In[ ]:


# delete Id variable, because it is unnecessary in compute
df_train = df_train.drop(['Id'], axis = 1)


# In[ ]:


df_train.shape


# In[ ]:


df_train.describe()


# In[ ]:


df_train.nunique()


# In[ ]:


df_train.isnull().any().any()


# In[ ]:


def missing_values(df):
    for column in df.columns:
        null_rows = df[column].isnull()
        if null_rows.any() == True:
            print('%s: %d nulls' % (column, null_rows.sum()))


# In[ ]:


missing_values(df_train)


# In[ ]:


df_train.sample(5)


# In[ ]:


plt.rcParams['figure.figsize']=(30,20)
sns.heatmap(df_train.corr(method='spearman'), annot=True, linewidths=.5, cmap="Blues");


# In[ ]:


# five most correlation variable with 'SalePrice'
corr = df_train.corr()
corr['SalePrice'].sort_values(ascending = False)[1:6]


# In[ ]:


corr['SalePrice'].sort_values(ascending = False)[-5:]


# In[ ]:


# get correlation matrix, where correlation value is greater than 70%
corr_matrix = df_train.corr(method='spearman')
corr_columns = corr_matrix[corr_matrix[corr_matrix > 0.7] < 1.0].any()
corr_matrix = corr_matrix[corr_columns][corr_columns.index[corr_columns]]
plt.rcParams['figure.figsize']=(15,7)
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap="Blues");


# In[ ]:


print(df_train.SalePrice.skew())
# A skewness value > 0 means, that there is more weight in the left tail of the distribution.


# ### Visualisation

# In[ ]:


# target variable
plt.rcParams['figure.figsize']=(10,5)
df_train['SalePrice'].hist();


# In[ ]:


df_train['SalePrice_bc'], _ = stats.boxcox(df_train['SalePrice'])
df_train['SalePrice_bc'].hist();


# In[ ]:


df_train['SalePrice_log'] = np.log2( df_train['SalePrice'] + 1)


# In[ ]:


df_train['SalePrice_log'].hist();


# In[ ]:


print(np.log2(df_train.SalePrice).skew())
# Normally distributed data, the skewness should be about 0


# The varable 'SalePrice' have skewed distribution, so I need transforn this variable, because Linear Regresin require it. I used log2. Also I tried use boxcox in order to comparison two transforn function.
# After I used log2 or boxcox I get normalize distribution.

# In[ ]:


def good_feats(df):
    feats_from_df = set(df.select_dtypes([np.int]).columns.values)
    bad_feats = {'SalePrice', 'SalePrice_bc'}
    return list(feats_from_df - bad_feats)


# In[ ]:


def make_hist(df):
    feats = good_feats(df)
    for index, feat in enumerate(feats):
        plt.subplot(len(feats)/5+1, 5, index+1);
        plt.title(feat);
        df[feat].hist();


# In[ ]:


def make_scatter(df):
    feats = good_feats(df)
    for index, feat in enumerate(feats):
        plt.subplot(len(feats)/5+1, 5, index+1)
        sns.regplot(x=feat, y='SalePrice', data=df_train)


# In[ ]:


def make_bar(df):
    cat_feats = df_train.select_dtypes(exclude = [np.int, np.float]).columns.values
    cat_feats = cat_feats[:-1]
    for index, feat in enumerate(cat_feats):
        plt.subplot(len(cat_feats)/5+1, 5, index+1)
        sns.barplot(x=feat, y='SalePrice', data=df_train, palette="PRGn");
        plt.xticks(rotation=90);


# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(25,25));\nplt.subplots_adjust(hspace = 0.35);\n\nmake_hist(df_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(25,25))\nplt.subplots_adjust(hspace = 0.3)\n\nmake_scatter(df_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(20,40))\nplt.subplots_adjust(hspace = 0.5, wspace = 0.4)\n\nmake_bar(df_train)')


# In[ ]:


df_train.Neighborhood.head(10)


# In[ ]:


plt.rcParams['figure.figsize']=(15,8)
sns.boxplot(x='Neighborhood', y='SalePrice', data=df_train, palette="PRGn");
plt.xticks(rotation=20);


# ### Fix NaN variable

# In[ ]:


missing_values(df_train)


# In[ ]:


df_train['LotFrontage'].head(5)


# In[ ]:


LFbyN = df_train.groupby('Neighborhood')['LotFrontage'].median().to_dict()
df_train['LotFrontage'] = df_train.apply(lambda row: LFbyN[row['Neighborhood']]
                                                    if pd.isnull(row['LotFrontage'])
                                                    else row['LotFrontage'], axis = 1)


# In[ ]:


df_train['Electrical'] = df_train['Electrical'].fillna('SBrkr')


# In[ ]:


cats_nan = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',  'BsmtFinType1', 'BsmtFinType1', 'BsmtFinType2', 'GarageYrBlt']
for cat in cats_nan:
    df_train[cat] = df_train[cat].fillna("None")


# In[ ]:


missing_values(df_train)


# ### Transform categorial variable

# In[ ]:


cat_feats = df_train.select_dtypes(exclude = [np.number]).columns.values
cat_feats


# In[ ]:


def factorize(df, *columns):
    feats = set(df.select_dtypes(exclude = [np.int, np.float]).columns.values)
    for column in feats:
        df[column + '_cat'] = pd.factorize(df[column])[0]


# In[ ]:


#factorize(df_train)


# In[ ]:


#df_train = df_train.select_dtypes(include=[np.number]).interpolate().dropna()
#df_train.head(5)


# In[ ]:


#df_train.sample(5)


# ### Basic model

# In[ ]:


def model_train_predict(model, X, y, success_metric=rmse):
    print('split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print('fit')
    model.fit(X_train, y_train)
    print('pred')
    y_pred = model.predict(X_test)
    return success_metric(y_test, y_pred)


# In[ ]:


X = df_train[good_feats(df_train)].values
y = df_train['SalePrice']


# In[ ]:


model_train_predict(LinearRegression(), X, y)


# ### Feature engineering
# 
# To feature engineering I choose feature based on correlation and visualisation. These are: OverallQual, GrLivArea, GarageCars, GarageArea, YearRemodeAdd, TotalBsmtSF, 1stFirSF, FullBath, TotRmsAbvGrd, YearBuilt, Street, SalesCondition, BsmtQual, KitchenQual, CentralAir, RoofMatl.

# In[ ]:


# the most correlaated variable with target
df_train.OverallQual.isnull().any()


# In[ ]:


df_train.OverallQual.nunique()


# In[ ]:


df_train.OverallQual.unique()


# In[ ]:


df_train.OverallQual.value_counts()


# In[ ]:


df_train.OverallQual.hist();


# In[ ]:


plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice']);


# In[ ]:


df_train = df_train[df_train['GrLivArea'] < 4000]
df_test = df_test[df_test['GrLivArea'] < 4000]


# In[ ]:


df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])


# In[ ]:


# GarageArea - correlation with target: 0.623431
df_train.GarageArea.head(5)


# In[ ]:


plt.scatter(x=df_train['GarageArea'], y=df_train['SalePrice']);


# In[ ]:


# GarageCars - correlation with target: 0.640409
df_train.GarageCars.head(5)


# In[ ]:


plt.scatter(x=df_train['GarageCars'], y=df_train['SalePrice']);


# In[ ]:


df_train[['GarageCars','GarageArea']].sample(10)


# In[ ]:


df_train = df_train[df_train['GarageArea'] < 1200]
df_test = df_test[df_test['GarageArea'] < 1200]
#df_train = df_train[df_train['GarageCars'] <= 3]
#df_test = df_test[df_test['GarageCars'] <= 3]


# In[ ]:


df_train.TotalBsmtSF.sample(5)


# In[ ]:


plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice']);


# In[ ]:


df_train = df_train[df_train['TotalBsmtSF'] < 3000]
df_test = df_test[df_test['TotalBsmtSF'] < 3000]


# In[ ]:


df_train.FullBath.sample(5)


# In[ ]:


plt.scatter(x=df_train['FullBath'], y=df_train['SalePrice']);


# In[ ]:


df_train.YearBuilt.sample(5)


# In[ ]:


plt.scatter(x=df_train['YearBuilt'], y=df_train['SalePrice']);


# In[ ]:


# street 
df_train['Street'].value_counts()


# Street is a feature, which contains information about type of road access. It has two type: pave and gravel.

# In[ ]:


factorize(df_train, 'Street')
factorize(df_test, 'Street')


# In[ ]:


# sale condition
df_train['SaleCondition'].value_counts()


# In[ ]:


sns.barplot(x='SaleCondition', y='SalePrice', data=df_train, palette="PRGn");


# In dataset usually occurs type of SaleCondition is Normal type. However we must see another dependence. Namely what effect the type has on the target variable. Supreme dependence has Partial variable. Other types have the some dependence on target.

# In[ ]:


df_train['SaleCondition'] = df_train['SaleCondition'].apply(lambda x: 1 if x == 'Partial' else 0)


# In[ ]:


sns.barplot(x='SaleCondition', y='SalePrice', data=df_train, palette="PRGn");


# In[ ]:


df_train['SaleCondition'].value_counts()


# In[ ]:


df_train['BsmtQual'].value_counts()


# In[ ]:


sns.barplot(x='BsmtQual', y='SalePrice', data=df_train, palette="PRGn");


# In[ ]:


df_train['BsmtQual'] = df_train['BsmtQual'].apply(lambda x: 1 if x == 'Ex' else 0)
df_test['BsmtQual'] = df_test['BsmtQual'].apply(lambda x: 1 if x == 'Ex' else 0)


# In[ ]:


sns.barplot(x='BsmtQual', y='SalePrice', data=df_train, palette="PRGn");


# In[ ]:


df_train['KitchenQual'].value_counts()


# In[ ]:


sns.barplot(x='KitchenQual', y='SalePrice', data=df_train, palette="PRGn");


# In[ ]:


df_train['KitchenQual'] = df_train['KitchenQual'].apply(lambda x: 1 if x == 'Ex' else 0)
df_test['KitchenQual'] = df_test['KitchenQual'].apply(lambda x: 1 if x == 'Ex' else 0)


# In[ ]:


sns.barplot(x='KitchenQual', y='SalePrice', data=df_train, palette="PRGn");


# In[ ]:


df_train['CentralAir'].value_counts()


# In[ ]:


factorize(df_train, 'CentralAir')
factorize(df_test, 'CentralAir')


# In[ ]:


df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']


# In[ ]:


plt.scatter(x=df_train['TotalSF'], y=df_train['SalePrice']);


# In[ ]:


df_train.TotalSF.hist();


# In[ ]:


features = ['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalSF', 'CentralAir_cat', 'KitchenQual_cat']


# In[ ]:


X = df_train[features].values
y = df_train['SalePrice_log']


# In[ ]:


model_train_predict(LinearRegression(), X, y)


# In[ ]:


feats = ['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalSF', 'CentralAir_cat', 'KitchenQual_cat']

X = df_train[feats].values
y = df_train['SalePrice_log'].values

model = LinearRegression()

cv = KFold(n_splits=4)

scores = []
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = rmse(y_test, y_pred)
    scores.append(score)
    
    
print(np.mean(scores), np.std(scores))


# In[ ]:


def run_cv(model, X, y, folds=4, target_log=False, cv_type=KFold, success_metric=rmse):
    cv = cv_type(n_splits=folds)
    
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if target_log:
            y_train = np.log(y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if target_log: 
            y_pred = np.exp(y_pred)
        y_pred[y_pred < 0] = 0

        score = success_metric(y_test, y_pred)
        scores.append(score)
        
    return np.mean(scores), np.std(scores)


# In[ ]:


run_cv(model, X, y, folds=3, target_log='SalePrice_log')


# In[ ]:


def plot_learning_curve(model, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    rmse_scorer = make_scorer(rmse)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=rmse_scorer)
    
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


models = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=10),
    RandomForestRegressor(max_depth=10),
    ExtraTreesRegressor(max_depth=20)
]

for model in models:
    print(str(model) + ": ")
    get_ipython().run_line_magic('time', 'score = model_train_predict(model, X, y)')
    print(str(score) + "\n")
    plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.4, 0.0), n_jobs=4)
    plt.show()


# In[ ]:


model = LinearRegression()
model.fit(X, y)

weights = list(model.coef_)

dict_feats = {label :weight for label, weight in zip(feats, weights) }
feats = pd.DataFrame([dict_feats])
feats.plot(kind='bar', figsize=(13, 8), title="Feature importances");


# In[ ]:


models = [
    DecisionTreeRegressor(max_depth=10),
    RandomForestRegressor(max_depth=10),
    ExtraTreesRegressor(max_depth=20)
]

for model in models:
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title('Feature importances: ' + str(model).split('(')[0])
    plt.bar(range(X.shape[1]), model.feature_importances_[indices],
           color = '#00bfff', align = 'center')
    plt.xticks(range(X.shape[1]), [ good_feats(df_train)[x] for x in indices])
    plt.xticks(rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


# In[ ]:


feats = ['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalSF', 'CentralAir_cat', 'KitchenQual_cat']
X = df_train[feats].values
y = df_train['SalePrice_log'].values

run_cv(xgb.XGBRegressor(), X, y, folds=4, target_log='SalePrice_log')


# In[ ]:


X = df_train[feats].values
y = df_train['SalePrice_log'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
def objective(space):    
    xgb_params = {
        'max_depth': int(space['max_depth']),
        'colsample_bytree': space['colsample_bytree'],
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'seed': int(space['seed']),
        'min_child_weight': int(space['min_child_weight']),
        'reg_alpha': space['reg_alpha'],
        'reg_lambda': space['reg_lambda'],
        'n_estimators': 150
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    means_score, std_score = run_cv(xgb.XGBRegressor(), X, y, target_log=True)

    print(means_score, std_score)
    
    print("SCORE: {0}".format(score))
    
    return{'loss':score, 'status': STATUS_OK }

space ={
    'max_depth': hp.quniform ('x_max_depth', 3, 8, 1),
    'colsample_bytree': hp.uniform ('x_colsample_bytree', 0.3, 0.7),
    'learning_rate': hp.uniform ('x_learning_rate', 0.1, 0.3), 
    'subsample': hp.uniform ('x_subsample', 0.3, 0.7),
    'seed': hp.quniform ('x_seed', 0, 10000, 50),
    'min_child_weight': hp.quniform ('x_min_child_weight', 1, 10, 1),
    'reg_alpha': hp.loguniform ('x_reg_alpha', 0., 0.1),
    'reg_lambda': hp.uniform ('x_reg_lambda', 0.9, 1.),
    'n_estimators': hp.quniform ('x_n_estimators', 50, 300, 10)
}

trials = Trials()
best_params = fmin(fn=objective,
            space=space,
            algo=partial(tpe.suggest, n_startup_jobs=1),
            max_evals=100,
            trials=trials)

print("The best params: ", best_params)


# In[ ]:


X = df_test[features].values

df_test['SalePrice'] = model.predict(X)
df_test[ ['Id','SalePrice'] ].to_csv('../lr.csv', index=False)

