#!/usr/bin/env python
# coding: utf-8

# Dataset webpage: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# ### Import libraries

# In[ ]:


import pandas as pd
from pandas.api.types import CategoricalDtype

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import scipy.stats as stats
from scipy.stats import norm
from scipy.special import boxcox1p

import statsmodels
import statsmodels.api as sm
#print(statsmodels.__version__)

import collections
import math

from sklearn.preprocessing import scale, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample

from xgboost import XGBRegressor
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ### Import data

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# # EDA (Part I)

# ### Distribution of the response

# We start by looking at how the response (Sale Price) is distributed.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(18, 6))
sns.distplot(train_data['SalePrice'], ax=axes[0]);
sm.qqplot(train_data['SalePrice'], stats.norm, fit=True, line='45', ax=axes[1]);


# In[ ]:


print('This distribution is far from normal. In particular, it has a high skew ({:.4f}).'.format(train_data['SalePrice'].skew()))
print('A log-transformation of Sale Price generates a distribution that is much closer to a Gaussian.')


# In[ ]:


train_data['log1pSalePrice'] = np.log1p(train_data['SalePrice'])
fig, axes = plt.subplots(1,2, figsize=(18, 6))
sns.distplot(train_data['log1pSalePrice'], ax=axes[0]);
axes[0].set_xlabel('log(1+SalePrice)')
sm.qqplot(train_data['log1pSalePrice'], stats.norm, fit=True, line='45', ax=axes[1]);


# ### SalePrice vs. Living Area

# Next we look at the relationship between Sale Price and Living Area. Typically a more expensive house is sold for more.

# In[ ]:


plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c='blue', marker='s')
plt.title('In search of outliers...')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')


# We see that there are a few outliers: houses with large living area that sold at a low price. We remove these points. We also focus on Normal Sales only. With these modifications, we replot the relationship between Sale Price and Living Area.

# In[ ]:


train_data = train_data[train_data['GrLivArea'] < 4500]
train_data = train_data[train_data['SaleCondition'] == 'Normal']


# In[ ]:


plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c='blue', marker='s')
plt.title('In search of outliers...')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')


# In[ ]:


corrmatrix = train_data.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmatrix, vmax=0.8, square=True)


# In[ ]:


k = 10
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# We will bin the Neighborhood variable by median SalePrice.

# In[ ]:


Neighborhood = train_data.groupby('Neighborhood')
Neighborhood['SalePrice'].median()


# We drop the ID column since we don't need it. We save it, however, since it will be needed later.

# In[ ]:


ID_train = train_data['Id']
ID_test = test_data['Id']


# We are looking to remove multicollinearity at this stage. There are four sets of variables which contain roughly the same information:
# 1. TotalBsmntSF and 1stFlrSF. High correlation because basement sits below 1st floor.
# 2. GarageCars and GarageArea. Very high correlation because the bigger the area the more cars can fit in.
# 3. GarageYrBlt and YearBuilt. Very highly correlated - usually the garage is built at the same time as the house.
# 4. TotRmsAbvGrd and GrLivArea. Very highly correlated. The more area the more rooms there are.
# 
#  We drop one variable from each pairing, the one whose correlation with the response is smaller.

# In[ ]:


train_data.drop("Id", axis=1, inplace=True)
test_data.drop("Id", axis=1, inplace=True)
train_data.drop("TotRmsAbvGrd", axis=1, inplace=True)
test_data.drop("TotRmsAbvGrd", axis=1, inplace=True)
train_data.drop("GarageYrBlt", axis=1, inplace=True)
test_data.drop("GarageYrBlt", axis=1, inplace=True)
train_data.drop("GarageArea", axis=1, inplace=True)
test_data.drop("GarageArea", axis=1, inplace=True)
train_data.drop("1stFlrSF", axis=1, inplace=True)
test_data.drop("1stFlrSF", axis=1, inplace=True)


# At this point we separately save the log-transformed SalePrice column and drop it from the features dataframe.

# In[ ]:


y = train_data['log1pSalePrice']
train_data.drop(['SalePrice','log1pSalePrice'], axis=1, inplace=True)


# We are going to combine the training and test data in order to do cleaning. For this reason we need to get the number of records in each set and set a variable that identifies the split point of the combined dataset.

# In[ ]:


#print(train_data.shape)
#print(test_data.shape)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
print("Number of training (testing) examples: {} ({})".format(ntrain, ntest))


# In[ ]:


Combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)
print("Thecombined dataset has size: {}".format(Combined_data.shape))


# We will now deal with missing values. We first look at the numbers of missing values for each feature.

# In[ ]:


total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent']).sort_values('Total', ascending=False)
missing_data.head(40)


# There are three columns where most of the data is missing and it's irrelevant to the response (Sale Price of a home). We drop these.

# In[ ]:


Combined_data.drop('PoolQC', axis=1, inplace=True)
Combined_data.drop('MiscFeature', axis=1, inplace=True)
Combined_data.drop("Alley", axis=1, inplace=True)


# In a number of categories, there is an obvious way to impute a missing value. For example, NA in the 'GarageFinish' column likely means that the propery doesn't have a garage.

# In[ ]:


# Lot Frontage: NA most likely means no lot frontage
Combined_data['LotFrontage'].fillna(0, inplace=True)

# Fence: NA most likely means no fence
Combined_data['Fence'].fillna('None', inplace=True)

# FireplaceQu: NA means no fireplace
Combined_data['FireplaceQu'].fillna('None', inplace=True)

# GarageCond: NA means no garage
Combined_data['GarageCond'].fillna('None', inplace=True)
# GarageFinish: NA means no garage
Combined_data['GarageFinish'].fillna('None', inplace=True)
# GarageQual: NA means no garage
Combined_data['GarageQual'].fillna('None', inplace=True)
# GarageType: NA means no garage
Combined_data['GarageType'].fillna('None', inplace=True)
# BsmtFinType1: NA means no basement
Combined_data["BsmtFinType1"].fillna('None', inplace=True)
# BsmtFinType2: NA means no basement
Combined_data["BsmtFinType2"].fillna('None', inplace=True)
#BsmtExposure: NA means no basement
Combined_data['BsmtExposure'].fillna("None", inplace=True)
#BsmtQual: NA means no basement
Combined_data['BsmtQual'].fillna('None', inplace=True)
#BsmtCond: NA means no basement
Combined_data['BsmtCond'].fillna('None', inplace=True)
# MasVnrType: NA means none
Combined_data['MasVnrType'].fillna('None', inplace=True)
# MasVnrAreaL NA most likely means 0
Combined_data['MasVnrArea'].fillna(0, inplace=True)
# MasVnrAreaL NA most likely means 0
Combined_data['Electrical'].fillna('SBrkr', inplace=True)
# BsmtHalfBath: NA most likely means 0
Combined_data['BsmtHalfBath'].fillna(0, inplace=True)
# BsmtFullBath: NA most likely means 0
Combined_data['BsmtFullBath'].fillna(0, inplace=True)
# BsmtFinSF1: NA most likely means 0
Combined_data['BsmtFinSF1'].fillna(0, inplace=True)
# BsmtFinSF2: NA most likely means 0
Combined_data['BsmtFinSF2'].fillna(0, inplace=True)
# BsmtUnfinSF: NA most likely means 0
Combined_data['BsmtUnfSF'].fillna(0, inplace=True)
# TotalBsmtSF: NA most likely means 0
Combined_data['TotalBsmtSF'].fillna(0, inplace=True)
# GarageCars: NA most likely means 0
Combined_data['GarageCars'].fillna(0, inplace=True)
# GarageArea: NA likely means 0
# Combined_data['GarageArea'].fillna(0, inplace=True)
# Basement Condition: NA likely means no basement
Combined_data['Utilities'].fillna('ELO', inplace=True)
# Basement Condition: NA likely means no basement
Combined_data['Functional'].fillna('No', inplace=True)
# Basement Condition: NA likely means no basement
Combined_data['KitchenQual'].fillna('Po', inplace=True)
# MSZoning (general Zoning Classification): RL is the most common value
Combined_data['MSZoning'].fillna('RL', inplace=True)
# Sale Type: fill in with the most frequent value ('WD')
Combined_data['SaleType'].fillna('WD', inplace=True)

#Exterior 1 and Exterior2: fill in with most frequent value
Combined_data['Exterior1st'] = Combined_data['Exterior1st'].fillna(Combined_data['Exterior1st'].mode()[0])
Combined_data['Exterior2nd'] = Combined_data['Exterior2nd'].fillna(Combined_data['Exterior2nd'].mode()[0])


# We notice that some features are encoded as ordinal categories even though there is no meaning to their ordering. We switch them back to a nominal category representation.

# In[ ]:


Combined_data = Combined_data.replace(
    {
        "MSSubClass": {20: "SC20", 30: "SC30" , 40: "SC40", 50: "SC50", 60: "SC60", 70: "SC70", 80: "SC80",
                        90: "SC90", 100: "SC100", 110: "SC110", 120: "SC120", 130: "SC130", 140: "SC140", 150: "SC150", 160: "SC160",
                         170: "SC170", 180: "SC180", 190: "SC190"},
        "MoSold":{1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    }
)


# Conversely, there are some featurees encoded as nominal categories even though their ordering carries information. We switch them back to an ordinal category representation.

# In[ ]:


Combined_data.BsmtCond = pd.factorize(Combined_data.BsmtCond.astype(CategoricalDtype(categories=["None", "Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.BsmtExposure = pd.factorize(Combined_data.BsmtExposure.astype(CategoricalDtype(categories=['None', 'No', "Mn", "Av", "Gd"], ordered=True)))[0]
Combined_data.BsmtFinType1 = pd.factorize(Combined_data.BsmtFinType1.astype(CategoricalDtype(categories=['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], ordered=True)))[0]
Combined_data.BsmtFinType2 = pd.factorize(Combined_data.BsmtFinType2.astype(CategoricalDtype(categories=['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], ordered=True)))[0]
Combined_data.BsmtQual = pd.factorize(Combined_data.BsmtQual.astype(CategoricalDtype(categories=["None", "Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.ExterCond = pd.factorize(Combined_data.ExterCond.astype(CategoricalDtype(categories=["None", "Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.ExterQual = pd.factorize(Combined_data.ExterQual.astype(CategoricalDtype(categories=["None", "Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.FireplaceQu = pd.factorize(Combined_data.FireplaceQu.astype(CategoricalDtype(categories=["None", "Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.Functional = pd.factorize(Combined_data.Functional.astype(CategoricalDtype(categories=['No','Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], ordered=True)))[0]
Combined_data.GarageCond = pd.factorize(Combined_data.GarageCond.astype(CategoricalDtype(categories=['None', 'Po', 'Fa', 'TA', "Gd", "Ex"], ordered=True)))[0]
Combined_data.GarageFinish = pd.factorize(Combined_data.GarageFinish.astype(CategoricalDtype(categories=['None', 'Unf', 'RFn', 'Fin'], ordered=True)))[0]
Combined_data.GarageQual = pd.factorize(Combined_data.GarageQual.astype(CategoricalDtype(categories=["None", "Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.HeatingQC = pd.factorize(Combined_data.HeatingQC.astype(CategoricalDtype(categories=["Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.KitchenQual = pd.factorize(Combined_data.KitchenQual.astype(CategoricalDtype(categories=["Po", "Fa", "TA", 'Gd', "Ex"], ordered=True)))[0]
Combined_data.LandSlope = pd.factorize(Combined_data.LandSlope.astype(CategoricalDtype(categories=['Sev', "Mod", "Gtl"], ordered=True)))[0]
Combined_data.LotShape = pd.factorize(Combined_data.LotShape.astype(CategoricalDtype(categories=['IR3', "IR2", "IR1", "Reg"], ordered=True)))[0]
Combined_data.PavedDrive = pd.factorize(Combined_data.PavedDrive.astype(CategoricalDtype(categories=['N', "P", "Y"], ordered=True)))[0]
Combined_data.Street = pd.factorize(Combined_data.Street.astype(CategoricalDtype(categories=['Grvl', "Pave"], ordered=True)))[0]
Combined_data.Utilities = pd.factorize(Combined_data.Utilities.astype(CategoricalDtype(categories=['ELO', "NoSeWa", "NoSewr", "AllPub"], ordered=True)))[0]


# Now we create new features. For example, we define a 'Total_Home_Quality' feature which is the sum of 'OverallQual'  and 'OverallCond'. To avoid multicollinearity, we drop the individual components of the combination from the dataframe.

# In[ ]:


Combined_data['Total_Home_Quality'] = Combined_data['OverallQual'] + Combined_data['OverallCond']
Combined_data['Total_Basement_Quality'] = Combined_data['BsmtQual'] + Combined_data['BsmtCond']
Combined_data['Total_Basement_Finished_SqFt'] = Combined_data['BsmtFinSF1'] + Combined_data['BsmtFinSF2']
Combined_data['Total_Exterior_Quality'] = Combined_data['ExterQual'] + Combined_data['ExterCond']
Combined_data['Total_Garage_Quality'] = Combined_data['GarageCond'] + Combined_data['GarageQual'] + Combined_data['GarageFinish']
Combined_data['Total_Basement_Finish_Type'] = Combined_data['BsmtFinType1'] + Combined_data['BsmtFinType2'] 
Combined_data['Total_Bathrooms'] = Combined_data['BsmtFullBath'] + (0.5 * Combined_data['BsmtHalfBath']) + Combined_data['FullBath'] + (0.5 * Combined_data['HalfBath'])
Combined_data['Total_Land_Quality'] = Combined_data['LandSlope'] + Combined_data['LotShape']

Combined_data.drop(['OverallQual','OverallCond','BsmtQual', 'BsmtCond', 'BsmtFinSF1',
                    'BsmtFinSF2', 'ExterQual', 'ExterCond', 'GarageCond', 'GarageQual', 'GarageFinish', 'BsmtFinType1', 
                    'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'LandSlope', 'LotShape'], 
                   axis=1, inplace=True)


# Finally we will bin the 'Neighborhood' feature into quartiles based on 'SalePrice'.

# In[ ]:


Combined_data = Combined_data.replace({'Neighborhood':{
        "MeadowV" : 0,"IDOTRR" : 0, "BrDale" : 0, "OldTown" : 0, "Edwards" : 0, "BrkSide" : 0, "Sawyer" : 0, 
        "Blueste" : 1, "SWISU" : 1, "NAmes" : 1, "NPkVill" : 1, "Mitchel" : 1, "SawyerW" : 1,
        "Gilbert" : 2, "NWAmes" : 2, "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 2, "Crawfor" : 2, 
        "Veenker" : 3, "Somerst" : 3, "Timber" : 3, "StoneBr" : 3, "NoRidge" : 3, "NridgHt" : 3} })


# We check that we have successfully dealt with all missing values. This is the case.

# In[ ]:


total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent']).sort_values('Total', ascending=False)
missing_data.head(10)


# We check for skew in the distributions of continous numerical features. If the distribution is skewed we normalize the variable with a log transformation.

# In[ ]:


Skewed_Feature_Check = ['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF',
                       'OpenPorchSF', 'PoolArea','MiscVal', 'Total_Basement_Finished_SqFt']
n = len(Skewed_Feature_Check)
#fig, ax = plt.subplots(3, 4, figsize=(36, 16))

for i  in range(n):
    feature = Skewed_Feature_Check[i]
    #sns.distplot(Combined_data[feature], kde=False, fit=norm, ax = ax[i][0]) 
    #sm.qqplot(Combined_data[feature], stats.norm, fit=True, line='45', ax=ax[i%3][i//3], label = feature);
    #ax[i%3][i//3].legend(fontsize=12)
    print('{:>15}: {:.6g}        {}'.format(feature, stats.skew(Combined_data[feature]), stats.skewtest(Combined_data[feature])))
    #from scipy.special import boxcox1p
    #lam = 0.15
    #Combined_data[feature] = boxcox1p(Combined_data[feature], lam)
    Combined_data[feature] = np.log1p(Combined_data[feature])
    #sns.distplot(Combined_data[feature], kde=False, fit=stats.norm, ax = ax[i][1]) 
    #sm.qqplot(Combined_data[feature], stats.norm, fit=True, line='45', ax=ax[i][1]);


# We now split into categorical and numerical features.

# In[ ]:


categorical_features = Combined_data.select_dtypes(include=["object"]).columns
numerical_features = Combined_data.select_dtypes(exclude=["object"]).columns
print('Numerical features: {}'.format(numerical_features.shape[0]))
print('Categorical features: {}'.format(categorical_features.shape[0]))

Combined_data_numerical = Combined_data[numerical_features]
Combined_data_categorical = Combined_data[categorical_features]


# We now look at the correlation matrix of numerical predictors.

# In[ ]:


corrmatrix_combined = Combined_data_numerical.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmatrix_combined, vmax=.8, square=True)


# In[ ]:


Combined_data_categorical = pd.get_dummies(Combined_data_categorical, drop_first=True)
Combined_data = pd.concat([Combined_data_categorical, Combined_data_numerical], axis=1)


# In[ ]:


train_data = Combined_data[:ntrain]
test_data = Combined_data[ntrain:]
test_data = test_data.reset_index(drop=True)
print("Number of training (testing) examples: {} ({})".format(train_data.shape[0], test_data.shape[0]))
print("Size of combined dataset is  {}".format(Combined_data.shape))


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.2, random_state=1)
print('X_train : {}'.format(X_train.shape))
print('X_val : {}'.format(X_val.shape))
print('y_train : {}'.format(y_train.shape))
print('X_val : {}'.format(y_val.shape))


# In[ ]:


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)


# # MODELLING

# ### Cross-validation scoring

# In[ ]:


n_folds = 5

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(X_train)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)


# In[ ]:


# run Cross-Validation Score on a basic model with no parameter Tuning

for Model in [LinearRegression, Ridge, Lasso, XGBRegressor, ElasticNet, RandomForestRegressor,  HuberRegressor, GaussianProcessRegressor, SVR, KernelRidge]:
    if Model == XGBRegressor: cv_res = rmse_cv(XGBRegressor(objective='reg:squarederror'))
    else: cv_res = rmse_cv(Model())
    print('{}: {:.5f} +/- {:5f}'.format(Model.__name__, -cv_res.mean(), cv_res.std()))


# ### Ridge Regression

# In[ ]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [-rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Ridge Regression Cross-Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

RR_best = Ridge(alpha = np.argmin(cv_ridge))
RR_best.fit(X_train, y_train)
predicted_prices = RR_best.predict(test_data)

my_submission = pd.DataFrame({'Id': ID_test, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)


# ### LASSO

# In[ ]:


alphas = [0.0001,0.0005,0.001, 0.005, 0.01,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_lasso = [-rmse_cv(Lasso(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Lasso Regression Cross-Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.xscale('log')

LASSO_best = Lasso(alpha = np.argmin(cv_ridge))
LASSO_best.fit(X_train, y_train)
predicted_prices = LASSO_best.predict(test_data)

my_submission = pd.DataFrame({'Id': ID_test, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_LASSO.csv', index=False)


# In[ ]:




