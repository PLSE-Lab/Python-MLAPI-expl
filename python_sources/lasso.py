#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# # Read data and understand the variable

# In[ ]:


dfTrain = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
dfTest = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
with open('../input/house-prices-advanced-regression-techniques/data_description.txt') as f:
    description = f.read()
print('Train shape {}'.format(dfTrain.shape))
print('Test shape {}'.format(dfTest.shape))


# # Outlier
# ## Find outlier by visualizing scatter plot between target variable and highly correlated variables

# In[ ]:


# plt.figure(figsize=[10, 7])
# sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data=dfTrain)
# plt.title('Scatter plot of SalePrice and Living area above ground(before removing outliers)')


# In[ ]:


# dfTrain = dfTrain.drop(dfTrain['GrLivArea'].sort_values(ascending = False)[:2].index, axis = 0)
# plt.figure(figsize=[10, 7])
# sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data=dfTrain)
# plt.title('Scatter plot of SalePrice and Living area above ground(after removing outliers)')


# In[ ]:


cols = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(dfTrain[cols], diag_kind='kde')
dfTrain = dfTrain.drop(dfTrain['GrLivArea'].sort_values(ascending = False)[:2].index, axis = 0)
# dfTrain = dfTrain.drop(dfTrain['GrLivArea'][dfTrain['GrLivArea'] >= 4000].index, axis = 0)
# sns.pairplot(dfTrain[cols], diag_kind='kde')

# not sure the reason for this yet, saw this in other kernels
outliers = [30, 88, 462, 631, 1322]
dfTrain = dfTrain.drop(outliers, axis = 0)


# In[ ]:


columnsTrain = dfTrain.columns
df = pd.concat([dfTrain, dfTest]).reset_index(drop = True)
dfTrainId = dfTrain['Id']
dfTestId = dfTest['Id']
dfTrainSalePrice = df['SalePrice'][df['SalePrice'].notnull()]
nTrain = dfTrain.shape[0]
dropVars = ['Id', 'SalePrice']
df = df.drop(labels=dropVars, axis=1)


# # Verify the logic of data
# ## Year variable shouldn't exceed 2019
# ## Areas, distances, prices shouldn't be less than 0
# ## Month should be between 1 and 12

# In[ ]:


years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
         'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
# GarageYrBlt has some very high values
df['GarageYrBlt'][df['GarageYrBlt'] > 2019] = df['YearBuilt'][df['GarageYrBlt'] > 2019]


# # Verify each variable type matches the actual meaning

# ## Verify the type of the variables

# In[ ]:


dtypes = df.dtypes
columnNames = df.columns
dtypes_columnNames = zip(dtypes, columnNames)
for dtype, columnName in dtypes_columnNames:
    print(columnName, ':', dtype, end = ' | ')


# In[ ]:


# MSSubClass and MoSold should be categorical variable instead of numerical, 
num2cate = ['MSSubClass', 'MoSold']
for name in num2cate:
    df[name] = df[name].map(lambda x: str(x))


# # Missing value

# In[ ]:


totalMissing = df.isnull().sum().sort_values(ascending = False)
percentMissing = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
missing = pd.concat([totalMissing, percentMissing], axis = 1)
missing.columns = ['total', 'percent']
missing


# In[ ]:


test = missing.reset_index().rename(columns = {'index': 'variable name'})


# In[ ]:


test.loc[test['total'] > 5]


# In[ ]:


plt.figure(figsize=[10, 7])
sns.barplot(y = 'variable name', x = 'total', data = test.loc[test['total'] > 5])
plt.title('Barplot of variables with missing values')


# Missing in some varaibles means there is no such facilities
# 
# Missing in some variables means missing data
# 

# In[ ]:


# two non missing values in garage type is special, they have garage type as detached
missing_garage_index = df[df['GarageType'].notnull() & df['GarageQual'].isnull()].index
df['GarageYrBlt'].iloc[missing_garage_index] = df['YearBuilt'].iloc[missing_garage_index]
df['GarageFinish'].iloc[missing_garage_index] = df[df['GarageType'] == 'Detchd']['GarageFinish'].mode(dropna = False)[0]
df['GarageCars'].iloc[missing_garage_index[1]] = df[df['GarageType'] == 'Detchd']['GarageCars'].mode(dropna = False)[0]
df['GarageArea'].iloc[missing_garage_index[1]] = df[df['GarageType'] == 'Detchd']['GarageArea'].median()
df['GarageQual'].iloc[missing_garage_index] = df[df['GarageType'] == 'Detchd']['GarageQual'].mode(dropna = False)[0]
df['GarageCond'].iloc[missing_garage_index] = df[df['GarageType'] == 'Detchd']['GarageCond'].mode(dropna = False)[0]
df['GarageYrBlt'][df['GarageYrBlt'].isnull()] = df['YearBuilt'][df['GarageYrBlt'].isnull()]

# basement variables which are all NA means actual no basement
basementColumns = []
for columnName in df.columns:
    if columnName.lower().find('bsmt') != -1:
        basementColumns.append(columnName)
numBasementColumns = df[basementColumns].dtypes[df[basementColumns].dtypes != 'object'].index
cateBasementColumns = df[basementColumns].dtypes[df[basementColumns].dtypes == 'object'].index
noBasementIndex = df[(df['BsmtCond'].isnull() & df['BsmtQual'].isnull()
    & df['BsmtExposure'].isnull() & df['BsmtFinType1'].isnull() & df['BsmtFinType2'].isnull())].index
for index in noBasementIndex:
    for column in numBasementColumns:
        df.at[index, column] = 0
for index in noBasementIndex:
    for column in cateBasementColumns:
        df.at[index, column] = 'None'

# bsmtQual, replace na value with value of BsmtCond
df['BsmtQual'][df['BsmtQual'].isnull()] = df['BsmtCond'][df['BsmtQual'].isnull()]
# BsmtCond, replace na value with value of BsmtQual
df['BsmtCond'][df['BsmtCond'].isnull()] = df['BsmtQual'][df['BsmtCond'].isnull()]
# BsmtExposure, replace with mode
df['BsmtExposure'][df['BsmtExposure'].isnull()] = df['BsmtExposure'].mode()[0]
# BsmtFinType2, replace with FinType1
df['BsmtFinType2'][df['BsmtFinType2'].isnull()] = df['BsmtFinType2'].mode()[0]

# drop dominate level and combine too many levels:
dropDominate = ['PoolQC', 'MiscFeature', 'Alley', 'Utilities']

df = df.drop(dropDominate, axis = 1)

cols = ['Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType'] #replace categorical with none

for col in cols:
    df[col].fillna('None', inplace=True)

cols = ['LotFrontage'] #replace numerical with LotFrontage median in neighbor
for col in cols:
    df[col] = df.groupby('Neighborhood')[col].transform(lambda x: x.fillna(x.median()))
    
    
# index 2608 has MasVnrArea, but no MasVnrType, rest replace with mode/median
missing_MasVnrType = df[df['MasVnrArea'].notnull() & df['MasVnrType'].isnull()].index
df['MasVnrType'].iloc[missing_MasVnrType[0]] = 'BrkFace'
df['MasVnrType'][df['MasVnrType'].isnull()] = df['MasVnrType'].mode()[0]
df['MasVnrArea'][df['MasVnrArea'].isnull()] = df['MasVnrArea'].median()

# the rest missing categorical variable
cols = ['MSZoning', 'Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'] #replace categorical with the mode

for col in cols:
    df[col].fillna(df[col].mode()[0], inplace = True)


# # Normality
# May log transform independent variables later to make them normal

# In[ ]:


norm


# In[ ]:


plt.figure(figsize = [10, 7])
sns.distplot(dfTrainSalePrice, fit=norm, kde_kws={'label': 'Kernel Density Estimation'}, fit_kws={'label': 'Normal Distribution'});
plt.ylabel('pdf')
plt.title('Distribution of SalePrice(before log transformed)')
plt.legend()


# In[ ]:


res = stats.probplot(dfTrainSalePrice, plot=plt)


# In[ ]:


dfTrainSalePrice = np.log(dfTrainSalePrice)


# In[ ]:


plt.figure(figsize = [10, 7])
sns.distplot(dfTrainSalePrice, fit=norm, kde_kws={'label': 'Kernel Density Estimation'}, fit_kws={'label': 'Normal Distribution'});
plt.ylabel('pdf')
plt.title('Distribution of SalePrice(after log transformed)')
plt.legend()


# # Feature Engeneering

# ## remove or combine categorical variable with two many levels, use 10 as a bar

# In[ ]:


categorical_name = df.dtypes[df.dtypes == 'object'].index.tolist()
df_cate = df[categorical_name]
num_levels = 10
categoryOver10 = []
for column in df_cate.columns:
    if len(df_cate[column].value_counts()) > 10:
        categoryOver10.append(column)


# ### MSSsubClass

# In[ ]:


# minor_MSSubClass = df_cate['MSSubClass'].value_counts()[df_cate['MSSubClass'].value_counts() < 100].index.tolist()
# def combine_levels_MSSubClass(x):
#     if x in minor_MSSubClass:
#         return 'Others'
#     else:
#         return x
# df['MSSubClass'] = df['MSSubClass'].apply(combine_levels_MSSubClass)


# ### Neighborhood
# Doesn't want to combine the neighborhood, so first combine levels or remove variable, see the performance

# In[ ]:


# minor_Neighborhood = df_cate['Neighborhood'].value_counts()[df_cate['Neighborhood'].value_counts() < 100].index.tolist()
# def combine_levels_Neighborhood(x):
#     if x in minor_Neighborhood:
#         return 'Others'
#     else:
#         return x
# df['Neighborhood'] = df['Neighborhood'].apply(combine_levels_Neighborhood)


# ### Exterior1st

# In[ ]:


# minor_Exterior1st = df_cate['Exterior1st'].value_counts()[df_cate['Exterior1st'].value_counts() < 100].index.tolist()
# def combine_levels_Exterior1st(x):
#     if x in minor_Exterior1st:
#         return 'Others'
#     else:
#         return x
# df['Exterior1st'] = df['Exterior1st'].apply(combine_levels_Exterior1st)


# ### Exterior2nd

# In[ ]:


# minor_Exterior2nd = df_cate['Exterior2nd'].value_counts()[df_cate['Exterior2nd'].value_counts() < 100].index.tolist()
# def combine_levels_Exterior2nd(x):
#     if x in minor_Exterior2nd:
#         return 'Others'
#     else:
#         return x
# df['Exterior2nd'] = df['Exterior2nd'].apply(combine_levels_Exterior2nd)


# ### MoSold
# combine them based on season

# In[ ]:


# def combine_levels_MoSold(x):
#     if x in ['3', '4', '5']:
#         return 'Spring'
#     elif x in ['6', '7', '8']:
#         return 'Summer'
#     elif x in ['9', '10', '11']:
#         return 'Fall'
#     else:
#         return 'Winter'
# df['MoSold'] = df['MoSold'].apply(combine_levels_MoSold)


# ## remove variable with dominant level

# In[ ]:


percent_levels = 0.95
categoryOver95 = []
for column in df_cate.columns:
    if (df_cate[column].value_counts()/df_cate.shape[0]).sort_values(ascending = False)[0] > 0.95:
        categoryOver95.append(column)


# In[ ]:


percent_levels = 0.95
categoryOver95 = []
for column in df_cate.columns:
    if (df_cate[column].value_counts()/df_cate.shape[0]).sort_values(ascending = False)[0] > 0.95:
        categoryOver95.append(column)
df = df.drop(categoryOver95, axis = 1)


# ## Encode categorical variables
# ### based on the group mean of SalePrice

# In[ ]:


ordinal_cate = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish',
        'LotShape', 'PavedDrive', 'CentralAir', 'OverallCond', 
        'YrSold', 'MoSold']
categorical_name = ordinal_cate
df_train = df[:nTrain]
def encode(train, feature):
    df_train = train
    ordering = pd.concat([df_train[feature], dfTrainSalePrice], axis = 1)
    ordering = ordering.groupby(by = feature).mean().sort_values(by = 'SalePrice')
    ordering['SalePrice'] = range(1, len(ordering) + 1)
    ordering = ordering['SalePrice'].to_dict()
    for cate, order in ordering.items():
        df.loc[df[feature] == cate, feature + '_E'] = order
for feature in categorical_name:
    encode(train = df_train, feature = feature)
# df = df.drop(ordinal_cate, axis = 1)


# ## Create new variables

# In[ ]:


df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['YrBltAndRemod'] = df['YearBuilt']+df['YearRemodAdd']
df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                                 df['1stFlrSF'] + df['2ndFlrSF'])
df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                               df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                              df['EnclosedPorch'] + df['ScreenPorch'] +
                              df['WoodDeckSF'])


df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
# # Overall quality of the garage
# df["GarageGrade"] = df["GarageQual_E"] * df["GarageCond_E"]
# # Overall quality of the exterior
# df["ExterGrade"] = df["ExterQual_E"] * df["ExterCond_E"]
# # Overall kitchen score
# df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
# # Overall fireplace score
# df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]
# # Overall garage score
# df["GarageScore"] = df["GarageArea"] * df["GarageQual_E"]


# # Correlation between variables
# ## Find high correlated variable with target variable
# GarageCars and GarageArea has a high correlation, keep GarageCars.
# 
# TotalBsmtSF and 1stFlrSF have a high correlation, keep TotalBsmtSF.
# 
# GrLivArea and TotRmsAbvGrd have a high correlation, keep GrLivArea
# 
# GarageYrBlt and YearBuilt have a high correlation, keep YearBlt because it has a larger correlation with SalePrice

# In[ ]:


tmp_dfTrain = pd.concat([df[:nTrain], dfTrainSalePrice], axis = 1)
high_corr_index = tmp_dfTrain.corr().abs().nlargest(10, 'SalePrice').index
sns.heatmap(tmp_dfTrain.corr().loc[high_corr_index, high_corr_index], annot = True)


# In[ ]:


# create new features for top 10 variables
def add_poly(features):
    for i in range(1, len(features)):
        df[features[i] + '-2'] = df[features[i]]**2
        df[features[i] + '-3'] = df[features[i]]**3
        df[features[i] + '-sqrt'] = np.sqrt(df[features[i]])
add_poly(high_corr_index)


# ## Skewness for independent variables

# In[ ]:


from scipy.special import boxcox1p
lam = 0.15
skewed_feats = df.dtypes[df.dtypes != 'object'].index
skewed_feats = df[skewed_feats].apply(lambda x: x.skew(skipna=True)) #compute skewness
skewed_feats = skewed_feats[abs(skewed_feats) > 0.5]
skewed_feats = skewed_feats.index

df[skewed_feats] = boxcox1p(df[skewed_feats], lam)


# # Linear regression Model

# In[ ]:


from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


df = pd.get_dummies(df)
train = df[:nTrain]
test = df[nTrain:]


# ## Return cv_rmse

# In[ ]:


def cv_rmse(model = None):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X = train, y = dfTrainSalePrice)
    rmse_cv = np.sqrt(-cross_val_score(estimator=model, X = train, y = dfTrainSalePrice, scoring='neg_mean_squared_error', cv = kf))
    return rmse_cv


# ## Linear Regression

# In[ ]:


clf = LinearRegression()
rmse_cv = cv_rmse(model = clf)
print('Linear Regression root mean squared error: {}'.format(rmse_cv.mean()))


# ## Lasso

# In[ ]:


alphas = np.linspace(0.00001, 0.1, num = 100)
cv_lasso = [cv_rmse(model = Lasso(alpha = alpha)).mean() for alpha in alphas]
print("Alpha = {} of Lasso achieves RMSE = {}".format(alphas[np.argmin(cv_lasso)], cv_lasso[np.argmin(cv_lasso)]))


# ## Random Forest

# In[ ]:


regr = RandomForestRegressor()
rmse_cv = cv_rmse(model = regr)


# In[ ]:


rmse_cv.mean()


# ## predict test data

# In[ ]:


clf = make_pipeline(RobustScaler(), Lasso(alpha = 0.00102))
clf.fit(X = train, y = dfTrainSalePrice)
predSalePrice = pd.Series(np.exp(clf.predict(test)))
pred = pd.concat([dfTestId, predSalePrice], axis = 1)
pred.columns = ['Id', 'SalePrice']
pred.to_csv('submission.csv', index = False)

