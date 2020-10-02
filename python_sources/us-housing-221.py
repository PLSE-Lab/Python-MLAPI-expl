#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Storing the test and train csv file under a variable name

# In[ ]:


test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col='Id')
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col='Id')

print(test.shape)
print(train.shape)


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


import matplotlib

missing1 = train.isnull().sum()
missing1 = missing1[missing1>0]
missing1.sort_values()
missing1.plot.bar()

missing1


# In[ ]:


missing2 = test.isnull().sum()
missing2 = missing2[missing2>0]
missing2.sort_values()
missing2.plot.bar()

missing2


# In[ ]:


numerical_data = train.select_dtypes(exclude=['object']).drop(['SalePrice'],axis=1).copy()
print(numerical_data.columns)


# In[ ]:


categorical_data = train.select_dtypes(include = ['object']).copy()
print(categorical_data.columns)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# VISUALIZATION OF THE DATA THROUGH GRAPHS  

# DISTPLOT FUNCTION - A WAY TO APPLY KERNEL DENSITY ESTIMATION (ESTIMATION OF PROBABILITY DENSITY FUNCTION FOR OBVIOUSLY AN SINGLE VARIABLE)

# In[ ]:


fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(numerical_data.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})
    plt.xlabel(numerical_data.columns[i])
plt.tight_layout()
plt.show()


# BOX PLOT - JUST TO AGAIN HAVE A LOOK AT THE DISTRIBUTION OF DATA IN FORM OF 25,50,75 PERCENTILE DISTRIBUTIONS

# In[ ]:


fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=numerical_data.iloc[:,i])

plt.tight_layout()
plt.show()


# SCATTERPLOT TO ANALYSE RELATION BETWEEN EACH PARAMETER AND THE OUTPUT(SALEPRICE)

# In[ ]:


fig3 = plt.figure(figsize=(12,18))
for i in range(len(numerical_data.columns)):
    fig3.add_subplot(9, 4, i+1)
    sns.scatterplot(numerical_data.iloc[:, i],train['SalePrice'])
plt.tight_layout()
plt.show()


# DATA PREPROCESSING 
# 
# 

# OUTLIERS - Outlier is a rare chance of occurrence within a given data set

# 

# In[ ]:


sns.regplot(train['LotFrontage'],train['SalePrice'])


# In[ ]:





# 1. Similarly we may create regplot for all the variables

# WE NEED TO DEFINE CERTAIN VALUES ABOVE AND BELOW WHICH WE MAY NOT WORK FOR SPECIFIED DATA, SO AS IN ORDER TO REMOVE THE ERROR OF OUTLIERS 

# In[ ]:


train = train.drop(train[train['LotFrontage']>200].index)
train = train.drop(train[train['LotArea']>100000].index)
train = train.drop(train[train['MasVnrArea']>1200].index)
train = train.drop(train[train['BsmtFinSF1']>4000].index)
train = train.drop(train[train['TotalBsmtSF']>4000].index)
train = train.drop(train[train['1stFlrSF']>4000].index)
train = train.drop(train[train['EnclosedPorch']>500].index)
train = train.drop(train[train['MiscVal']>5000].index)
train = train.drop(train[train['BsmtFinSF1']>4000].index)
train = train.drop(train[train['WoodDeckSF']>800].index)
train = train.drop(train[train['BsmtFinSF1']>4000].index)
train = train.drop(train[(train['LowQualFinSF']>600) & (train['SalePrice']>400000)].index)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# CORELATION CHECK - We will be looking forward to see if two of the input features are highly related that is can act as two dependent variables in LAL , so we apply an correlation check , using the heat map.

# In[ ]:


numerical_corelation = train.select_dtypes(exclude='object').corr()
plt.figure(figsize=(20,20))
plt.title("High Corelation")
sns.heatmap(numerical_corelation>0.8, annot=True, square=True)


# FROM THE TWO FACTORS SHOWN TO BE CORRELATED REMOVE ANY ONE OF THE FACTORS

# In[ ]:


train.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)


# Now look for data which is useless (column which may belong to categorial_type such as we may in future come to a conclusion that 'Street' and 'Utilities' do not make any difference to our data )

# In[ ]:


train=train.drop(columns=['Street','Utilities']) 
test=test.drop(columns=['Street','Utilities'])


# Which column has the most of the values empty

# In[ ]:


train.isnull().mean().sort_values(ascending=False).head(5)


# From the above result we conclude that the given three top features have more than 90% of the given as not defined , so we decide on dropping the three values and without PoolQC , PoolArea makes less of sense , thus making the drop column to four.

# In[ ]:


train.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'], axis=1, inplace=True)
test.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'], axis=1, inplace=True)


# FILLING THE DATA WHICH IS LEFT NOT FILLED

# In[ ]:


#look at the percentage of each data missing 
null = pd.DataFrame(data={'Train Null Percentage': train.isnull().sum()[train.isnull().sum() > 0], 
'Test Null Percentage': test.isnull().sum()[test.isnull().sum() > 0]})
null = (null/len(train)) * 100

null.index.name='Feature'
null


# In[ ]:


home_num_features = train.select_dtypes(exclude='object').isnull().mean()
test_num_features = test.select_dtypes(exclude='object').isnull().mean()

num_null_features = pd.DataFrame(data={'Missing Num Train Percentage: ': home_num_features[home_num_features>0]*100, 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]*100})
num_null_features.index.name = 'Numerical Features'
num_null_features


# In[ ]:


for df in [train, test]:
    for col in ('GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotalBsmtSF',
                'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal',
                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea'):
                    df[col] = df[col].fillna(0)


# In[ ]:


_=sns.regplot(train['LotFrontage'],train['SalePrice'])


# In[ ]:


home_num_features = train.select_dtypes(exclude='object').isnull().mean()
test_num_features = test.select_dtypes(exclude='object').isnull().mean()

num_null_features = pd.DataFrame(data={'Missing Num Home Percentage: ': home_num_features[home_num_features>0]*100, 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]*100})
num_null_features.index.name = 'Numerical Features'
num_null_features


# *WE DECIDE TO WORK ON LOT FRONTAGE FEATURE LATER AS IT IS AN IMPORTANT FEATURE **

# Categorial data  ******

# In[ ]:


cat_col = train.select_dtypes(include='object').columns
print(cat_col)


# In[ ]:


home_cat_features = train.select_dtypes(include='object').isnull().mean()
test_cat_features = test.select_dtypes(include='object').isnull().mean()

cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0]*100, 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]*100})
cat_null_features.index.name = 'Categorical Features'
cat_null_features


# In[ ]:


cat_col = train.select_dtypes(include='object').columns

columns = (len(cat_col)/4)+1

fg, ax = plt.subplots(figsize=(20, 30))

for i, col in enumerate(cat_col):
    fg.add_subplot(columns, 4, i+1)
    sns.countplot(train[col])
    plt.xlabel(col)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# * HERE WE BEGAN WITH SOME INCREASE AS IN THE CASE TO MODIFY THE DATA GIVEN AS NAN
# 

# In[ ]:


for df in [train, test]:
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence'):
        df[col] = df[col].fillna('None')


# In[ ]:


for df in [train, test]:
    for col in ('LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'RoofStyle',
                  'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'ExterQual', 'ExterCond',
                  'Foundation', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition'):
        df[col] = df[col].fillna(df[col].mode()[0])                            #returns mode of each column if no value is passed ,else if axis = 1 ,then we may return mode of row instead.


# * WE LEFT HERE THE PARAMETER OF MSZONING

# In[ ]:


home_cat_features = train.select_dtypes(include='object').isnull().mean()
test_cat_features = test.select_dtypes(include='object').isnull().mean()

cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0]*100, 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]*100})
cat_null_features.index.name = 'Categorical Features'
cat_null_features


# *  **AS WE KNOW THAT LOTFRONTAGE WAS AN IPORTANT PARAMETER AND WE NEEDED TO FILL THE MISSING VALUES IN A DELICATE MANNER**

# FILLING VALUES IN LotFrontage

# In[ ]:


sns.regplot(train['LotFrontage'],train['SalePrice'])


# In[ ]:


train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))


# 

# In[ ]:


train.corr()['LotFrontage'].sort_values(ascending=False)


# In[ ]:


train.corr()['SalePrice'].sort_values(ascending=False)


# * WE HAVE HANDLED ALL THE COLUMNS IN THE REQUIRED WAY IN THE TRAIN SET
# 
# * WE NOW JUST NEED TO HANDLE THE PARAMETER MSZONING THAT TOO ONLY IN THE TEST SET AS ALL ITS VALUE IN THE TRAIN SET ARE PREDEFINED.
# 

# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# FILLING MISSING VALUES IN MSZoning 
# 

# In[ ]:


train['MSSubClass'] = train['MSSubClass'].apply(str)
train['MSSubClass']


# In[ ]:


train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)

train['MoSold'] = train['MoSold'].apply(str)
test['MoSold'] = test['MoSold'].apply(str)

train['YrSold'] = train['MoSold'].apply(str)
test['YrSold'] = test['MoSold'].apply(str)


# We changed the type of MSSubClass to string so we can impute the median based on MSSubClass in MSZoning in the next step. We also changed MoSold and YrSold because they should be strings not integers

# In[ ]:


train['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test['MSZoning'] = test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# * SEEING THE OUTPUT OF THE ABOVE DATA SET WE CONCLUDE NOW ALL THE PARAMETERS ARE HAVING FILLED DATA IN BOTH TEST AND TRAIN SET

# In[ ]:


train['TotalSF']=train['TotalBsmtSF']  + train['2ndFlrSF']
test['TotalSF']=test['TotalBsmtSF']  + test['2ndFlrSF']


# In[ ]:


train['TotalBath']= train['BsmtFullBath'] + train['FullBath'] + (0.5*train['BsmtHalfBath']) + (0.5*train['HalfBath'])
test['TotalBath']=test['BsmtFullBath'] + test['FullBath'] + 0.5*test['BsmtHalfBath'] + 0.5*test['HalfBath']


# In[ ]:


train['YrBltAndRemod']=train['YearBuilt']+(train['YearRemodAdd']/2)
test['YrBltAndRemod']=test['YearBuilt']+(test['YearRemodAdd']/2)


# In[ ]:


train['Porch_SF'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])
test['Porch_SF'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])


# In[ ]:


train['Has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasBsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasFirePlace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train['Has2ndFlr']=train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasBsmt']=train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['Has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasFirePlace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFlr']=test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt']=test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


print(type(train['LotArea'][10]))

train['LotArea'] = train['LotArea'].astype(np.int64)
test['LotArea'] = test['LotArea'].astype(np.int64)
train['MasVnrArea'] = train['MasVnrArea'].astype(np.int64)
test['MasVnrArea'] = test['MasVnrArea'].astype(np.int64)


# In[ ]:



print ("Skew of SalePrice:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='yellow')
plt.show()


# In[ ]:


print ("Skew of Log-Transformed SalePrice:", np.log1p(train.SalePrice).skew())
plt.hist(np.log1p(train.SalePrice), color='green')
plt.show()


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn import metrics 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from mlxtend.regressor import StackingCVRegressor


# In[ ]:


X = train.drop(['SalePrice'], axis=1)
y = np.log1p(train['SalePrice'])


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2)


# In[ ]:


categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() <= 30 and
                    X[cname].dtype == "object"] 
                


numerical_cols = [cname for cname in X.columns if
                 X[cname].dtype in ['int64','float64']]


my_cols = numerical_cols + categorical_cols

X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()
X_test = test[my_cols].copy()


# In[ ]:


num_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='constant'))
    ])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),       
        ('cat',cat_transformer,categorical_cols),
        ])


# **Don't get confused , I have used the same 3 models twice but with different hyperparamets (as I do not know tuning hyperparameters for such regressor)**

# In[ ]:


def inv_y(transformed_y):
    return np.exp(transformed_y)

n_folds = 10

# XGBoost
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

      
# Lasso   
model = LassoCV(max_iter=1e6)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
  
      
      
# GradientBoosting   
model = GradientBoostingRegressor()
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))


# In[ ]:


def inv_y(transformed_y):
    return np.exp(transformed_y)

n_folds = 10

# XGBoost
model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

      
# Lasso   
model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
  
      
      
# GradientBoosting   
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))


# In[ ]:


n_folds = 10

model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])


scores = cross_val_score(clf, X, y, scoring='neg_mean_squared_error', 
                         cv=n_folds)
gbr_mae_scores = -scores

print('RMSE: ' + str(gbr_mae_scores.mean()))
print('Error std deviation: ' +str(gbr_mae_scores.std()))


# 

# In[ ]:


model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': inv_y(final_predictions)})

output.to_csv('submission.csv', index=False)


# 

# In[ ]:





# In[ ]:




