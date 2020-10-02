#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement: House price prediction on a dataset using Sacked Regression and model ensemble

# In[ ]:


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#All imports
import os
import pandas as pd
import numpy as np
from datetime import datetime
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn import metrics

import statsmodels.api as sm


# In[ ]:


#list all the filenames
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Approach:
# **=> Data Reading and Understanding<br>
# => Data Cleaning (Data Visualization, Null value treatment, Outlier treatment)<br>
# => Data Preparation (Converting categorical variables to numeric variables, Scaling)<br>
# => Model Building and Evaluation<br>
# => Predictions on test set**

# ## Step 1 - Data Reading and Understanding

# In[ ]:


#Read the train data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()


# In[ ]:


#Read the test data
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_df.head()


# In[ ]:


#Store test Ids in a seperate column to be used for submission
train_Ids = train_df.pop('Id')
test_Ids = test_df.pop('Id')

test_df['SalePrice'] = np.NaN


# **Concatenate train and test dataset for data cleaning and preparation**

# In[ ]:


housing_all_df = pd.concat([train_df, test_df], axis = 0, ignore_index=True)

print ("Train shape: ", train_df.shape)
print ("Test shape: ", test_df.shape)
print ("Combined shape: ", housing_all_df.shape)


# ## Step 2 - Data Visualization and Cleaning

# ### Data Visualization

# In[ ]:


#Distribution of SalePrice column
plt.figure(figsize=(10,4))
sns.distplot(housing_all_df[~housing_all_df['SalePrice'].isnull()]['SalePrice'])
plt.show()


# **Target column is certainly right skewed and will require transformation to fit linear regression**

# In[ ]:


#Visualization of Overall quality and SalePrice
plt.figure(figsize=(6,4))
sns.boxplot(x='OverallQual', y ='SalePrice', data=housing_all_df)
plt.show()


# **SalePrice shows a direct relation with Overall quality of the house. Higher is the overall quality, appreciation of house prices can be seen**

# In[ ]:


#Visualization of SalePrice and AboveGradeLivingArea
sns.scatterplot(x='GrLivArea', y= 'SalePrice', data = housing_all_df)
plt.show()


# In[ ]:


sns.scatterplot(x='GrLivArea', y= 'SalePrice', hue='OverallQual', data = housing_all_df)
plt.show()


# **General trend seems to be - higher the GrLivArea, higher is the SalePrice with a couple of outliers. These outliers have higher GrLivArea but SalePrice is low even though these houses have higher OverallQuality**

# ### Data Cleaning

# In[ ]:


#As feature 'Utilities' has no variance,dropping column: Utilities
housing_all_df.drop(columns=['Utilities'], inplace = True)


#  ### Null value Imputation

# In[ ]:


#checking the null count for each column
null_counts = housing_all_df.isnull().sum()
null_counts[null_counts.values > 0].index


# In[ ]:


#Impute MSZoning with mode values
print("Null count: ", housing_all_df['MSZoning'].isnull().sum())
print (housing_all_df['MSZoning'].value_counts())

housing_all_df['MSZoning'].fillna(housing_all_df['MSZoning'].mode().values[0], inplace = True)


# In[ ]:


#Impute PoolQC
print ("Null count: ", housing_all_df['PoolQC'].isnull().sum())
housing_all_df['PoolQC'].value_counts()


# In[ ]:


#Check if records have a valid PoolArea and still PoolQC is missing
housing_all_df[(housing_all_df['PoolQC'].isnull()) & (housing_all_df['PoolArea'] > 0)][['PoolArea','PoolQC','OverallQual']]


# **Houses have a pool but their PoolQC is missing. This is certainly a miss and missing values should be imputed.**

# In[ ]:


#Check relation between PoolArea and PoolQC
sns.boxplot(x='PoolQC', y='PoolArea', data=housing_all_df)
plt.show()


# **As PoolArea and PoolQC does not show a direct relation, so: <br>- Houses with PoolArea but missing PoolQC will be subtituted with OverallQuality rating of the house<br>- Rest of the houses with PoolQC mising are the ones with PoolArea as 0, so substituting 'None', indicating no Pool in the house**

# In[ ]:


housing_all_df['PoolQC'].fillna(value='None', inplace=True)
housing_all_df['PoolQC'] = housing_all_df['PoolQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

housing_all_df[housing_all_df.index == 2420]['PoolQC'] = 4
housing_all_df[housing_all_df.index == 2503]['PoolQC'] = 6
housing_all_df[housing_all_df.index == 2599]['PoolQC'] = 3


# In[ ]:


#Check impact of MiscFeature and MiscVal on SalePrice column
sns.scatterplot(x='MiscVal',y='SalePrice',hue='MiscFeature',data=housing_all_df)
plt.show()


# In[ ]:


#As MiscFeature and MiscVal does not show any impact on SalePrice column, so dropping these features
housing_all_df.drop(columns=['MiscFeature','MiscVal'], inplace=True)


# In[ ]:


#Impute Alley with mode value
print ("Null count: ", housing_all_df['Alley'].isnull().sum())
print (housing_all_df['Alley'].value_counts())

housing_all_df['Alley'].fillna(value='None', inplace = True)


# In[ ]:


#Impute Fence with mode value
print ("Null count: ", housing_all_df['Fence'].isnull().sum())
print (housing_all_df['Fence'].value_counts())

housing_all_df['Fence'].fillna(value='None', inplace=True)


# In[ ]:


#Impute FireplaceQu
housing_all_df['FireplaceQu'].fillna(value='None', inplace=True)


# In[ ]:


#Impute LotFrontage
grouped_by_neighborhood = housing_all_df.groupby('Neighborhood')
lot_frontage_median = grouped_by_neighborhood['LotFrontage'].median()
lot_frontage_substitute = housing_all_df['Neighborhood'].map(lot_frontage_median.to_dict())
housing_all_df['LotFrontage'] = lot_frontage_substitute.where(pd.isnull(housing_all_df['LotFrontage']), housing_all_df['LotFrontage'])


# **Impute GarageVariables<br>-Houses where GarageArea is 0 indicates NoGarage, so substituting 'No Garage' and 'None' for GarageQuality and GarageCondition**

# In[ ]:


housing_all_df['GarageYrBlt'] = housing_all_df['YearBuilt'].where(pd.isnull(housing_all_df['GarageYrBlt']), housing_all_df['GarageYrBlt'])
housing_all_df.loc[housing_all_df['GarageType'].isnull() & (housing_all_df['GarageArea'] == 0), 'GarageType'] = 'No Garage'
housing_all_df.loc[housing_all_df['GarageFinish'].isnull() & (housing_all_df['GarageArea'] == 0), 'GarageFinish'] = 'No Garage'
housing_all_df.loc[housing_all_df['GarageQual'].isnull() & (housing_all_df['GarageArea'] == 0), 'GarageQual'] = 'None'
housing_all_df.loc[housing_all_df['GarageCond'].isnull() & (housing_all_df['GarageArea'] == 0), 'GarageCond'] = 'None'


# In[ ]:


#Check if any house with GarageQual and GarageCond missing
housing_all_df[(housing_all_df['GarageQual'].isnull()) | (housing_all_df['GarageCond'].isnull())][['GarageCars','GarageArea',
                                'GarageType','GarageFinish','GarageQual','GarageCond']]


# In[ ]:


#House with index 2576 indicates it does not have Garage and GarageType is Detchd because of erroneous data, 
#so fixing values to indicate there is no garage
housing_all_df.loc[housing_all_df.index==2576, 'GarageCars']= 0
housing_all_df.loc[housing_all_df.index==2576, 'GarageArea']= 0
housing_all_df.loc[housing_all_df.index==2576, 'GarageType']= 'No Garage'
housing_all_df.loc[housing_all_df.index==2576, 'GarageFinish']= 'No Garage'
housing_all_df.loc[housing_all_df.index==2576, 'GarageQual']= 'None'
housing_all_df.loc[housing_all_df.index==2576, 'GarageCond']= 'None'


# In[ ]:


#For rest of the cases, substituting mode values
housing_all_df.loc[housing_all_df['GarageFinish'].isnull(),'GarageFinish'] = housing_all_df['GarageFinish'].mode().values[0]
housing_all_df.loc[housing_all_df['GarageCond'].isnull(),'GarageCond'] = housing_all_df['GarageCond'].mode().values[0]
housing_all_df.loc[housing_all_df['GarageQual'].isnull(),'GarageQual'] = housing_all_df['GarageQual'].mode().values[0]


# In[ ]:


#Impute Exterior1st
print("Null count: ", housing_all_df['Exterior1st'].isnull().sum())
print (housing_all_df['Exterior1st'].value_counts())

housing_all_df['Exterior1st'].fillna(value=housing_all_df['Exterior1st'].mode().values[0], inplace=True)


# In[ ]:


#Impute Exterior2nd
print("Null count: ",housing_all_df['Exterior2nd'].isnull().sum())
print (housing_all_df['Exterior2nd'].value_counts())

housing_all_df['Exterior2nd'].fillna(housing_all_df['Exterior2nd'].mode().values[0], inplace = True)


# **Impute MasVnrType and MasVnrArea: <br><br>- Houses where MasVnrArea is 0, substituting MasVnrType as None. <br>- Houses with valid MasVnrArea, substituting them with mode values.<br>- Houses with missing MasVnrArea, substituting them with 0 value**

# In[ ]:


print ("Null count: ", housing_all_df['MasVnrType'].isnull().sum())
print (housing_all_df['MasVnrType'].value_counts())

housing_all_df.loc[housing_all_df['MasVnrType'].isnull() & housing_all_df['MasVnrArea'].isnull(), 'MasVnrType'] = 'None'
housing_all_df.loc[housing_all_df['MasVnrType'].isnull() & housing_all_df['MasVnrArea'], 'MasVnrType'] = housing_all_df['MasVnrType'].mode().values[0]

housing_all_df['MasVnrArea'].fillna(value=0, inplace=True)


# **Impute Basement variables:<br><br>- Houses with missing TotalBsmtSF or TotalBsmtSF as 0, substituting Basement related variables with None indicating there is no basement. <br> - For rest of the houses with Basement, substituting basement related variables with mode values**

# In[ ]:


housing_all_df.loc[housing_all_df['TotalBsmtSF'].isnull(),'BsmtFinSF1'] = 0
housing_all_df.loc[housing_all_df['TotalBsmtSF'].isnull(),'BsmtFinSF2'] =0
housing_all_df.loc[housing_all_df['TotalBsmtSF'].isnull(),'BsmtUnfSF'] =0
housing_all_df.loc[housing_all_df['TotalBsmtSF'].isnull(),'BsmtFullBath'] =0
housing_all_df.loc[housing_all_df['TotalBsmtSF'].isnull(),'BsmtHalfBath'] =0
housing_all_df.loc[housing_all_df['TotalBsmtSF'].isnull(),'TotalBsmtSF'] =0


# In[ ]:


housing_all_df.loc[(housing_all_df['TotalBsmtSF'] == 0) & (housing_all_df['BsmtQual'].isnull()), 'BsmtQual'] = 'None'
housing_all_df.loc[(housing_all_df['TotalBsmtSF'] == 0) & (housing_all_df['BsmtCond'].isnull()), 'BsmtCond'] = 'None'
housing_all_df.loc[(housing_all_df['TotalBsmtSF'] == 0) & (housing_all_df['BsmtExposure'].isnull()), 'BsmtExposure'] = 'None'
housing_all_df.loc[(housing_all_df['TotalBsmtSF'] == 0) & (housing_all_df['BsmtFinType1'].isnull()), 'BsmtFinType1'] = 'None'
housing_all_df.loc[(housing_all_df['TotalBsmtSF'] == 0) & (housing_all_df['BsmtFinType2'].isnull()), 'BsmtFinType2'] = 'None'


# In[ ]:


housing_all_df.loc[housing_all_df['BsmtQual'].isnull(), 'BsmtQual'] = housing_all_df['BsmtQual'].mode().values[0]
housing_all_df.loc[housing_all_df['BsmtCond'].isnull(), 'BsmtCond'] = housing_all_df['BsmtCond'].mode().values[0]
housing_all_df.loc[housing_all_df['BsmtExposure'].isnull(), 'BsmtExposure'] = housing_all_df['BsmtExposure'].mode().values[0]
housing_all_df.loc[housing_all_df['BsmtFinType2'].isnull(), 'BsmtFinType2'] = housing_all_df['BsmtFinType2'].mode().values[0]
housing_all_df.loc[housing_all_df['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0
housing_all_df.loc[housing_all_df['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0


# In[ ]:


#Impute Electrical
print("Null counts: " , housing_all_df['Electrical'].isnull().sum())
print (housing_all_df['Electrical'].value_counts())

housing_all_df['Electrical'].fillna(housing_all_df['Electrical'].mode().values[0], inplace=True)


# In[ ]:


#Impute kitchenQual
print("Null count: " , housing_all_df['KitchenQual'].isnull().sum())
print (housing_all_df['KitchenQual'].value_counts())

housing_all_df['KitchenQual'].fillna(housing_all_df['KitchenQual'].mode().values[0], inplace=True)


# In[ ]:


#Impute Functional
print("Null count: ", housing_all_df['Functional'].isnull().sum())
print (housing_all_df['Functional'].value_counts())

housing_all_df['Functional'].fillna(housing_all_df['Functional'].mode().values[0], inplace = True)


# In[ ]:


#Impute SaleType
print("Null count: ", housing_all_df['SaleType'].isnull().sum())
print (housing_all_df['SaleType'].value_counts())

housing_all_df['SaleType'].fillna(housing_all_df['SaleType'].mode().values[0], inplace=True)


# ### Feature Engineering

# In[ ]:


#Compute TotalBathroom from all types of bathrooms
housing_all_df['TotalBath'] = housing_all_df['FullBath'] + housing_all_df['HalfBath'] * 0.5 + housing_all_df['BsmtFullBath'] + housing_all_df['BsmtHalfBath']*0.5

sns.boxplot(x='TotalBath', y ='SalePrice', data = housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()


# **Higher the number of TotalBath in the house, better is the price appreciation**

# **Compute:<br><br> IsRemod - Indicating if house is renovated.<br>Age - Indicating how many years house is with an owner after renovation/built.<br> IsNew - Indicating house is not sold yet after construction or if its a new property**

# In[ ]:


housing_all_df['IsRemod'] = np.where(housing_all_df['YearBuilt'] == housing_all_df['YearRemodAdd'], 0,1)
housing_all_df['Age'] = housing_all_df['YrSold'] - housing_all_df['YearRemodAdd']
                                                                  
sns.scatterplot(x='Age', y='SalePrice', data = housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()
                                                                  
sns.boxplot(x='IsRemod', y='SalePrice', data = housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()
                                                                  
housing_all_df['IsNew'] = np.where(housing_all_df['YearBuilt'] == housing_all_df['YrSold'], 0, 1)
                                                                  
sns.boxplot(x='IsNew', y='SalePrice', data = housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()                                                             


# In[ ]:


#Relation of GrLivArea with SalePrice
sns.scatterplot(x='GrLivArea', y='SalePrice', data=housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()

housing_all_df[['GrLivArea','SalePrice']].corr()


# In[ ]:


#Relation of TotalBasementArea with SalePrice
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()

housing_all_df[['SalePrice','TotalBsmtSF']].corr()


# **Compute TotalArea with above ground living area and basement area and correlation of SalePrice with TotalArea**

# In[ ]:


housing_all_df['TotalArea'] = housing_all_df['GrLivArea'] + housing_all_df['TotalBsmtSF']

sns.scatterplot(x='TotalArea', y='SalePrice', data=housing_all_df)
plt.show()

housing_all_df[['TotalArea','SalePrice']].corr()


# In[ ]:


#Compute TotalPorchArea by combining all the PorchArea and their relation with SalePrice
housing_all_df['TotalPorchArea'] = housing_all_df['OpenPorchSF'] + housing_all_df['EnclosedPorch'] + housing_all_df['3SsnPorch'] + housing_all_df['ScreenPorch']

sns.scatterplot(x='TotalPorchArea', y='SalePrice',data=housing_all_df[~housing_all_df['SalePrice'].isnull()])
plt.show()

housing_all_df[['TotalPorchArea', 'SalePrice']].corr()


# **Check correlations amongst numerical columns and drop the ones with high correlation**

# In[ ]:


numerical_cols = housing_all_df.select_dtypes(include='number').columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['MSSubClass', 'OverallQual','OverallCond','PoolQC','MoSold']]
housing_corr = housing_all_df[numerical_cols].corr()


# In[ ]:


high_corr_matrix = housing_corr[abs(housing_corr.loc[:,:]) >= 0.5]
high_corr_matrix.to_csv('./correlations.csv')


# **YearBuilt and GarageYrBlt has correlation of 0.85<br>Age and YearRemodAdd has a correlation has -0.99<br>1stFlrSF and TotalBsmtSF has a correlation of 0.80<br>TotalArea and TotalBsmtSF has a correlation of 0.82<br>TotalRoomsAbvGrd and GrLivArea has a correlation of 0.80<br>TotalArea and GrLivArea has a correlation of 0.87<br>GarageCars and GarageArea has a correlation of 0.88**

# In[ ]:


cols_to_drop = ['GarageYrBlt','YearRemodAdd','1stFlrSF','TotalBsmtSF','TotRmsAbvGrd','GrLivArea','GarageArea']
housing_all_df.drop(columns=cols_to_drop, inplace=True)


# ### Outlier treatment

# In[ ]:


#Houses with higher GrLivArea, lower SalePrice but higher Overall house quality
train_df[train_df['OverallQual'] == 10][['GrLivArea', 'SalePrice']].sort_values('GrLivArea', ascending=False)


# In[ ]:


#dropping the houses with index 1298 and 523
housing_all_df.drop(index=[523,1298], inplace=True)


# ### Skewness treatment of numerical columns

# In[ ]:


numerical_cols = housing_all_df.select_dtypes(include='number').columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['MSSubClass','OverallQual','OverallCond','PoolQC','MoSold','SalePrice','IsRemod','IsNew','YearBuilt','YrSold']]

#treat skewness of numerical columns
for col in numerical_cols:
    skew_val = scipy.stats.skew(housing_all_df[col])
    if abs(skew_val) > 0.8:
        housing_all_df[col] = np.log(housing_all_df[col] + 1)    


# ## Step 3 - Data preparation

# In[ ]:


#For features with categories having implicit ordering, assigning them incremental numeric values
for col in ['FireplaceQu', 'ExterQual', 'ExterCond', 'BsmtQual','BsmtCond','KitchenQual','GarageQual','GarageCond','HeatingQC']:
    housing_all_df[col] = housing_all_df[col].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
    
housing_all_df['BsmtExposure'] = housing_all_df['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})

for col in ['BsmtFinType1','BsmtFinType2']:
    housing_all_df[col] = housing_all_df[col].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
    
housing_all_df['GarageFinish'] = housing_all_df['GarageFinish'].map({'No Garage':0,'Unf':1,'RFn':2,'Fin':3})


# In[ ]:


#Check if MasVnrType has any ranking by considering its impact on SalePrice
sns.boxplot(x='MasVnrType', y = 'SalePrice', data=housing_all_df)
plt.show()

housing_all_df['MasVnrType'] = housing_all_df['MasVnrType'].map({'None':0, 'BrkCmn':0, 'BrkFace':1, 'Stone':2}) 


# In[ ]:


#Check if feature 'Functional' has any ranking by considering its impact on SalePrice
sns.boxplot(x='Functional', y = 'SalePrice', data=housing_all_df)
plt.show()

housing_all_df['Functional'] = housing_all_df['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,
                                                                 'Min1':6,'Typ':7}) 


# **Combining neighborhood categories**

# In[ ]:


#Determine the neighborhood categories that can be combined by checking their average and median sale prices
groupd_by_neighborhood = housing_all_df[~housing_all_df['SalePrice'].isnull()].groupby('Neighborhood')
neighborhood_median_prices = groupd_by_neighborhood['SalePrice'].median()
neighborhood_median_prices = neighborhood_median_prices.sort_values()

neighborhood_mean_prices = groupd_by_neighborhood['SalePrice'].mean()
neighborhood_mean_prices = neighborhood_mean_prices.sort_values()


# In[ ]:


#plot of neighboorhood median prices
plt.figure(figsize=(12,6))
sns.barplot(x=neighborhood_median_prices.index, y = neighborhood_median_prices.values)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#plot of neighborhood average prices
plt.figure(figsize=(12,6))
sns.barplot(x=neighborhood_mean_prices.index, y = neighborhood_mean_prices.values)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Combining the categories and assigning them incremental values to indicate ordering
housing_all_df.loc[~housing_all_df['Neighborhood'].isin(['StoneBr','NridgHt','NoRidge','MeadowV','IDOTRR','BrDale']), 'Neighborhood'] = 1
housing_all_df.loc[housing_all_df['Neighborhood'].isin(['StoneBr','NridgHt','NoRidge']), 'Neighborhood'] = 2
housing_all_df.loc[housing_all_df['Neighborhood'].isin(['MeadowV','IDOTRR','BrDale']), 'Neighborhood'] = 0


# In[ ]:


housing_all_df['Neighborhood'] = housing_all_df['Neighborhood'].astype(int)


# **Convert nominal categories into dummies using one-hot encoding**

# In[ ]:


all_dummies = []
cat_cols = housing_all_df.select_dtypes(include='object').columns.tolist()
cat_cols.extend(['MSSubClass', 'MoSold','YrSold'])
for col in cat_cols:
    dummies = pd.get_dummies(housing_all_df[col], prefix=col, prefix_sep='_', drop_first=True)
    for dum_col in dummies.columns.tolist():
        if dummies[dum_col].sum() < 10:
            print ("Dummy: {0} has less than 10 records".format(dum_col))
            dummies.drop(columns=[dum_col], inplace =True)
    all_dummies.extend(dummies)
    housing_all_df.drop(columns=col, inplace=True)
    housing_all_df = pd.concat([housing_all_df, dummies], axis=1)


# **Split data back to train and test for modelling**

# In[ ]:


housing_train_df = housing_all_df[~housing_all_df['SalePrice'].isnull()]
housing_test_df = housing_all_df[housing_all_df['SalePrice'].isnull()]

print ("Train data: ", housing_train_df.shape)
print ("Test data: ", housing_test_df.shape)


# **Fix SalePrice distribution - Perform log transformation to treat right skewed feature**

# In[ ]:


housing_train_df['SalePrice_log'] = np.log(housing_train_df['SalePrice'])

sns.distplot(housing_train_df['SalePrice_log'])
plt.show()


# ## Step 4 - Model building

# In[ ]:


X = housing_train_df.drop(columns=['SalePrice', 'SalePrice_log'])
y = housing_train_df['SalePrice_log']

#split the training data into train and test set. This test set will be used as validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


#A custom function to be used for scoring
def root_mean_squared_error(actual, predicted):
    return np.sqrt(metrics.mean_squared_error(actual,predicted))

custom_scorer = metrics.make_scorer(root_mean_squared_error, greater_is_better = False)


# In[ ]:


#Custom function to perform GridSearchCV with 5 folds and rmse scoring mechanism.
def grid_search_cv(params, model, X_train, y_train, X_test, y_test):
    folds = 5
    
    #perform GridSearchCV
    model_cv = GridSearchCV(estimator = model,
                        param_grid = params,
                        scoring = custom_scorer,                        
                        cv = folds,
                        verbose = True,
                        return_train_score = True,
                        n_jobs=-1)

    model_cv.fit(X_train, y_train)    
    print ("Best score: ", model_cv.best_score_)
    print ("Best params: ", model_cv.best_params_)
    y_test_pred = model_cv.best_estimator_.predict(X_test)
    print ("Test score: ", root_mean_squared_error(y_test, y_test_pred))


# ### Model 1 - Lasso Regression

# In[ ]:


#regularization coefficient alpha to tune
params = {'model__alpha' : [0.0001, 0.001,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 1, 2, 
                    5, 10, 50]}
    
#model to be used
lasso = Pipeline([('scaler',RobustScaler()), ('model',Lasso(random_state = 100, normalize=True))])

#Invoke GridSearchCV
grid_search_cv(params, lasso, X_train, y_train, X_test, y_test)


# **Making a final lasso model with tuned parameters**

# In[ ]:


lasso_final = Pipeline([('scaler', RobustScaler()), ('model', Lasso(random_state=100, alpha=0.0001, normalize=True))])
lasso_final.fit(X, y)

y_pred = lasso_final.predict(X)
print ("Lasso rmse: ", root_mean_squared_error(y, y_pred))


# ### Model 2 - Ridge Regression

# In[ ]:


#regularization coefficient alpha to tune
params = {'model__alpha' : [0.0001, 0.001,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 1, 2, 
                    5, 10, 50]}
    
#model to be used
ridge = Pipeline([('scaler',RobustScaler()), ('model',Ridge(random_state = 100, normalize=True))])

#Invoke GridSearchCV
grid_search_cv(params, ridge, X_train, y_train, X_test, y_test)


# **Making a final ridge final model with tuned parameters**

# In[ ]:


ridge_final = Pipeline([('scaler',RobustScaler()), ('model',Ridge(random_state=100, alpha =0.04, normalize=True))])
ridge_final.fit(X, y)

y_pred = ridge_final.predict(X)
print ("Ridge rmse: ", root_mean_squared_error(y, y_pred))


# ### Model 3 - XGBoostRegressor

# **Fix learning rate to 0.1 and other tree parameters to default values to determine n_estimators**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1,max_depth=5, min_child_weight=1,subsample=0.8,
                  colsample_by_tree=0.8)
hyper_params = {'n_estimators':[50,100,200]}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1,max_depth=5, min_child_weight=1,subsample=0.8,
                  colsample_by_tree=0.8)
hyper_params = {'n_estimators':[250,300,350]}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# **Keeping learning rate to 0.1 and n_estimators to 200, tune tree parameters**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1,n_estimators=200, subsample=0.8,
                  colsample_by_tree=0.8)
hyper_params = {'max_depth':range(3,10,2), 'min_child_weight':range(1,9,2)}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# **Tune gamma**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1,n_estimators=200, max_depth=3, min_child_weight=5,
                   subsample=0.8,colsample_by_tree=0.8)
hyper_params = {'gamma':[0,0.1,0.2,0.3,0.4,0.5]}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# **Tune subsample and colsample_by_tree**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1, n_estimators=200, max_depth=3, min_child_weight=5,
                   gamma = 0)
hyper_params = {'subsample':[0.6,0.7,0.8,0.9], 'colsample_by_tree':[0.6,0.7,0.8,0.9]}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# **Tune regularization parameter alpha**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1, n_estimators=200, max_depth=3, min_child_weight=5,
                   gamma = 0, subsample=0.8, colsample_by_tree=0.6)
hyper_params = {'reg_alpha':[0.0001,0.001,0.01,0.1,1,10]}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# **Tune regularization parameter lambda**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1, n_estimators=200, max_depth=3, min_child_weight=5,
                   gamma = 0, subsample=0.8, colsample_by_tree=0.6,reg_alpha=0.001)
hyper_params = {'reg_lambda':[0.0001,0.001,0.01,0.1,1,10]}

grid_search_cv(hyper_params, xgb, X_train, y_train, X_test, y_test)


# **Building a model with above tuned parameters**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.1, n_estimators=200, max_depth=3, min_child_weight=5,
                   gamma = 0, subsample=0.8, colsample_by_tree=0.6,reg_alpha=0.001, reg_lambda=1)
xgb.fit(X,y)

y_pred = xgb.predict(X)
print ("Train error: ", root_mean_squared_error(y, y_pred))


# **Trying a lower learning rate 0.05 and increasing the number of estimators to 400 proportionally**

# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.05, n_estimators=400, max_depth=3, min_child_weight=5,
                   gamma = 0, subsample=0.8, colsample_by_tree=0.6,reg_alpha=0.001, reg_lambda=1)
xgb.fit(X,y)

y_pred = xgb.predict(X)
print ("Train error: ", root_mean_squared_error(y, y_pred))


# In[ ]:


xgb = XGBRegressor(random_state=100, learning_rate=0.01, n_estimators=2000, max_depth=3, min_child_weight=5,
                   gamma = 0, subsample=0.8, colsample_by_tree=0.6,reg_alpha=0.001, reg_lambda=1)
xgb.fit(X,y)

y_pred = xgb.predict(X)
print ("Train error: ", root_mean_squared_error(y, y_pred))


# **Considering the above parameters and make a final model**

# In[ ]:


xgb_final = XGBRegressor(random_state=100, learning_rate=0.01, n_estimators=2000, max_depth=3, min_child_weight=5,
                   gamma = 0, subsample=0.8, colsample_by_tree=0.6,reg_alpha=0.001, reg_lambda=1)
xgb_final.fit(X,y)

y_pred = xgb_final.predict(X)
print ("XGBRegressor rmse: ", root_mean_squared_error(y, y_pred))


# ### Model 4 - Elastic Net Regression

# In[ ]:


elastic_net = Pipeline([('scaler',RobustScaler()), ('model',ElasticNet(random_state=100, normalize=True))])
hyper_params = [{'model__alpha':[0.0001, 0.001,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 1, 2, 
                    5, 10, 50], 'model__l1_ratio':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search_cv(hyper_params, elastic_net, X_train, y_train, X_test, y_test)


# **Making a final elastic net model**

# In[ ]:


elastic_net_final = Pipeline([('scaler',RobustScaler()), 
                              ('model',ElasticNet(random_state=100, normalize=True, alpha=0.0001, l1_ratio=0.9))])
elastic_net_final.fit(X, y)

y_pred = elastic_net_final.predict(X)
print ("ElasticNet rmse: ", root_mean_squared_error(y, y_pred))


# ### Model 5 - Gradient Boosting Regressor

# **Fix learning rate and tree parameters for tuning of n_estimators**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.1,subsample=0.8,min_samples_split=150,
                               min_samples_leaf=50, max_depth=5,max_features='sqrt')
hyper_params = {'n_estimators':[50,150,200,250,300,350,400]}

grid_search_cv(hyper_params, grb, X_train, y_train, X_test, y_test)


# **Tune max_depth and min_samples_split**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.1, n_estimators=350, subsample=0.8,
                                min_samples_leaf=50, max_features='sqrt')
hyper_params = {'max_depth':range(3,16,2), 'min_samples_split':range(150,300,50)}

grid_search_cv(hyper_params, grb, X_train, y_train, X_test, y_test)


# **Tune min_samples_leaf**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.1, n_estimators=350, max_depth=5, 
                                min_samples_split=150, subsample=0.8, max_features='sqrt')
hyper_params = {'min_samples_leaf': range(50,101,10)}

grid_search_cv(hyper_params, grb, X_train, y_train, X_test, y_test)


# **Tuning subsample**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.1, n_estimators=350, max_depth=5, 
                                min_samples_split=150, min_samples_leaf=50, max_features='sqrt')

hyper_params = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
grid_search_cv(hyper_params, grb, X_train, y_train, X_test, y_test)


# **Making a model with above tuned parameters**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.1, n_estimators=350, max_depth=5, 
                                min_samples_split=150, min_samples_leaf=50, max_features='sqrt', subsample=0.75)

grb.fit(X, y)
y_pred = grb.predict(X)
print ("grb rmse: ", root_mean_squared_error(y, y_pred))


# **Trying a learning rate of half 0.05 and doubling n_estimators proportionally to 700**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.05, n_estimators=700, max_depth=5, 
                                min_samples_split=150, min_samples_leaf=50, max_features='sqrt', subsample=0.75)

grb.fit(X, y)
y_pred = grb.predict(X)
print ("grb rmse: ", root_mean_squared_error(y, y_pred))


# **Trying a learning rate of 0.01 and n_estimators to 3500**

# In[ ]:


grb = GradientBoostingRegressor(random_state=100, learning_rate=0.01, n_estimators=3500, max_depth=5, 
                                min_samples_split=150, min_samples_leaf=50, max_features='sqrt', subsample=0.75)

grb.fit(X, y)
y_pred = grb.predict(X)
print ("grb rmse: ", root_mean_squared_error(y, y_pred))


# **Making a final model of Gradient Boosting Regressor**

# In[ ]:


grb_final = GradientBoostingRegressor(random_state=100, learning_rate=0.01, n_estimators=3500, max_depth=5, 
                                min_samples_split=150, min_samples_leaf=50, max_features='sqrt', subsample=0.75)

grb_final.fit(X, y)
y_pred = grb_final.predict(X)
print ("grb rmse: ", root_mean_squared_error(y, y_pred))


# ### Model 6 - StackingCVRegressor

# In[ ]:


stacked = StackingCVRegressor(regressors=(lasso_final, ridge_final, elastic_net_final, grb_final), 
                              meta_regressor=xgb_final, use_features_in_secondary=True)
stacked.fit(X, y)
y_pred = stacked.predict(X)
print ("StackingCVRegressor rmse: ", root_mean_squared_error(y, y_pred))


# ### Model 7 - ensemble/blending

# In[ ]:


def get_blended_predictions(df):
    return (
                (0.05 * lasso_final.predict(df)) +                
                (0.05 * ridge_final.predict(df)) +
                (0.05 * elastic_net_final.predict(df)) +
                (0.15 * grb_final.predict(df)) + 
                (0.25 * xgb_final.predict(df)) +                
                (0.45 * stacked.predict(df)) 
           )


# In[ ]:


y_pred = get_blended_predictions(X)
print ("Blended model rmse: ", root_mean_squared_error(y, y_pred))


# ### Step 5 - Predictions on test data set

# In[ ]:


housing_test_df.pop('SalePrice')
housing_test_pred = get_blended_predictions(housing_test_df)

housing_test_submission = pd.DataFrame()
housing_test_submission['Id'] = test_Ids
housing_test_submission['SalePrice'] = np.exp(housing_test_pred)
housing_test_submission.to_csv('./submission.csv', index=False)


# In[ ]:




