#!/usr/bin/env python
# coding: utf-8

# # **House Prices: Advanced Regression Techniques**

# # 1.0 Exploratory analysis of data

# ## 1.1 Read the datasets

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from scipy.special import inv_boxcox
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()


# ## 1.2 Get the number of rows and columns (shape)

# In[ ]:


print(df_train.shape)
print(df_test.shape)
print(set(df_train.columns.values) - set(df_test.columns.values))


# Results:
# * The train dataset has 1.460 rows and 81 columns
# * The test dataset has 1.459 rows and 80 columns
# * The train dataset has then target variable "Sale

# ## 1.3 Get the column types

# In[ ]:


df_train.dtypes.value_counts()


# Results:
# * The dataset has 3 types of columns: strings, integers and floats

# ## 1.4 Check distinct value counts for categorical features

# In[ ]:


col_types = df_train.dtypes
unique_count = df_train.nunique()
unique_count[col_types[col_types == 'object'].index].sort_values(ascending = False)


# In[ ]:


df_train.Neighborhood.value_counts()


# In[ ]:


df_train.Exterior2nd.value_counts()


# In[ ]:


df_train.Exterior1st.value_counts()


# Results:
# * Neighborhood, Exterior2nd, Exterior1st are the features with higher distinct values
# * Some values are in few samples

# ## 1.5 Check columns with NANs

# This check should be done in both train and test dataset

# In[ ]:


df_total = pd.concat([df_train.drop('SalePrice', axis=1), df_test],axis=0)
null_cols = df_total.isnull().sum()
print(len(null_cols[null_cols > 0]))
null_cols[null_cols > 0].sort_values(ascending = False)


# Results:
# * There are 19 columns with NAs
# * Alley, FireplaceQu, PoolQC, Fence, MiscFeature have too many NA

# In[ ]:


df_total.PoolQC.unique()


# NaN values of PoolQC means that there is no pool in the property

# In[ ]:


df_total.MiscFeature.unique()


# NaN values of MiscFeature means that there is no special feature in the property

# In[ ]:


df_total.Alley.unique()


# NaN values of MiscFeature means that there is no alley in the property

# In[ ]:


df_total.Fence.unique()


# NaN values of Fence means that there is no fence in the property

# In[ ]:


df_total.FireplaceQu.unique()


# NaN values of FireplaceQu means that there is no fence in the property

# In[ ]:


df_total.LotFrontage.unique()


# NaN values of LotFrontage could be a "real" missing value or that the property is not directly connected to the street

# In[ ]:


df_total.GarageType.unique()


# NaN values of GarageType means that there is no garage in the property

# The other columns related to garage (GarageYrBlt, GarageFinish, GarageQual, GarageCond, GarageCars) are NaN or 0 when GarageType is missing

# In[ ]:


print(df_total[df_total.GarageType.isnull()]['GarageYrBlt'].unique())
print(df_total[df_total.GarageType.isnull()]['GarageFinish'].unique())
print(df_total[df_total.GarageType.isnull()]['GarageQual'].unique())
print(df_total[df_total.GarageType.isnull()]['GarageCond'].unique())
print(df_total[df_total.GarageType.isnull()]['GarageCars'].unique())


# In[ ]:


df_total[['GarageType','GarageQual','GarageCond', 'GarageYrBlt','GarageCars']][(df_total.GarageType.isnull() == False) 
    & (df_total.GarageQual.isnull() | df_total.GarageCond.isnull() |
      df_total.GarageYrBlt.isnull() | df_total.GarageCars.isnull())]


# In[ ]:


df_total.BsmtFinType1.unique()


# NaN values of BsmtFinType1 means that there is no basemente in the property. Let's see all basement related features.

# In[ ]:


df_total[['BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond','BsmtQual']][df_total.BsmtFinType1.isnull() | df_total.BsmtExposure.isnull()]


# In[ ]:


df_total.MasVnrType.unique()


# In[ ]:


df_total['MasVnrArea'][df_total.MasVnrType.isnull()]


# In[ ]:


df_total['Electrical'].unique()


# In[ ]:


df_total['MSZoning'].unique()


# In[ ]:


df_total['Functional'].unique()


# In[ ]:


df_total['BsmtHalfBath'].unique()


# In[ ]:


df_total['BsmtFullBath'].unique()


# In[ ]:


df_total['Utilities'].unique()


# In[ ]:


df_total['SaleType'].unique()


# Results:
# * PoolQC, MiscFeature, Alley, Fence, FireplaceQu could not exist in the property and there is a predefined values for NaNs 
# * NaN values of LotFrontage could be a "real" missing value or that the property is not directly connected to the street
# * When BsmtFinType1 is NaN, the property has no basement, so the other 4 features are also NaN
# * BsmtFinType2 and BsmtExposure could be NaN even if there is a basement
# * When GarageType is missing, GarageYrBlt, GarageFinish, GarageQual, GarageCond, GarageCars are NaN too
# * There are few samples where GarageType is not missing, but the other features ara NaN
# * When MasVnrType is NaN, MasVnrArea is 0
# * Electrical could be NaN, but there must be an electrical system in the property
# * MSZoning doesn't have a predefined value for NaNs
# * Functional could be NaN and has a typical value 
# * NaN values for BsmtHalfBath and BsmtFullBath could mean that there is no bathroom in the basement
# * Utilities, SaleType, KitchenQual, BsmtUnfSF, BsmtFinSF1, BsmtFinSF2, Exterior1st, Exterior2nd NaNs need to be treated (ex. mode)

# ## 1.6 Check for outliers

# In[ ]:


df_train.describe().T


# In[ ]:


desc_train = df_train.describe().T
desc_train[desc_train['max'] > desc_train['mean'] + desc_train['std'] * 3].index


# In[ ]:


desc_train = df_train.describe().T
desc_train[desc_train['min'] < desc_train['mean'] - desc_train['std'] * 3].index


# Results:
# * Looking at the columns meanings and min-max values, probably these are not outliers. 

# ## 1.7 Look for columns that have too many unique values

# In[ ]:


unique_count = df_train.nunique()
unique_count[unique_count > len(df_train)*0.75]


# Results:
# * Id column must be removed

# ## 1.8 Check the correlation between features

# In[ ]:


corr_matrix = df_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr_matrix, vmax=0.8, vmin=0.05, annot=True);


# In[ ]:


plt.style.use('dark_background')
fig, axes = plt.subplots(nrows= 2,ncols = 3, figsize=(20,12))

axes[0,0].scatter(df_train['OverallQual'], df_train['SalePrice'], marker = 'o', color='red')
axes[0,1].scatter(df_train['GrLivArea'], df_train['SalePrice'], marker = 'o', color='green')
axes[0,2].scatter(df_train['GarageCars'], df_train['SalePrice'], marker = 'o', color='blue')
axes[1,0].scatter(df_train['GarageArea'], df_train['SalePrice'], marker = 'o', color='red')
axes[1,1].scatter(df_train['TotalBsmtSF'], df_train['SalePrice'], marker = 'o', color='green')
axes[1,2].scatter(df_train['1stFlrSF'], df_train['SalePrice'], marker = 'o', color='blue')

axes[0,0].set_title('Overall material and finish')
axes[0,1].set_title('Ground living area square feet')
axes[0,2].set_title('Car capacity')
axes[1,0].set_title('Garage square feet')
axes[1,1].set_title('Basement square feet')
axes[1,2].set_title('First floor square feet')

axes[0,0].set_xlabel('Rate')
axes[0,0].set_ylabel('Sale price');
axes[0,1].set_xlabel('Square feet')
axes[0,1].set_ylabel('Sale price');
axes[0,2].set_xlabel('Car capacity')
axes[0,2].set_ylabel('Sale price');
axes[1,0].set_xlabel('Square feet')
axes[1,0].set_ylabel('Sale price');
axes[1,1].set_xlabel('Square feet')
axes[1,1].set_ylabel('Sale price');
axes[1,2].set_xlabel('Square feet')
axes[1,2].set_ylabel('Sale price');


# Results:
# * The columns that are more correlated with the target are: OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF
# * For Linear Regression, I should keep one of these columns correlated columns:
#   * YearBuilt or GarageYrBlt
#   * GrLivArea or TotRmsAbvGrd
#   * GarageCars or GarageArea
#   * TotalBsmtSF or 1stFlrSF

# # 1.9 Check the distributions of continuous numeric features

# In[ ]:


fig, ax = plt.subplots(nrows= 1,ncols = 1, figsize=(10,5))
ax.hist(df_train['SalePrice'], bins=30)


# In[ ]:


fig, axes = plt.subplots(nrows= 2,ncols = 3, figsize=(20,12))
axes[0,0].hist(df_train['LotFrontage'], bins=30)
axes[0,0].set_title('LotFrontage', fontsize=12)
axes[0,1].hist(df_train['LotArea'], bins=30)
axes[0,1].set_title('LotArea', fontsize=12)
axes[0,2].hist(df_train['MasVnrArea'], bins=30)
axes[0,2].set_title('MasVnrArea', fontsize=12)
axes[1,0].hist(df_train['BsmtUnfSF'], bins=30)
axes[1,0].set_title('BsmtUnfSF', fontsize=12)
axes[1,1].hist(df_train['BsmtFinSF1'], bins=30)
axes[1,1].set_title('BsmtFinSF1', fontsize=12)
axes[1,2].hist(df_train['BsmtFinSF2'], bins=30)
axes[1,2].set_title('BsmtFinSF2', fontsize=12)


# In[ ]:


fig, axes = plt.subplots(nrows= 2,ncols = 3, figsize=(20,12))
axes[0,0].hist(df_train['TotalBsmtSF'], bins=30)
axes[0,0].set_title('TotalBsmtSF', fontsize=12)
axes[0,1].hist(df_train['1stFlrSF'], bins=30)
axes[0,1].set_title('1stFlrSF', fontsize=12)
axes[0,2].hist(df_train['2ndFlrSF'], bins=30)
axes[0,2].set_title('2ndFlrSF', fontsize=12)
axes[1,0].hist(df_train['LowQualFinSF'], bins=30)
axes[1,0].set_title('LowQualFinSF', fontsize=12)
axes[1,1].hist(df_train['GrLivArea'], bins=30)
axes[1,1].set_title('GrLivArea', fontsize=12)
axes[1,2].hist(df_train['GarageArea'], bins=30)
axes[1,2].set_title('GarageArea', fontsize=12)


# # 1.10 Looking for chances to aggregate categorical variables

# In[ ]:


df_train.groupby('Neighborhood')     .agg({'Neighborhood':'size', 'SalePrice':'mean'})     .sort_values(by='SalePrice')     .rename(columns={'Neighborhood':'count','SalePrice':'mean'}) 


# # 2. Cleaning the dataset

# ## 2.1 Drop the ID and correlated columns from train and test datasets

# In[ ]:


test_ids = df_test['Id']
df_train.drop(columns=['Id', 'GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageCars'], axis=1, inplace=True)
df_test.drop(columns=['Id', 'GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageCars'], axis=1, inplace=True)


# ## 2.2 Drop outliers

# This is the function to drop outliers for a single column

# In[ ]:


def dropColOutliers(df, col, factor):
    mean = df[col].mean()
    std = df[col].std()
    df.drop(df[df[col] > (mean + factor * std)].index, inplace=True)
    df.drop(df[df[col] < (mean - factor * std)].index, inplace=True)
    return df


# Don't drop outliers in the train dataset for now

# ## 2.3 Fill NAs

# In[ ]:


# Save mode values from train dataset
all_mode = df_train.mode()

def fillNAs(df, all_mode):
    df['PoolQC'].fillna('NA', inplace=True)
    df['MiscFeature'].fillna('NA', inplace=True)
    df['Alley'].fillna('NA', inplace=True)
    df['Fence'].fillna('NA', inplace=True)
    df['FireplaceQu'].fillna('NA', inplace=True)
    # I assume that when LotFrontage is NA, there is no street connections with property
    df['LotFrontage'].fillna(0, inplace=True)
    # Absence of garage
    df['GarageType'].fillna('NA', inplace=True)
    NA_indx = df[(df.GarageType.isnull() == False) & df.GarageFinish.isnull()].index
    df['GarageFinish'].loc[NA_indx] = all_mode['GarageFinish'][0]
    df['GarageFinish'].fillna('NA', inplace=True)
    NA_indx = df[(df.GarageType.isnull() == False) & df.GarageQual.isnull()].index
    df['GarageQual'].loc[NA_indx] = all_mode['GarageQual'][0]
    df['GarageQual'].fillna('NA', inplace=True)
    NA_indx = df[(df.GarageType.isnull() == False) & df.GarageCond.isnull()].index
    df['GarageCond'].loc[NA_indx] = all_mode['GarageCond'][0]
    df['GarageCond'].fillna('NA', inplace=True)
    NA_indx = df[(df.GarageType.isnull() == False) & df.GarageArea.isnull()].index
    df['GarageArea'].loc[NA_indx] = all_mode['GarageArea'][0]
    df['GarageArea'].fillna(0, inplace=True)    
    # Absence of basement
    df['BsmtFinType1'].fillna('NA', inplace=True)
    df['BsmtExposure'].fillna('NA', inplace=True)
    df['BsmtCond'].fillna('NA', inplace=True)
    df['BsmtQual'].fillna('NA', inplace=True)
    df['BsmtFinType2'].fillna('NA', inplace=True)
    df['TotalBsmtSF'].fillna(0, inplace=True)
    # No masonry veneer
    df['MasVnrType'].fillna('None', inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    # When Electrical is not defined, better use Mix value
    df['Electrical'].fillna('Mix', inplace=True)    
    df['MSZoning'].fillna('NA', inplace=True)
    df['Functional'].fillna('Typ', inplace=True)
    df['BsmtHalfBath'].fillna(0, inplace=True)
    df['BsmtFullBath'].fillna(0, inplace=True)
    df['Utilities'].fillna('AllPub', inplace=True)
    df['SaleType'].fillna('Oth', inplace=True)
    df['KitchenQual'].fillna(all_mode['KitchenQual'][0], inplace=True)
    df['BsmtUnfSF'].fillna(0, inplace=True)
    df['BsmtFinSF1'].fillna(0, inplace=True)
    df['BsmtFinSF2'].fillna(0, inplace=True)
    df['Exterior1st'].fillna(all_mode['Exterior1st'][0], inplace=True)
    df['Exterior2nd'].fillna(all_mode['Exterior2nd'][0], inplace=True)


# In[ ]:


fillNAs(df_train, all_mode)
fillNAs(df_test, all_mode)
print(df_train.isnull().any().any())
print(df_test.isnull().any().any())


# ## 2.4 Separate the target feature from the training dataset

# In[ ]:


target = df_train['SalePrice']
df_train.drop('SalePrice', axis=1, inplace=True)


# # 3. Feature Engineering

# ## 3.1 Aggregations

# In[ ]:


def aggregation(df):
    df['Neighborhood'] = df['Neighborhood'].map(
        {'MeadowV': 0,
        'IDOTRR': 0,
        'BrDale': 0,
        'BrkSide': 0,
        'Edwards': 0,
        'OldTown': 0,
        'Sawyer': 0,
        'Blueste': 0,
        'SWISU': 0,
        'NPkVill': 0,
        'NAmes': 0,
        'Mitchel': 0,
        'SawyerW': 0,
        'NWAmes': 0,
        'Gilbert': 0,
        'Blmngtn': 0,
        'CollgCr': 0,
        'Crawfor': 1,
        'ClearCr': 1,
        'Somerst': 1,
        'Veenker': 1,
        'Timber': 1,
        'StoneBr': 2,
        'NridgHt': 2,
        'NoRidge': 2})
    
    df['MSZoning'] = df['MSZoning'].map(
        {'A': 0,
         'C': 0,
         'FV': 1,
         'I': 0,
         'RH': 1,
         'RL': 1,
         'RP': 1,
         'RM': 1})
    
    df['TotFullBath'] = df['BsmtFullBath'] + df['FullBath']
    df['TotHalfBath'] = df['BsmtHalfBath'] + df['HalfBath']
    
    df['HouseStyle_stories'] = df['HouseStyle'].map(
        {'1Story': 1,
         '1.5Fin': 1.5,
         '1.5Unf': 1.5,
         '2Story': 2,
         '2.5Fin': 2.5,
         '2.5Unf': 2.5,
         'SFoyer': 1.5,
         'SLvl': 1.5})

    df['HouseStyle_fin'] = df['HouseStyle'].map(
        {'1Story': 1,
         '1.5Fin': 1,
         '1.5Unf': 0,
         '2Story': 1,
         '2.5Fin': 1,
         '2.5Unf': 0,
         'SFoyer': 1,
         'SLvl': 1})


# In[ ]:


aggregation(df_train)
aggregation(df_test)


# ## 3.2 One-hot-encoding

# Change all categorical columns in binaries with one-hot-encoding for train and test dataset

# In[ ]:


col_types = df_train.dtypes
df_train = pd.get_dummies(df_train, columns=col_types[col_types == 'object'].index.values, drop_first=True)
df_test = pd.get_dummies(df_test, columns=col_types[col_types == 'object'].index.values, drop_first=True)
print(df_train.shape)
print(df_test.shape)


# Add missing column in test dataset and drop those that don't exists in trainig dataset. The difference could be caused by the one-hot-encoding

# In[ ]:


def adapt_columns(train_columns, df):
    # Add missing columns
    for column in train_columns:
        if (column not in df.columns):
            df[column] = 0

    # Delete columns that don't exist in train
    for column in df.columns:
        if (column not in train_columns):
            df.drop(column, axis=1, inplace=True)
    return df


# In[ ]:


adapt_columns(df_train.columns, df_test)
print(df_train.shape)
print(df_test.shape)


# ## 3.3 Normalization

# Rescale all features in numbers between 0 and 1

# In[ ]:


def normalization(df):
    array_val = df.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    array_norm = min_max_scaler.fit_transform(array_val)
    return pd.DataFrame(data=array_norm, columns=df.columns.values)


# In[ ]:


df_train = normalization(df_train)
df_test = normalization(df_test)


# # 4. Hyperparameters optimization

# In[ ]:


#params = {'min_child_weight': [10, 15, 20],
#          'reg_lambda': [1, 5, 10],
#          'gamma': [0.5, 0.8],
#          'max_depth': [8, 15, 20],
#          'learning_rate':[0.2],
#          'n_estimators': [20, 30, 40]}
params = {
 'max_depth': [4], #[5, 10, 15, 20, 25],
 'min_child_weight': [9],  #[5, 10, 15, 20, 25],
 'gamma': [0.0],  #[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
 'subsample': [1.0], #[0.7, 0.8, 0.9, 1.0],
 'colsample_bytree': [1.0], #[0.7, 0.8, 0.9, 1.0]
 'reg_alpha': [0.0, 0.02, 0.1, 0.5, 2, 10]
}

#model = xgb.XGBRegressor(objective='reg:linear', verbosity=0)

#xgb_tune = GridSearchCV(estimator=model, param_grid=params, cv=5, verbose=2, n_jobs=4)
#xgb_tune.fit(df_train.values, target.values)

#print(xgb_tune.best_params_)


# Difinitive hyperparameters:

# In[ ]:


xgb_params = {'max_depth': 4, 
              'min_child_weight': 9,
              'gamma': 0.0,
              'colsample_bytree': 1.0,
              'subsample': 1.0,
              'reg_alpha': 0.005,
              'learning_rate': 0.01,
              'n_estimators': 5000}


# # 5. Train the model with XGBoost Regressor

# In[ ]:


mae = 0
rmsle = 0
splits = 10

kf = KFold(n_splits=splits, shuffle=True, random_state=12345)

for train_index, test_index in kf.split(df_train):
    X_train_k, X_test_k = df_train.values[train_index], df_train.values[test_index]
    y_train_k, y_test_k = target.values[train_index], target.values[test_index]      
    
    model_k = xgb.XGBRegressor(params=xgb_params)    
    model_k.fit(X_train_k, y_train_k)
    y_pred_k = model_k.predict(X_test_k)
    
    np.round(y_pred_k)
    mae = mae + median_absolute_error(y_test_k, y_pred_k)
    #y_pred_k[y_pred_k < 0] = 0
    rmsle = rmsle + np.sqrt(mean_squared_log_error(y_test_k, y_pred_k))

print('MAE: ' + '{:.2f}'.format(mae/splits)) 
print('RMSLE: ' + '{:.4f}'.format(rmsle/splits)) 


# # 6. Train the final model and make predictions on the test dataset

# In[ ]:


model = xgb.XGBRegressor(params=xgb_params)
model.fit(df_train.values, target.values)
y_pred = model.predict(df_test.values)

df_pred = pd.DataFrame(data=y_pred, columns=['SalePrice'])
df_pred = pd.concat([test_ids, df_pred['SalePrice']], axis=1)
df_pred.SalePrice = df_pred.SalePrice.round(0)
df_pred.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)

