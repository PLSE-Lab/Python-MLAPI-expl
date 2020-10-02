#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition for House Prices: Advanced Regression Techniques 
# ***
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**

# ## Load packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', 100)


# ## Load Data

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()


# In[ ]:


test.head()


# ## Exploratory Data Analysis
# 
# In Data Analysis We will Analyze To Find out the below stuff
# 1. Missing Values
# 1. All The Numerical Variables
# 1. Distribution of the Numerical Variables
# 1. Categorical Variables
# 1. Cardinality of Categorical Variables
# 1. Outliers
# 1. Relationship between independent and dependent feature(SalePrice)

# In[ ]:


print(f"Train data shape {train.shape}")
print(f"Trest data shape {test.shape}")


# ### 1. Train Data

# In[ ]:


train.info()


# ## Missing Values

# In[ ]:


# Counts the missing values in every column and there type, using pd.info() is not helpful
# because we have 81 feature.

missing_val_obj = []
missing_val_float = []

for column in train.columns:
    if train[column].isnull().sum() != 0:
        print(f"{column} : {train[column].isnull().sum()}, {train[column].dtypes}")
        if train[column].dtypes == object:
            missing_val_obj.append(column)
        else:
            missing_val_float.append(column)


# In[ ]:


missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[ ]:


print(f"Object type columns with missing values :\n {missing_val_obj}")
print(f"Other types columns with missing values :\n {missing_val_float}")


# In[ ]:


# We will drop PoolQC, Fence, MiscVal, MiscFeature, and Alley because they have a lot of missing data
train.drop(["Alley" ,"PoolQC", "Fence", "MiscFeature", "Id"], axis="columns", inplace=True)
to_delete_features = ["Alley" ,"PoolQC", "Fence", "MiscFeature"]
for item in to_delete_features:
    missing_val_obj.remove(item)


# In[ ]:


# Filling the missing values
# using "mean" for features with type float64
for column in missing_val_float:
    train[column] = train[column].fillna(train[column].mean())


# In[ ]:


# Counts the missing values in every column and there type, using pd.info() is not helpful
# because we have 81 feature.
for column in train.columns:
    if train[column].isnull().sum() != 0:
        print(f"{column} : {train[column].isnull().sum()}, {train[column].dtypes}")


# In[ ]:


# using "mode" for features with type object
for column in missing_val_obj:
    train[column] = train[column].fillna(train[column].mode()[0])


# In[ ]:


for column in train.columns:
    if train[column].isnull().sum() != 0:
        print(f"{column} : {train[column].isnull().sum()}, {train[column].dtypes}")


# ## Numerical Values
# 
# Some of the numerical features are temporary features `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold`

# In[ ]:


num_variables = [column for column in train.columns if train[column].dtype != object]
len(num_variables)


# In[ ]:


yr_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
for col in yr_features:
    print(f"============= {col} : {train[col].nunique()} ============= \n {train[col].unique()}")


# In[ ]:


train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[ ]:


plt.figure(figsize=(12, 8))

for i, feature in enumerate(yr_features, 1):
    
    data = train.copy()
    ## We will capture the difference between year variable and year the house was sold for
    data[feature] = data['YrSold'] - data[feature]
    plt.subplot(2, 2, i)
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')


# ## Categorical Variables

# In[ ]:


cat_variables = [column for column in num_variables if train[column].nunique() < 25]
cat_variables.remove('YrSold')
len(cat_variables)


# In[ ]:


train[cat_variables].head()


# In[ ]:


plt.figure(figsize=(20, 30))

for i, feature in enumerate(cat_variables, 1):
    data = train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.subplot(6, 3, i)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)


# ## Continous Variables

# In[ ]:


disc_variables = [column for column in num_variables if train[column].nunique() > 25 and column not in yr_features]
len(disc_variables)


# In[ ]:


train[disc_variables].hist(figsize=(12, 12));


# In[ ]:


plt.figure(figsize=(12, 15))

i = 1
for feature in disc_variables:
    data = train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        plt.subplot(3, 3, i)
        i += 1
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)


# ## Outliers

# In[ ]:


plt.figure(figsize=(12, 15))

i = 1
for feature in disc_variables:
    data = train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        plt.subplot(3, 2, i)
        i += 1
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)


# In[ ]:


cat_features = [column for column in train.columns if train[column].dtype == object]
len(cat_features)


# In[ ]:


train[cat_features].head()


# In[ ]:


for column in cat_features:
    print(f"{column}: Number of unique values {train[column].nunique()}")


# ## Relation between categorical features and dependent variable

# In[ ]:


plt.figure(figsize=(20, 120))

for i, feature in enumerate(cat_features, 1):
    plt.subplot(13, 3, i)
    data = train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)


# In[ ]:


sns.distplot(train.SalePrice)


# - Deviate from the normal
# - Have appreciable positive skewness
# - Show peakedness

# In[ ]:


print(f"Skeweness: {train.SalePrice.skew()}")
print(f"Kurtosis: {train.SalePrice.kurt()}")


# In[ ]:


data = pd.concat([train.SalePrice, train.GrLivArea], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice')


# In[ ]:


data = pd.concat([train.SalePrice, train.TotalBsmtSF], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')


# In[ ]:


plt.figure(figsize=(12, 8))
data = pd.concat([train.SalePrice, train.OverallQual], axis=1)
sns.boxplot(x='OverallQual', y='SalePrice', data=data)


# In[ ]:


plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), vmax=.8, square=True)


# In[ ]:


cols = train.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(10, 8))
sns.heatmap(train[cols].corr(), annot=True, vmax=.8, square=True)


# In[ ]:


cols = train.corr().nlargest(8, 'SalePrice')['SalePrice'].index
sns.pairplot(train[cols])


# In[ ]:


train.shape


# In[ ]:


train.head()


# ### 2. Test data

# In[ ]:


missing_val_obj = []
missing_val_float = []

for column in test.columns:
    if test[column].isnull().sum() != 0:
        print(f"{column} : {test[column].isnull().sum()}, {test[column].dtypes}")
        if test[column].dtypes == object:
            missing_val_obj.append(column)
        else:
            missing_val_float.append(column)


# In[ ]:


print(f"Object type columns with missing values : {missing_val_obj}")
print(f"Other types columns with missing values : {missing_val_float}")


# In[ ]:


# We will drop PoolQC, Fence, MiscVal, MiscFeature, and Alley because they have a lot of missing data
test.drop(["Alley" ,"PoolQC", "Fence", "MiscFeature", "Id"], axis="columns", inplace=True)
to_delete_features = ["Alley" ,"PoolQC", "Fence", "MiscFeature"]
for item in to_delete_features:
    missing_val_obj.remove(item)


# In[ ]:


# Filling the missing values
# using "mean" for features with type float64
for column in missing_val_float:
    test[column] = test[column].fillna(test[column].mean())


# In[ ]:


for column in test.columns:
    if test[column].isnull().sum() != 0:
        print(f"{column} : {test[column].isnull().sum()}, {test[column].dtypes}")


# In[ ]:


# using "mode" for features with type object
for column in missing_val_obj:
    test[column] = test[column].fillna(test[column].mode()[0])


# In[ ]:


for column in test.columns:
    if test[column].isnull().sum() != 0:
        print(f"{column} : {test[column].isnull().sum()}, {test[column].dtypes}")


# In[ ]:


test.shape


# ### 3. Data Pre-processing

# In[ ]:


# from 2 features high correlated, removing the less correlated with SalePrice
train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)


# In[ ]:


#removing outliers recomended by author
train = train[train['GrLivArea'] < 4500]


# In[ ]:


train.shape


# In[ ]:


full_df = pd.concat([train, test], axis=0, sort=False)
full_df.shape


# ## Handle categorical features

# In[ ]:


# numeric features with less than 20 unique values
numeric_features = []
for column in full_df.columns:
    if train[column].dtype != object and len(train[column].unique()) < 22:
        numeric_features.append(column)

# Excluding "id", "SalePrice" columns.

# categorical_20 = ["MSSubClass", "OverallQual", "OverallCond", "LowQualFinSF", 
#                   "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", 
#                   "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "3SsnPorch", 
#                   "PoolArea", "MiscVal", "MoSold", "YrSold"]

# categorical_100 = ["LotFrontage", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "EnclosedPorch", 
#                    "ScreenPorch"]

# non_categorical = ["LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", 
#                    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", 
#                    "WoodDeckSF", "OpenPorchSF"]

print(numeric_features)


# In[ ]:


for column in numeric_features:
    encoding = full_df.groupby(column).size()
    encoding = encoding / len(full_df)
    full_df[column] = full_df[column].map(encoding)


# In[ ]:


full_df[numeric_features].head()


# In[ ]:


categorical_col = []
label_col = []
for column in full_df.columns:
    if len(full_df[column].unique()) <= 10 and full_df[column].dtypes == object:
        categorical_col.append(column)
    elif full_df[column].dtypes == object:
        label_col.append(column)


# In[ ]:


print(categorical_col)
print(label_col)


# In[ ]:


full_df = pd.get_dummies(full_df, columns=categorical_col + label_col)


# In[ ]:


# for column in label_col:
#     encoding = full_df.groupby(column).size()
#     encoding = encoding / len(full_df)
#     full_df[column] = full_df[column].map(encoding)


# In[ ]:


full_df.info()


# In[ ]:


df_train = full_df[full_df.SalePrice.notna()]
df_train.shape


# In[ ]:


df_test = full_df[full_df.SalePrice.isna()]
df_test.drop('SalePrice', axis='columns', inplace=True)


# In[ ]:


df_test.shape


# In[ ]:


X_train = df_train.drop('SalePrice', axis='columns')
y_train = df_train.SalePrice


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# ## Prediciton and selecting the Algorithm

# In[ ]:


import xgboost
# classifier = xgboost.XGBRegressor()
# classifier.fit(X_train, y_train)


# In[ ]:


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 1]
learning_rate = [0.05, 0.1, 0.15, 0.20]
min_child_weight = [1, 2, 3, 4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'learning_rate' : learning_rate,
    'min_child_weight' : min_child_weight,
    'booster' : booster,
    'base_score' : base_score
    }


# In[ ]:


# Set up the random search with 4-fold cross validation
# from sklearn.model_selection import RandomizedSearchCV
# import xgboost

# regressor = xgboost.XGBRegressor()

# random_cv = RandomizedSearchCV(estimator=regressor, param_distributions=hyperparameter_grid, cv=5, 
#                                n_iter=50, scoring = 'neg_mean_absolute_error',n_jobs = 4, 
#                                verbose = 5, return_train_score = True, random_state=42)


# In[ ]:


# regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0,
#              importance_type='gain', learning_rate=0.1, max_delta_step=0,
#              max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
#              n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, verbosity=1)


# In[ ]:


regressor = xgboost.XGBRegressor(base_score=0.25, 
                                 booster='gbtree', 
                                 learning_rate=0.1, 
                                 max_delta_step=0,
                                 max_depth=2, 
                                 min_child_weight=1, 
                                 n_estimators=900,
                                 verbosity=1)


# In[ ]:


regressor.fit(X_train, y_train)


# In[ ]:


import pickle
filename = "finalization_model.pkl"
pickle.dump(regressor, open(filename, "wb"))


# In[ ]:


y_pred = regressor.predict(df_test)


# In[ ]:


# Create Sample Submittions file and Submit
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
datasets = pd.concat([sub_df['Id'], pred], axis=1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv(path_or_buf='submission.csv', index=False)

