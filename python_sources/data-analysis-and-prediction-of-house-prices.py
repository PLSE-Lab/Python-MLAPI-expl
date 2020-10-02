#!/usr/bin/env python
# coding: utf-8

# This notebook has next structure:
# - environment setup 
# - data preparation and feature engineering
# - model built

# # Setup

# Setting the environment by adding needed libraries. Some extra scikit libraries will be added at the later stages. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


#As dataset has a lot of columns, let's change the default display options
#Doing that we will have full picture of our dataframe structure
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# Uploading and checking data. 
# *Note:* We have 'Id' column at a position 0 in our dataset. This column is not informative within further analysis, but I don't want to drop it, since it is needed for submission. I decided to use this column as main index.

# In[ ]:


houses_train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 0)
houses_test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col = 0)


# First of all let's take a look on a dataset and short information about it. From that point we can find out data types that are used, memory usage, missing data. etc.

# In[ ]:


houses_train_data.head()


# In[ ]:


houses_train_data.info()


# In[ ]:


houses_test_data.head()


# In[ ]:


houses_test_data.info()


# # Handling of the missing data

# Checking how many empty rows there are in dataset:

# In[ ]:


houses_train_data.isnull().sum().sort_values(ascending = False)


# In[ ]:


houses_test_data.isnull().sum().sort_values(ascending = False)


# We can see that we have a lot of columns with missing values. Let's analyse them.
# 
# - There are 5 parameters, where near or more then 50% of data are absent:  'Alley' (1369 and 1352 null values in train and test datasets), 'PoolQC' (1453/1456 null values), 'Fence' (1179/1169 null values), 'MiscFeature' (1406/1408 null values), 'FireplaceQu' (690/730 null values). Since we cannot recover all the missing data having less then half known information, the best option here is to remove these columns from our analysis. 
# - I assume that 'LotFrontage' parameter is correlated with 'LotArea' parameter and is less important one. If it is so, then we'll able to remove 'LotFrontage' from the dataset, as 'LotArea' handels affection on sales price. Let's check this relation before we make a decision.
# - In our dataset we have multiple garage-related parameters: 'GarageArea', 'GarageCars', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt'. We can see that the number of missing values is the same for last 5 parameters (81 and 78 in train and test datasets), which means that these parameters are missing for the same houses. These values can be absent as garage condition is not relevant at all (so people didn't provide this information) or if garage is still important for buyer, then most probably he will check 'GarageArea' or 'GarageCars'. At this stage I'm goint to remove redundant garage details ('GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt') and I'll also check correlation between 'GarageArea' and 'GarageCars' and will leave only one parameter, if correlation has place.
# - There are also several columns related to basement: 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF'. Let's analyse them separately and remove all the redundant ones.

# In[ ]:


# Checking correlation between LotFrontage and LotArea.

houses_train_data[['LotFrontage', 'LotArea']].corr()


# Some correlation has place there. Let's remove 'LotFrontage' for now and check how accurate our model will be. 

# In[ ]:


# Checking correlation between Garage Cars and Garage Area.

houses_train_data[['GarageCars', 'GarageArea']].corr()


# Correlation between Garage Cars and Garage Area is quite strong, so we can say that multicollinearity has place there. Multicollinearity cannot reduce accuracy of sales predictions in general, but it can affect analysis of every single parameter separately. Because of that I'm going to remove numerical parameter 'GarageArea' and leave categorical varaible 'GarageCars'.
# 
# There is also one missing value in 'GarageCars' column. Let's look at that row and check the distribution:

# In[ ]:


houses_test_data[houses_test_data['GarageCars'].isnull()]


# In[ ]:


houses_test_data['GarageCars'].value_counts()


# Knowing that house ID 2577 has a garage of a detached type and most often garages are built for 2 cars, let's fill a gap with a most possible option - 2. 

# To check if basement parameters affect sales price, I'm going to find correlation between 'SalePrice' and 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF'. I'm going to use 'get_dummies' method to include categorical variables in this analysis:

# In[ ]:


pd.get_dummies(houses_train_data[['SalePrice', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 
                                  'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 'BsmtFullBath', 
                                  'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']]).corr()


# We can see that next variables affect sales price the most: 'BsmtFinSF1', 'BsmtFinType1', 'TotalBsmtSF', 'BsmtExposure', 'BsmtQual'. However, 'BsmtFinSF1' and 'BsmtFinType1' are also correlated with 'TotalBsmtSF' and 'BsmtExposure', which can cause some distortions in analysis of individual parameters. So for now let's leave only 3 the most independent and important variables: 'TotalBsmtSF', 'BsmtExposure', 'BsmtQual'. 
# 
# For missing values in these columns I'm going to do the next steps:
# - gaps in 'TotalBsmtSF' I'll fill with '0'.
# - gaps in 'BsmtExposure' and 'BsmtQual' I'll fill with 'Not specified'.

# In[ ]:


# Now let's check how 'MasVnrType' and 'MasVnrArea' affect sales price. 

pd.get_dummies(houses_train_data[['SalePrice', 'MasVnrArea', 'MasVnrType']]).corr()


# All the parameters are inter-correlated, so they may affect sales price and our further predictions, however it will be better to leave only one variable to avoid collinearity. Here I'm going to leave 'MasVnrArea' as it has higher correlation with both 'SalePrice' and 'MasVnrType'. To reduce the negative affect of missing values, I'll fill gaps with mean 'MasVnrArea' value. 

# Let's check variables, that describe conditions and quality of the house:

# In[ ]:


pd.get_dummies(houses_train_data[['SalePrice', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
                                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'KitchenQual', 
                                  'SaleCondition' ]]).corr()


# Biggest correlation is between 'SalePrice' and 'OveralQual', which is logical and makes 'OveralQual' parameter important is our analysis.
# There are also other variables, that have strong correlation with sales price, however they also correlate with 'OveralQual' variable. This means, that if we include both 'OveralQual' and other variables, we will include multicollinearity in our analysis. Let's avoid that and leave only one quality parameter. 

# For now there are still few more gaps in our data:
# - in train set we have one gap in 'Electrical' column
# - in test set we have several gaps in 'MSZoning', 'Utilities', 'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd' columns
# 
# Let's check how these variables affect sales price.

# In[ ]:


pd.get_dummies(houses_train_data[['SalePrice', 'Electrical', 'MSZoning', 'Utilities', 
                                  'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd']]).corr()


# We can see that there is only minor impact of this variables on sales price. 
# - Of course, there is some correlation between SalePrice and SaleType_New, but the same relation is present between 'SalePrice' - 'YearBuilt' (scatter plot is presented below).
# - Some correlation between SalePrice and Exterior has also place, but let's remove it for now to make analysis simpler. If model won't be precise enough, we can add it back.

# In[ ]:


# Checking Year-Price dependency
print((houses_train_data[['YearBuilt', 'SalePrice']].corr()))
sns.scatterplot(houses_train_data['YearBuilt'], houses_train_data['SalePrice'])


# In[ ]:


#dropping columns, where most of parameters are missing or weak correlation was detected
houses_train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage', 
                        'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 
                        'GarageArea', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 
                        'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'MasVnrType', 
                        'OverallCond','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'KitchenQual', 'SaleCondition', 'Electrical', 'MSZoning', 'Utilities', 
                        'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd'], axis = 1, inplace = True)
houses_test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage', 
                        'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 
                        'GarageArea', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 
                        'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'MasVnrType', 
                        'OverallCond','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'KitchenQual', 'SaleCondition', 'Electrical', 'MSZoning', 'Utilities', 
                        'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd'], axis = 1, inplace = True)

#filling the gaps accordingly to notes above
houses_train_data.fillna({'TotalBsmtSF': 0, 'MasVnrArea': houses_train_data['MasVnrArea'].mean()}, 
                         inplace = True)
houses_test_data.fillna({'TotalBsmtSF': 0, 'GarageCars': 2, 
                         'MasVnrArea': houses_test_data['MasVnrArea'].mean()}, inplace = True)


# In[ ]:


#Confirming, that we do not have missing values in train dataset any more.
houses_train_data.isnull().sum().sort_values(ascending = False).head()


# In[ ]:


#Confirming, that we do not have missing values in test dataset any more.
houses_test_data.isnull().sum().sort_values(ascending = False).head()


# # Data analysis

# Let's take a look on sales price distribution

# In[ ]:


print(houses_train_data['SalePrice'].describe())
houses_train_data['SalePrice'].hist(bins = 30)


# Minimal house price is 34900 USD, maximal - 755000 USD. However, having 75-th percentile on 214000 USD we can say, that only small amount of houses are really expensive. Mostly price is between 163000 and 214000 USD with mean 180921 USD. Similar picture we expect to see in the predictions. 

# In[ ]:


houses_train_data.columns


# Now we have ~50 columns (independent parameters), which is way too much for our model. We need to find out, which parameters are the most important to make further analysis easier.
# 
# 
# Let's concatenate dataset to make analysis easier and then check distribution of some variables.

# In[ ]:


all_houses = pd.concat([houses_train_data, houses_test_data])


# In[ ]:


all_houses.dropna().nunique()


# Let's check those variables, where we have only 2 values.

# In[ ]:


sns.countplot(all_houses['Street'])


# Most of the houses are on Pave street... Let's check if Street = Grvl affects sales price. 

# In[ ]:


all_houses[all_houses['Street'] == 'Grvl']


# There are only 6 houses on Grvl street (with known sales price) and all the prices are around our mean value. 
# 
# I can conclude that Street parameter is not informative and I'm going to remove it from our analysis.

# In[ ]:


houses_train_data.drop(['Street'], axis = 1, inplace = True)
houses_test_data.drop(['Street'], axis = 1, inplace = True)


# In[ ]:


sns.countplot(all_houses['LandSlope'])


# In[ ]:


sns.boxplot(all_houses['LandSlope'], all_houses['SalePrice'])


# As most of the houses have LandSlope = Glt and mean values for all the LandSlope types are similar, I'll also remove this variable. 

# In[ ]:


houses_train_data.drop(['LandSlope'], axis = 1, inplace = True)
houses_test_data.drop(['LandSlope'], axis = 1, inplace = True)


# In[ ]:


sns.countplot(all_houses['CentralAir'])


# In[ ]:


sns.boxplot(all_houses['CentralAir'], all_houses['SalePrice'])


# Boxplot shows that there is a significant difference between those two types of 'CentralAir', so we cannot remove this variable. 
# 
# However let's replace string values Yes/No with numbers 1/0, so it would be easier to analyse the dataset.

# In[ ]:


houses_train_data['CentralAir'].replace(['Y', 'N'], [1, 0], inplace = True)
houses_test_data['CentralAir'].replace(['Y', 'N'], [1, 0], inplace = True)


# In[ ]:


houses_train_data.columns


# Let's check how the rest of variable affect sales price.

# In[ ]:


houses_train_data[['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 
                                'MasVnrArea', 'SalePrice']].corr()


# There are no strong dependency between sales price and 'MSSubClass' or 'LotArea', so I'll remove these variables. 
# 
# There is also very similar picture for 'YearBuilt' and 'YearRemodAdd' (extra plot presented below), so there is no need to have both variables. I'm going to leave only 'YearBuilt'.

# In[ ]:


fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 1, 1])

axes.plot(houses_train_data['YearBuilt'], houses_train_data['SalePrice'], 'b.', alpha = 0.5)
axes.plot(houses_train_data['YearRemodAdd'], houses_train_data['SalePrice'], 'r.', alpha = 0.5)
axes.set_xlabel('YearBuilt (blue), YearRemodAdd (red)')
axes.set_ylabel('SalePrice')


# In[ ]:


houses_train_data[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'SalePrice']].corr()


# 1. Sales price correlates with 'TotalBsmtSF', '1stFlrSF', '2stFlrSF' and 'GrLivAres'. However, 'TotalBsmtSF' is also highly correlated with '1stFlrSF', so from this two parameters we will use only 'TotalBsmtSF' further.
# 2. There is no correlation with 'LowQualFinSF', so this variable will be removed.

# In[ ]:


houses_train_data[['FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                   'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'SalePrice']].corr()


# From this batch I'm going to remove 'HalfBath', 'BedroomAbvGr' and 'KitchenAbvGr' as they do not really impact sales price.

# In[ ]:


houses_train_data[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']].corr()


# From all the presented varaibles there is some relation of sales price only with 'WoodDeckSF' and 'OpenPorchSF'. The rest are more chaotically distributed and I'm going to remove them from the analysis

# There are also few categorical varaibles that need to be analysed: 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'HeatingQC', 'PavedDrive'. Let's replace their string values with numerical and check the correlation.

# In[ ]:


# List of columns, where we need to replace values
categ_columns = ['LotShape', 'LandContour', 'LotConfig',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'HeatingQC', 'PavedDrive']


# In[ ]:


# Loop that replaces string values with numbers
for item in categ_columns:
    to_replace = houses_train_data[item].unique()
    values = list(range(len(pd.Series(houses_train_data[item].unique()))))
    houses_train_data[item].replace(to_replace, values, inplace = True)
    houses_test_data[item].replace(to_replace, values, inplace = True)


# In[ ]:


# Checking correlation
houses_train_data[['LotShape', 'LandContour', 'LotConfig',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'HeatingQC', 'PavedDrive', 'SalePrice']].corr()


# We can see that 'Foundation' and 'HeatingQC' has highest correlation with sales price (~ -0.42). The rest of variables do not impact the price, so we are going to remove them. 

# In[ ]:


#dropping columns which weak impact on sales price
houses_train_data.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MSSubClass',
                        'LotArea', 'YearRemodAdd', 'LowQualFinSF', '1stFlrSF', 'HalfBath', 'BedroomAbvGr',
                        'KitchenAbvGr', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
                        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Heating', 
                        'PavedDrive'], axis = 1, inplace = True)
houses_test_data.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MSSubClass',
                       'LotArea', 'YearRemodAdd', 'LowQualFinSF', '1stFlrSF', 'HalfBath', 'BedroomAbvGr',
                       'KitchenAbvGr', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
                       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Heating', 
                       'PavedDrive'], axis = 1, inplace = True)


# In[ ]:


houses_train_data.columns


# Let's analyse time data separately.

# In[ ]:


houses_train_data['Date'] = pd.to_datetime(houses_train_data['MoSold'].astype('str') + ' ' + houses_train_data['YrSold'].astype('str'))
houses_test_data['Date'] = pd.to_datetime(houses_test_data['MoSold'].astype('str') + ' ' + houses_test_data['YrSold'].astype('str'))


# In[ ]:


plt.figure(figsize = (12, 6))
sns.lineplot(houses_train_data['Date'], houses_train_data['SalePrice'])


# We can see that second part of the year is more efficient in terms of sales, but difference is not very significant.
# 
# Let's remove redundant Month and Year columns from datasets:

# In[ ]:


# Dropping columns which month and year
houses_train_data.drop(['MoSold', 'YrSold'], axis = 1, inplace = True)
houses_test_data.drop(['MoSold', 'YrSold'], axis = 1, inplace = True)

# Saving date in different variables
dates_in_houses_train_data = houses_train_data.pop('Date')
dates_in_houses_test_data = houses_test_data.pop('Date')


# In[ ]:


# Let's check that all variables affect sales price using heatmap and correlation values
plt.figure(figsize = (10, 10))
sns.heatmap(houses_train_data.corr())


# In[ ]:


houses_train_data.corr()


#  

# In[ ]:


# Uploading prediction libraries
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# In[ ]:


# Test split for model validation
X = houses_train_data.drop(['SalePrice'], axis = 1)
y = houses_train_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 100)


# In[ ]:


# Creating classificators
forest_reg = RandomForestRegressor()
booster_reg = GradientBoostingRegressor()


# In[ ]:


# Setting parameters for further search
# parameters for forest_reg
parameters_forest_reg = {'n_estimators': range(50, 300, 50), 'max_depth': range(5, 30, 2)}
# parameters for booster_reg
parameters_booster_reg = {'n_estimators': range(50, 300, 50)}


# In[ ]:


# Searching for best classificator settings
search_forest_reg = GridSearchCV(forest_reg, parameters_forest_reg, cv = 5)
search_booster_reg = GridSearchCV(booster_reg, parameters_booster_reg, cv = 5)


# In[ ]:


search_forest_reg.fit(X_train,y_train)
search_booster_reg.fit(X_train,y_train)


# In[ ]:


# Checking best settings

best_forest_reg = search_forest_reg.best_estimator_
best_booster_reg = search_booster_reg.best_estimator_

print(best_forest_reg)
print(best_booster_reg)


# In[ ]:


# Let's add random_state parameter and tune model a little bit more
best_forest_reg = RandomForestRegressor(max_depth = 17, n_estimators = 100, random_state = 50)
best_booster_reg = GradientBoostingRegressor(n_estimators = 100, random_state = 50)


# In[ ]:


# Making predictions
best_forest_reg.fit(X_train, y_train)
best_booster_reg.fit(X_train, y_train)
forest_prediction = best_forest_reg.predict(X_test)
booster_prediction = best_booster_reg.predict(X_test)


# In[ ]:


# Cheching distribution of differences
(y_test - forest_prediction).hist(alpha = 0.5, bins = 30)
(y_test - booster_prediction).hist(alpha = 0.5, bins = 30)


# Both models are quite similar, but let's choose GradientBoostingRegressor for submission.

# In[ ]:


X_train = houses_train_data.drop(['SalePrice'], axis = 1)
y_train = houses_train_data['SalePrice']
X_test = houses_test_data


# In[ ]:


best_booster_reg.fit(X_train, y_train)
booster_prediction = best_booster_reg.predict(X_test)


# In[ ]:


# Sales prices distribution in original dataset
houses_train_data['SalePrice'].describe()


# In[ ]:


# Distribution of predicted sales prices
pd.Series(booster_prediction).describe()


# In[ ]:


submission = pd.DataFrame({'Id': houses_test_data.index, 'SalePrice': booster_prediction})
submission.head()


# In[ ]:


submission.to_csv('/kaggle/working/submission.csv', index = False)

