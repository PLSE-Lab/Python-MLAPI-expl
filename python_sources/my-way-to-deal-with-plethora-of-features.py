#!/usr/bin/env python
# coding: utf-8

# ## Acknowledgement : 
# 
# Hello everyone, It's been aroud 5 months since I started on Machine Learning and I must admit that kaggle has been the best source of knowledge. I am still going through lots of fabulous kernels and learning various aspects of  ML. So thanks to the kaggle and all kagglers for making this learning easy. I will be glad to have your comments on this work. 
# 
# Below are some exceptionally good work that tought me many things in feature engineering, model training and cross validation.
# 
# 1. https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 2. https://www.kaggle.com/apapiu/regularized-linear-models
# 

# ## Import required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for plot visualization
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, ElasticNet
from lightgbm import LGBMRegressor

color = sns.color_palette()
sns.set_style('darkgrid')

import os
print(os.listdir("../input"))


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/train.csv')


# Let's checkout how our dataset looks like

# In[ ]:


print(f'Train dataset shape (Rows, Columns) - {train_data.shape}')
train_data.head(5)


# In[ ]:


print(f'Test dataset shape (Rows, Columns) - {test_data.shape}')
test_data.head(5)


# In[ ]:


train_data.describe()


# Before we go any further lets drop 'Id' column, since it is of no use for us

# In[ ]:


train_data.drop(columns='Id', inplace=True)
test_data.drop(columns='Id', inplace=True)


# ## Data Visualization

# Okay, so the first thing we should do is to perform various visulizations on our dataset, this is the best way to know the trends and behaviour of different predictors. 
# 
# I must admit that when I started working on this solution, I jumped directly to the data preprocessing before visualization which was not the prudent choice since that way I had to perform cleaning and other preprocessing again after visualization which was bit untidy and confusing.

# ### Univariate Analysis : Dist Plot (for numerical features)
# First we are listing out all numercal and categorical features in separate variables. 

# In[ ]:


numerical_variables = train_data.select_dtypes(include=[np.number]).columns
categorical_variables = train_data.select_dtypes(include=[np.object]).columns


# Let's plot a distplot for all the numerical variables. *Here, I wanted to show the skewness for each distplot as a label but coult not understand where to place plt.show() to make it work. Please comment if you know the solution.* 

# In[ ]:


def distplot(value, **kwargs):
    sns.distplot(value, color='teal', label=f'skewness: {value.skew()}')
    plt.legend()
    plt.xticks(rotation=90)
#     plt.show()
    
melted_df = pd.melt(train_data, value_vars = numerical_variables)
facet_grid_df = sns.FacetGrid(melted_df, col='variable', col_wrap=4, sharex=False, sharey = False, height=5)
facet_grid_df.map(distplot, 'value')


# My observations from these distplots are:
# *     LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea, OpenPorchSF and SalePrice are right skewed and GarageYrBlt is left skewed. Later we have to log transform these features.
# *     BsmtFinSF2, LowQualFinSF, BsmtHalfBath, EnclosedPorch, 3SsnPorch, ScreePorch, PoolArea, MiscVal contains mostly 0 values so they are not exactly going to make a big imapct on our predictions, we will see later how to deal with these features.
# *     Very few records are there for which SalePrice is beyond 500000. It will be too soon to consider them as outliers. We will check them later.

# ### Multivariate Analysis: Scatter Plot
# Lets draw scatter plots for visualizing the relationship between SalePrice and other numeric features

# In[ ]:


def scatterplot(x, y,**kwargs):
    sns.scatterplot(x=x, y=y, color='teal')
    plt.xticks(rotation=90)
    
melted_df = pd.melt(train_data, id_vars='SalePrice', value_vars = numerical_variables[:-1])
facet_grid_df = sns.FacetGrid (melted_df, col='variable', col_wrap=4, sharex=False, sharey=False, height=5)
facet_grid_df.map(scatterplot, 'value','SalePrice')


# My observation after analysing these scatter plots are : 
# 
# * Scatters MSSubClass shows that, Houses that have '1-STORY 1946 & NEWER ALL STYLES' & '60	2-STORY 1946 & NEWER' type dwellings are common and are availabe in all range of houses. 
# * LotFrontage, MasVnrArea, BsmtFinSF1 has a week linear positive relationship with SalePrice, So we can say that our models will barely get the advantage of these feature.
# * 1stFlrSF seems to have a strong linear relationship with target variable, same goes with TotalBsmtSF, GrLiveArea and GarageArea. They are going to be strong predictors.
# * From YearBuilt, YearRemoteAdd and GarageYrBlt it is clear that the house built after year 2000 are in good demand with higher SalePrice.
# * Price increases with OveralQuality of the house, which is very obvious, same goes with OverallCond.
# * Since most of the houses doesn't have a PoolArea and 3SsnPorch, So it also doesn't really affects the SalePrice.
# * YrSold and MoSold doesn't make any noticable impact on SalePrice
# 

# ### Multivariate Analysis: Bar Plot (for categorical features)

# In[ ]:


# bar chart for visualizing the relation ship between TargetVariable and remaining categorical features
def barplot(x, y,**kwargs):
    sns.barplot(x=x, y=y)
    plt.xticks(rotation=90)

melted_df = pd.melt(train_data, id_vars='SalePrice', value_vars = categorical_variables)
facet_grid_df = sns.FacetGrid (melted_df, col='variable', col_wrap=4, sharex=False, sharey=False, height=5)
facet_grid_df.map(barplot, 'value','SalePrice')


# Here all I could observe is : 
# * HeatingQC is directly proportional to SalePrice. It is the one among effective categorical features. GarageFinish, FireplaceQuality, CentralAirCondition, ExternalQuality, ExternalCondition also leave the imapct on SalePrice as expected.
# * Uncommon LotShape doesn't really impact SalePrice. It seemed one of the considerable aspect to me.
# * People tends to pay higher price SalePrice in Partial SaleCondition, may be because of the availabe scope of personalization in property.

# ### Feature Correlations: Heatmap

# In[ ]:


correlate_mat = train_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(correlate_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
plt.subplots(figsize=(25,9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlate_mat, mask=mask, center=0.5, square=True, linewidths=.5, cbar_kws={"shrink": .5})


# From here it is quite clear that : 
# * OverallQual and GrLivArea are two most important features.
# * There are around 20 features of hight importance. I just counted dark squares ;) So we should plot a new heatmap from them along with correlation coefficient.

# In[ ]:


corr_df = pd.DataFrame(correlate_mat['SalePrice'].sort_values(ascending=False))
corr_df.rename({'SalePrice':'Feature_Correlation_With_SalePrice'}, axis=1, inplace=True)
corr_df


# In[ ]:


# correlate_mat.sort_values(by='SalePrice', ascending=False).head(20)
top_columns = correlate_mat.sort_values(by='SalePrice', ascending=False).head(20).index
plt.figure(figsize=(14,14))
sns.heatmap(train_data[top_columns].corr(), center=0.5, annot=True, fmt='.2g', square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Observations (Just check first column):
# * TotalBsmtSF and 1stFlrSF are multicollinear.
# * WoodDeckSF, 2ndFlrSF and OpenPorchSF are multicollinear as well
# 
# We need to keep only one of the multicollinear variable, will do that later. Removing multicollinearity improves our predictions for sure. 

# In[ ]:


plt.subplots(figsize=(16, 9))
sns.barplot(y=corr_df.head(20).Feature_Correlation_With_SalePrice, x=corr_df.head(20).index)
plt.xticks(rotation=90)


# ## Feature Engineering

# It will be better if we combine the records of train and test dataset together before performing any feature related opration. This way we can keep them consistent and after we get done with this part we can re-split them in train and test datasets, for which we need to remember the row count of each dataset.
# 
# We will also remove the SalePrice since it is not there in test_dataset and is a target variable. From now on we will treat that separately.

# In[ ]:


print(f'train_dataset shape - {train_data.shape}');
print(f'test_dataset shape - {test_data.shape}');

complete_df = pd.concat((train_data, test_data)).reset_index(drop=True)
complete_df.drop(['SalePrice'], axis=1, inplace=True)
train_target_variable = train_data['SalePrice']


# Lets handle multicollinearity now

# In[ ]:


complete_df.drop(labels=['1stFlrSF','2ndFlrSF','OpenPorchSF'], axis=1, inplace=True)


# ### Handling missing values

# In[ ]:


def list_and_visualize_missing_data(dataset, type):

    # Listing total null items and its percent with respect to all nulls
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = ((dataset.isnull().sum())/(dataset.isnull().count())).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data = missing_data[missing_data.Total > 0]
    print(f'{type}_dataset features count with missing values - {missing_data.shape[0]}')

    # barplot for total missing values in each column
    plt.subplots(figsize=(16, 9))
    bar = sns.barplot(x=missing_data.index, y='Total', data=missing_data)
    bar.set_xticklabels(labels=missing_data.index,rotation=90)
    bar.set_title(f'{type} Dataset Missing Records')
    return pd.DataFrame(missing_data)


# list out the totoal missing value and its percent in all train_dataset features

# In[ ]:


list_and_visualize_missing_data(complete_df, 'complete')


# Most of the missing entries are in these five features PoolQC(Pool Quality), MiscFeature(Miscellaneous feature), Alley(type of alley access), Fence(Fence Quality) and FireplaceQu(Fireplace quality)
# 
# As per my understanding, none of these features are commonly available in houses, that's why they have missing values. These features should be droped, because of two reasons 
# 
# 1. They doesn't seem to be very robust.
# 2. Most of their values are missing, So they are not going to make any noticable positive imapact anyway.

# In[ ]:


complete_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], inplace=True)


# LotFrontage seems important to me, So I am going fill its missing values with mean of the column

# In[ ]:


imputer = SimpleImputer()
complete_df.LotFrontage = imputer.fit_transform(complete_df.LotFrontage.values.reshape(-1, 1)).reshape(1,-1)[0]
# let's check if all the LotFrontage NaN values has been replaced
print(f'count of NaN values in complete_df.LotFrontage - {complete_df.LotFrontage.isnull().sum()}')


# GarageYrBlt, GarageArea and GarageCars will contain a value only if there is any garage in house, So we can replace NaN with 0.
# 
# Simillarly GarageType, GarageFinish, GarageQual, GarageCond will contain some value only if garage is there in the house, Since they are categorical type so here we can replace NaN with NA.

# In[ ]:


# 'GarageYrBlt', 'GarageArea', 'GarageCars'
for feature in ('GarageArea','GarageYrBlt', 'GarageCars'):
    complete_df[feature] = complete_df[feature].fillna(0)

for feature in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    complete_df[feature] = complete_df[feature].fillna('NA')


# Now we can look into basement related columns 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1' and 'BsmtFinType2'. Let's handle these features in a same way we handled garage related features, since all of these are categorical, So we can fill NaN with NA.

# In[ ]:


for feature in ('BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    complete_df[feature] = complete_df[feature].fillna('NA')


# In[ ]:


# Let's handle MasVnrType and MasVnrArea in similar way
complete_df['MasVnrType'] = complete_df['MasVnrType'].fillna('NA')
complete_df['MasVnrArea'] = complete_df['MasVnrArea'].fillna(0)


# BsmtFullBath, BsmtHalfBath, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF should contain 0 at the place of missing values

# In[ ]:


for feature in ('BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'):
    complete_df[feature] = complete_df[feature].fillna(0)


# Now we are left with MSZoning, Functional, Utilities, SaleType, KitchenQual, Exterior1st, Exterior2nd features, we can replace their missing values with the mode of the columns, since all of them are categorical variables.

# In[ ]:


for feature in ('MSZoning', 'Functional', 'Utilities', 'SaleType', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'Electrical'):
    complete_df[feature] = complete_df[feature].fillna(complete_df[feature].mode()[0])


# lets check if we have any missing value in our test_data

# In[ ]:


complete_df.isnull().sum().sort_values(ascending=False).head(3)


# Looks good. 

# ### Handling skewness
# In distplots, we observed that most of the numeric variables are right skewed including SalePrice, lets apply log transformation on these variables and check if that fixes skewness.

# In[ ]:


sns.distplot(np.log1p(train_target_variable))


# Certaily it fixes the skewness.

# In[ ]:


train_target_variable = np.log1p(train_target_variable)


# Let's find the skewness in other features as well

# In[ ]:


numerical_variables = complete_df.select_dtypes(include=[np.number]).columns
categorical_variables = complete_df.select_dtypes(include=[np.object]).columns


# In[ ]:


numeric_variable_skewness = complete_df.loc[:,numerical_variables[:-1]].skew().sort_values(ascending=False)
pd.DataFrame(numeric_variable_skewness)


# We have too many features that are positive skewed, and one feature (GarageYrBlt) which is negative skewed. 
# 
# **Hanling Positive Skewness :** 
# Let's log transform features that has positive skewness greater than 1. 
# > Feature[i] =  np.log1p(Feature[i])
# 
# **Hanling Negative Skewness :** 
# For negative skewed feature, lets add 1 in mode of that column and then subsctract the cell value from that and at last take log transformation.
# > Feature[i] = np.log1p((Feature.mode() + 1) - Feature[i]) 
# 
# Note that first we are converting Negative Skewed feature to Positive Skewed feature that taking log transformation. Taking log transformation on already negative skewed feature will make in more negative skewed.

# In[ ]:


numeric_variable_skewness = numeric_variable_skewness[numeric_variable_skewness > 1]
# numeric_variable_skewness.index.values
for feature in numeric_variable_skewness.index.values:
        complete_df[feature] = np.log1p(complete_df[feature])


# In[ ]:


complete_df['GarageYrBlt'] = complete_df.GarageYrBlt.apply(lambda x: np.log((complete_df.GarageYrBlt.max()+1)-x))


# In[ ]:


# let's again check the distplot for these log transformed variables
melted_df = pd.melt(complete_df, value_vars = numerical_variables[:-1])
facet_grid_df = sns.FacetGrid(melted_df, col='variable', col_wrap=4, sharex=False, sharey = False, height=5)
facet_grid_df.map(distplot, 'value')


# Seems better. However, there are few features that contains mostly 0 value, and still are right skewed like 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea' and 'MiscVal'. Let them be like this for now, later will compare model performance with and without these features and will remove if without them model works better. 

# ### Handle Categorical Variables
# We should not convert all categorical features to dummy values, because most of the categorical features contains more than 3 possible classes of values, and converting them into dummy values will create lots of extra features which will not contain much usefull information, so we will convert features with <=3 classes to dummy values and features with >3 classes to lables using 'LabelEncoder'.

# In[ ]:


print(f'complete_df shape before one-hot-encoding - {complete_df.shape}')
labelencoder = LabelEncoder()

for variable in categorical_variables:
    if complete_df[variable].value_counts().shape[0] <= 3:
        new_df = pd.get_dummies(complete_df[variable])
        new_df.columns = [f'{variable}_'+str(col) for col in new_df.columns]
        complete_df.drop(variable, axis=1, inplace=True)
        complete_df = pd.concat([complete_df, new_df], axis=1)
    
    else:
        complete_df[variable] = labelencoder.fit_transform(complete_df[variable])

print(f'complete_df shape after one-hot-encoding - {complete_df.shape}')


# ## Training Regression Models

# Here we are going to use LGBM, Lasso and XGBoost. I will make submission separately for each and at last will weighted ensemble all the predictions.
# 
# Lets re-split train and test dataset

# In[ ]:


train_ds = complete_df[:train_data.shape[0]]
test_ds = complete_df[train_data.shape[0]:]

train_ds.shape, test_ds.shape


# We are going to use train_test_split to cross check the model performance by forming a test dataset from our train dataset.

# In[ ]:


X = train_ds.values
y = train_target_variable.values
X_test_data = test_ds.values

# Since train_dataset isn't too large, So we are going to keep only 10% of the train_dataset into test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/15, random_state=123)


# In[ ]:


# function for rmse calculation for train test dataset predictions
def calculate_rmse(model, model_name):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test.reshape(1, -1)[0],y_pred))
    print(f'{model_name} RMSE - {rmse}')


# ### Lasso Regression

# In[ ]:


model_lasso = LassoCV(alphas = [0.1, 0.001, 0.0005, 0.0001]).fit(X_train, y_train)
calculate_rmse(model_lasso, 'Lasso')


# ### XGBoost

# In[ ]:


# parameters = {
#                 'learning_rate': [0.07, 0.1, 0.3],
#                 'max_depth': [3, 4, 5],
#                 'n_estimators': [400, 600, 800]
#             }

# XGB_hyper_params = GridSearchCV(estimator=XGBRegressor(), param_grid=parameters, n_jobs=-1, cv=10)

# XGB_hyper_params.fit(X_train, y_train)
# # find out the best hyper parameters
# XGB_hyper_params.best_params_


# In[ ]:


model_XGB = XGBRegressor(learning_rate=0.07, max_depth=5, n_estimators=400)
model_XGB.fit(X_train, y_train)

calculate_rmse(model_XGB, 'XGBoost')


# ### LightGBM

# In[ ]:


model_LGBM = LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.07)
model_LGBM.fit(X_train, y_train)

calculate_rmse(model_LGBM, 'LGBM')


# ### ElasticNet

# In[ ]:


# I have manually done cross validation here to decide the alpha value.
model_ENET = ElasticNet(alpha=0.002)
model_ENET.fit(X_train, y_train)

calculate_rmse(model_ENET, 'ENET')


# # Submission

# In[ ]:


# making prediction using test_dataset predictors
y_lasso_predict = model_lasso.predict(X_test_data)
y_XGB_predict = model_XGB.predict(X_test_data)
y_ENET_predict = model_ENET.predict(X_test_data)
y_LGBM_predict = model_LGBM.predict(X_test_data)

# submitting our predictions
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


def submit_predictions(predictions, model_name):
    submission['SalePrice'] = np.expm1(predictions)
    submission.to_csv(f'{model_name}_submission.csv', index=False)

    return submission.head(10)


# In[ ]:


submit_predictions(y_lasso_predict, 'lasso')


# In[ ]:


submit_predictions(y_XGB_predict, 'xgb')


# In[ ]:


submit_predictions(y_ENET_predict, 'ENET')


# In[ ]:


submit_predictions(y_LGBM_predict, 'LGBM')


# In[ ]:


ensembled_predictions = (0.3*y_lasso_predict)+(0.3*y_ENET_predict)+(0.2*y_XGB_predict)+(0.2*y_LGBM_predict)
submit_predictions(ensembled_predictions, 'ensembled_all')


# I compared all of these predictions and as expected weighted ensemble gives the least RMSE. I think this is it from my side, again will be glad to have your suggestions. Thanks
