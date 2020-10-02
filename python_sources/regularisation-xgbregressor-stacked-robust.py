#!/usr/bin/env python
# coding: utf-8

# Using some techniques from the following kernels
# * https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# * https://www.kaggle.com/apapiu/regularized-linear-models
# * https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
# * https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# * https://www.kaggle.com/vikassingh1996/comprehensive-data-preprocessing-and-modeling
# 

# In[ ]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Limits floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 


# In[ ]:


# Files in directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#import data
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',index_col=0)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',index_col=0)

#Check size and look
print(train.shape)
train.head()


# In[ ]:


# Check size and look of data
print(test.shape)
test.head()


# In[ ]:


# Combine train and test for pre-processing
df_all = pd.concat([train[train.columns[:-1]],test])
df_all.head(5)


# In[ ]:


# Save training observations for later
y = train.SalePrice


# In[ ]:


# Number and types of columns
df_all.info()


# # Looking at final output 'SalePrice' on training data

# In[ ]:


# Looking at distribution of house prices
plt.figure(figsize=[20,5])

# Histogram plot
plt.subplot(1,2,1)
sns.distplot(y)
plt.title('Standard')

# Skewness and kurtosis
print("Skewness: %f" % y.skew())
print("Kurtosis: %f" % y.kurt())

# Due to skew (>1), we'll log it and show it now better approximates a normal distribution
plt.subplot(1,2,2)
sns.distplot(np.log(y))
plt.title('Log transformation')


# In[ ]:


# Convert y into log(y)
y_original = y.copy()
y = np.log(y)


# # Checking for missing data

# In[ ]:


# Look for missing data
plt.figure(figsize=[20,5])
sns.heatmap(df_all.isnull(),yticklabels=False,cbar=False)


# In[ ]:


# Dropping data that is heavily missing (will deal with partially missing later)
df_all.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# # Converting mislabelled 'numerical' features into categoric features

# In[ ]:


# Values for feature 'MSSubClass'
df_all.MSSubClass.unique()


# In[ ]:


# Use dictionaries to convert across
df_all = df_all.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"}})


# In[ ]:


# Values for feature 'MoSold'
df_all.MoSold.unique()


# In[ ]:


# Use dictionaries to convert into month strings
df_all = df_all.replace({"MoSold" : {1 : "January", 2 : "February", 3 : "March", 4 : "April", 
                                       5 : "May", 6 : "June", 7 : "July", 8 : "August", 
                                       9 : "September", 10 : "October", 11 : "November", 12 : "December"}})


# In[ ]:


# Check it worked as expected
df_all[['MSSubClass','MoSold']].head(5)


# # Identifying categoric and numerical variables

# In[ ]:


# No. of categoric variables
cat_feats = df_all.dtypes[df_all.dtypes == "object"].index.tolist()
print(str(len(cat_feats)) + ' categoric features')


# In[ ]:


# No. of numerical variables
num_feats = df_all.dtypes[df_all.dtypes != "object"].index.tolist()
print(str(len(num_feats)) + ' numeric features')


# # Categoric features

# ## Dealing with missing variables

# In[ ]:


# Return list of categoric columns with missing variables
cat_feats_missing = df_all[cat_feats].columns[df_all[cat_feats].isna().any()].tolist()
cat_feats_missing


# In[ ]:


# Show value occurences to determine appropriate NaN replacement
for i in cat_feats_missing:
    print(i)
    print(df_all[i].value_counts(dropna=False))
    print("")


# In[ ]:


# Make replacements into most likely value

# Likely RL
df_all.MSZoning.fillna('RL', inplace = True)
# Drop utilities as only 1 is different which doesn't help
df_all.drop('Utilities',axis=1,inplace=True)
# Likely VinylSd
df_all.Exterior1st.fillna('VinylSd', inplace = True)
# Likely VinylSd
df_all.Exterior2nd.fillna('VinylSd', inplace = True)
# Likely no masonary
df_all.MasVnrType.fillna('None', inplace = True)
# Likely no basement
df_all.BsmtQual.fillna('No basement',inplace = True)
# Likely no basement
df_all.BsmtCond.fillna('No basement',inplace = True)
# Likely no basement
df_all.BsmtExposure.fillna('No basement',inplace = True)
# Likely no basement
df_all.BsmtFinType1.fillna('No basement',inplace = True)
# Likely no basement
df_all.BsmtFinType2.fillna('No basement',inplace = True)
# Likely standard electrical
df_all.Electrical.fillna('SBrkr',inplace = True)
# Likely typical kitchen
df_all.KitchenQual.fillna('TA',inplace = True)
# Likely typical functionality
df_all.Functional.fillna('Typ',inplace = True)
# Likely no garage
df_all.GarageType.fillna('No garage',inplace = True)
# Likely no garage
df_all.GarageFinish.fillna('No garage',inplace = True)
# Likely no garage
df_all.GarageQual.fillna('No garage',inplace = True)
# Likely no garage
df_all.GarageCond.fillna('No garage',inplace = True)
# Likely typical sale type
df_all.SaleType.fillna('WD',inplace = True)

# Check it worked correctly
df_all.head(5)


# ## Converting categorical features into numerical features

# In[ ]:


# Using descriptions in 'About this file' we can order some categoric variables
df_all = df_all.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No basement" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No basement" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No basement" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )


# ## Checking for co-linearity between similar variables

# In[ ]:


# Checking for collinearity between similar variables
basement_feats = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']
exterior_feats = ['ExterCond','ExterQual']
garage_feats = ['GarageCond','GarageQual']


plt.figure(figsize=[20,5])

# basement_feats plot
plt.subplot(1,3,1)
sns.heatmap(df_all[basement_feats].corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")
plt.title('basement_feats')

# exterior_feats plot
plt.subplot(1,3,2)
sns.heatmap(df_all[exterior_feats].corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")
plt.title('exterior_feats')

# garage_feats plot
plt.subplot(1,3,3)
sns.heatmap(df_all[garage_feats].corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")
plt.title('garage_feats')


# In[ ]:


# Although 'BsmtQual' and 'ExterQual' are highly correlated, they should be independant of each other so will both stay.

# 'GarageCond'and 'GarageQual' are highly correlated. Check what's more correlated to SalePrice
train_temp = pd.concat([df_all[:train.shape[0]],y],axis=1)
train_temp[basement_feats+garage_feats + ['SalePrice']].corr()


# In[ ]:


# We'll keep 'BsmtQual' as it's more related to 'SalePrice'
df_all.drop('BsmtCond',axis=1,inplace=True)

# We'll keep 'GarageQual' as it's more related to 'SalePrice'
df_all.drop('GarageCond',axis=1,inplace=True)


# In[ ]:


# No. of categoric variables
cat_feats = df_all.dtypes[df_all.dtypes == "object"].index.tolist()
print(str(len(cat_feats)) + ' categoric features')
print("")
for i in cat_feats:
    print(i)
    print(df_all[i].value_counts(dropna=False))
    print("")


# In[ ]:


# No. of categoric variables
cat_feats = df_all.dtypes[df_all.dtypes == "object"].index.tolist()
print(str(len(cat_feats)) + ' categoric features')
df_all = pd.get_dummies(df_all)
df_all.head(5)


# # Numerical features

# ## Containing null values

# In[ ]:


null_columns=df_all.columns[df_all.isnull().any()].tolist()
df_all[null_columns].describe()


# In[ ]:


df_all[null_columns].isnull().sum()


# In[ ]:


# Some (to me) make sense to likely be 0
to_set_to_zero = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
df_all[to_set_to_zero] = df_all[to_set_to_zero].fillna(0)

# Drop GarageYrBlt as unlikely to be able to fill years, correlated to YearBuilt & has a value 2207 so question marks on it's validity...
df_all.drop('GarageYrBlt',axis=1,inplace = True)

# Replace the rest with the median of the column:
df_all = df_all.fillna(df_all.median())


# ## Correcting skewness

# In[ ]:


# Using original list of numerical features (with 'GarageYrBuilt' removed), check the skewness
num_feats.remove('GarageYrBlt')


# In[ ]:


# Calculate skewness
skewed_feats = df_all[num_feats].apply(lambda x: skew(x)) 
skewed_feats.sort_values()


# In[ ]:


# Return columns with high skewness
skewed_feats = skewed_feats[skewed_feats > 0.75].index
skewed_feats


# In[ ]:


# Annoyingly, Box-Cox is breaking for 3 features due to a precision issue (I think, see https://github.com/scipy/scipy/issues/7534)
# Therefore need to remove these 3 from list to transform
skewed_feats=skewed_feats.tolist()
skewed_feats = [e for e in skewed_feats if e not in ['GrLivArea','LotArea','1stFlrSF']]


# In[ ]:


# Transform numerical variables through the box-cox method (optimises transformation to a gaussian distribution)
# Requires a +1 as inputs need to be strictly positive
pt = PowerTransformer('yeo-johnson',standardize=False)
print(pt.fit(df_all[skewed_feats])) 
print("")

# Show lambdas to see what transformation was applied
print(pt.lambdas_)


# In[ ]:


# Insert these back into the dataframe
df_all[skewed_feats] = pt.transform(df_all[skewed_feats])

# Log the failed features
df_all[['GrLivArea','LotArea','1stFlrSF']]=df_all[['GrLivArea','LotArea','1stFlrSF']].apply(np.log)

# Read to list
skewed_feats = skewed_feats + ['GrLivArea','LotArea','1stFlrSF']

# Check the new skews (still could be improved, but a lot better than before!)
df_all[skewed_feats].skew().sort_values()


# # Separate back into train and test

# In[ ]:


X_train = df_all[:train.shape[0]]
X_test = df_all[train.shape[0]:]

# Check they are the same shape as started. They are, great.
print(X_train.shape)
print(X_test.shape)


# # Remove outliers in the training set

# In[ ]:


# Correlations with SalePrice (look to make sense)
pd.concat([X_train,y],axis=1).corr().iloc[:,-1].sort_values(ascending=False).head(10)


# In[ ]:


# Look for outliers
plt.figure(figsize=[20,5])

# 'OverallQual' plot
plt.subplot(1,2,1)
sns.scatterplot(x = X_train['OverallQual'], y = y)
plt.title('OverallQual')
plt.ylabel('SalePrice')
plt.xlabel('OverallQual')

# 'GrLivArea' plot
plt.subplot(1,2,2)
sns.scatterplot(x = X_train['GrLivArea'], y = y)
plt.title('GrLivArea')
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.grid(b=bool, which='major', axis='both')


# In[ ]:


# Will remove two values that don't match distribution on 'GrLivArea'. Turns out this is also the two outliers on OverallQual = 10 (see below)
index_to_drop = X_train[(X_train['GrLivArea']>8.3) & (y<12.5)].index.tolist()
# Remove from training feature set
X_train = X_train.drop(index_to_drop,axis=0)
# Remove from training observation set
y = y.drop(index_to_drop)

# Will also remove three values at the bottom that don't fit the pattern.
index_to_drop = X_train[(X_train['GrLivArea']>6.5) & (y<10.7)].index.tolist()
# Remove from training feature set
X_train = X_train.drop(index_to_drop,axis=0)
# Remove from training observation set
y = y.drop(index_to_drop)


# In[ ]:


# As above, checking they're gone
plt.figure(figsize=[20,5])

# 'OverallQual' plot
plt.subplot(1,2,1)
sns.scatterplot(x = X_train['OverallQual'], y = y)
plt.title('OverallQual')
plt.ylabel('SalePrice')
plt.xlabel('OverallQual')

# 'GrLivArea' plot
plt.subplot(1,2,2)
sns.scatterplot(x = X_train['GrLivArea'], y = y)
plt.title('GrLivArea')
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.grid(b=bool, which='major', axis='both')


# In[ ]:


# Check still the same
print(X_train.shape)
print(y.shape)


# # Scale features

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Scale features for improved regularisation performance

# Fit scaler (using RobustScaler to reduce effect of outliers) to training set mean and variance
scaler = RobustScaler()
scaler.fit(X_train)

# Transform both the training and testing sets
scaled_features_train = scaler.transform(X_train)
scaled_features_test = scaler.transform(X_test)

# Put scaled data back into a pandas dataframe
X_train = pd.DataFrame(scaled_features_train,columns = X_train.columns)
X_test = pd.DataFrame(scaled_features_test,index = X_test.index, columns = X_test.columns)
X_train.head(5)


# # Testing time

# In[ ]:


from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV


# ## Create training and testing subsets from the training data

# In[ ]:


# Check still the same
print(X_train.shape)
print(y.shape)


# In[ ]:


# Split train (t) into train (t_train) and test (t_test) sets.
# This allows us to evaluate the model for 'unseen' data and check for overfitting 

Xt_train, Xt_test, yt_train, yt_test, = train_test_split(X_train, y, test_size = 0.3)


# ## Define evaluation method

# In[ ]:


from sklearn.metrics import mean_squared_error, make_scorer

# Common method 1 across models

def evaluate_model(model):
    
    # Produce predictions for training and testing sets
    yt_train_pred = model.predict(Xt_train)
    yt_test_pred = model.predict(Xt_test)
    
    # Evaluate models
    rmse_train = np.sqrt(mean_squared_error(yt_train,yt_train_pred))
    rmse_test = np.sqrt(mean_squared_error(yt_test,yt_test_pred))
    print("RMSE on Training set :", rmse_train)
    print("RMSE on Test set :", rmse_test)
    
    # Graphically compare predictions on training and validation set
    plt.figure(figsize=[20,8])
    # Plot residuals
    plt.subplot(1,2,1)
    plt.scatter(yt_train_pred, yt_train_pred - yt_train, c = "blue", marker = "s", label = "Training data")
    plt.scatter(yt_test_pred, yt_test_pred - yt_test, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title(model)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc = "upper left")
    plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
    # Plot predictions
    plt.subplot(1,2,2)
    plt.scatter(yt_train_pred, yt_train, c = "blue", marker = "s", label = "Training data")
    plt.scatter(yt_test_pred, yt_test, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title(model)
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")    


# In[ ]:


# Method 2 - less obvious to see what's happening, but better for parameter hypertuning
# Decided to use the full X_train as cross validation shouldn't be testing on seen data and the more data the better the final parameter selection should be.

def cross_validation_evaluation(model):
    scores = -cross_val_score(model, Xt_train, yt_train, cv=10,scoring="neg_mean_squared_error")
    return scores.mean()


# ## Simple linear regression

# In[ ]:


# Instantiate the linear regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(Xt_train, yt_train)


# In[ ]:


# Evaluate model
evaluate_model(linear_regression_model)


# Not sure why it isn't working after scaling... we'll try to reduce this by using regularisation methods.

# ## Ridge regression

# In[ ]:


# Instantiate the linear regression model
ridge_regression_model = Ridge(alpha=1.0)
ridge_regression_model.fit(Xt_train, yt_train)


# In[ ]:


# Evaluate model
evaluate_model(ridge_regression_model)


# In[ ]:


# Test different alphas
alphas = np.logspace(start=-2,stop=2,base=10,num=30)
cv_ridge = [cross_validation_evaluation(Ridge(alpha = alpha)) 
            for alpha in alphas]
optimised_alpha = alphas[cv_ridge.index(min(cv_ridge))]
print('Optimised alpha is: ' + str(optimised_alpha))

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "alpha hypertuning")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.xscale('log')


# In[ ]:


# Instantiate the improved ridge regression model
ridge_regression_improved_model = Ridge(alpha=optimised_alpha)
ridge_regression_improved_model.fit(Xt_train, yt_train)


# In[ ]:


# Evaluate model
evaluate_model(ridge_regression_improved_model)


# In[ ]:


# Seeing what were considered important features (positive and negative)
coef = pd.Series(ridge_regression_improved_model.coef_, index = X_train.columns)
plt.figure(figsize=[20,8])
important_features = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
sns.barplot(y=important_features.index,x=important_features)


# ## Lasso regression

# In[ ]:


# Instantiate the lasso regression model
lasso_regression_model = Lasso()
lasso_regression_model.fit(Xt_train, yt_train)


# In[ ]:


# Evaluate model
evaluate_model(lasso_regression_model)


# In[ ]:


# Test different alphas
alphas = np.logspace(start=-5,stop=-1,base=10,num=30)
cv_lasso = [cross_validation_evaluation(Lasso(alpha = alpha)).mean() 
            for alpha in alphas]
optimised_alpha = alphas[cv_lasso.index(min(cv_lasso))]
print('Optimised alpha is: ' + str(optimised_alpha))

cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "alpha hypertuning")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.xscale('log')


# In[ ]:


# Instantiate the improved lasso regression model
lasso_regression_improved_model = Lasso(alpha=optimised_alpha)
lasso_regression_improved_model.fit(Xt_train, yt_train)


# In[ ]:


# Evaluate model
evaluate_model(lasso_regression_improved_model)


# In[ ]:


# Seeing how many features were removed by the Lasso model
coef = pd.Series(lasso_regression_improved_model.coef_, index = X_train.columns)
print("Lasso model picked " + str(sum(coef != 0)) + " variables and eliminated " +  str(sum(coef == 0)) + " variables")


# In[ ]:


# Seeing what were considered important features (positive and negative)
plt.figure(figsize=[20,8])
important_features = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
sns.barplot(y=important_features.index,x=important_features)


# Best linear regression model is lasso

# # Gradient boosting (xgboost)

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV


# In[ ]:


xg_reg = xgb.XGBRegressor()


# In[ ]:


xg_reg.fit(Xt_train, yt_train)


# In[ ]:


# Evaluate model
evaluate_model(xg_reg)


# ## Tune parameters

# In[ ]:


# # Commented out to improve performance

# # Set the parameters by cross-validation

# xgb1 = XGBRegressor()

# tuned_parameters = {'objective':['reg:linear'],
#               'learning_rate': [0.01,0.03,0.1],
#               'max_depth': [3,5],
#               'min_child_weight': [2,4,6],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.8,1],
#               'n_estimators': [500,1000],
#                    'gamma':[0]}

# xgb_grid = GridSearchCV(xgb1,
#                         tuned_parameters,
#                         cv = 5,
#                         n_jobs = 5,
#                         verbose=True)

# xgb_grid.fit(X_train,
#          y)
# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)
# p_dict = xgb_grid.best_params_


# In[ ]:


# THIS IS THE RESULT OF THE GRIDSEARCH COMMENTED OUT

p_dict = {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 2, 'n_estimators': 1000, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}


# In[ ]:


# Set the new parameters into the model
xg_reg_improved = xgb.XGBRegressor(objective = p_dict['objective'], colsample_bytree = p_dict['colsample_bytree'], 
                          learning_rate = p_dict['learning_rate'], max_depth = p_dict['max_depth'], min_child_weight = p_dict['min_child_weight'], 
                          n_estimators = p_dict['n_estimators'], silent=p_dict['silent'], subsample=p_dict['subsample'])


# In[ ]:


# Fit the model
xg_reg_improved.fit(Xt_train, yt_train)


# In[ ]:


# Show feature importances
import_feats_pf = pd.DataFrame({'Variable':Xt_train.columns,
              'Importance':xg_reg_improved.feature_importances_}).sort_values('Importance', ascending=False)
import_feats_pf.head(10)


# In[ ]:


# Evaluate model
evaluate_model(xg_reg_improved)


# # Stacking models test

# In[ ]:


# Returns an array of prediction results for a given model & feature set

def return_predictions_model(model,inputs):
    predictions = model.predict(inputs)
    return predictions


# In[ ]:


# Model predictions
lasso_regression_predict = return_predictions_model(lasso_regression_improved_model,Xt_test)
xg_reg_predict = return_predictions_model(xg_reg_improved,Xt_test)
ridge_regression_predict = return_predictions_model(ridge_regression_improved_model,Xt_test)


# In[ ]:


# Average them together
stacked_predictions = np.array([lasso_regression_predict,xg_reg_predict,ridge_regression_predict])
stacked_predictions_avg = np.average(stacked_predictions, axis=0)

rmse_test = np.sqrt(mean_squared_error(yt_test,stacked_predictions_avg))
print("RMSE on Test set (non-weighted) :", rmse_test)

# Weighted higher on the better performing models (lasso, xgboost, ridge)
stacked_predictions_weighted = (3*lasso_regression_predict + 2*xg_reg_predict + ridge_regression_predict)/6

rmse_test = np.sqrt(mean_squared_error(yt_test,stacked_predictions_weighted))
print("RMSE on Test set (weighted) :", rmse_test)


# In[ ]:


# Sensecheck against a) known observations b) predictions made off the training data
plt.figure(figsize=[20,8])
sns.distplot(y,hist=False,label= 'Known observations')
sns.distplot(lasso_regression_predict,hist=False, label='lasso_regression')
sns.distplot(xg_reg_predict,hist=False, label='xg_reg')
sns.distplot(ridge_regression_predict,hist=False, label='ridge_regression')
sns.distplot(stacked_predictions_avg,hist=False, label='stacked_predictions')
sns.distplot(stacked_predictions_avg,hist=False, label='stacked_predictions_weighted')
plt.legend()


# # Fit models on full test data

# In[ ]:


# Get as much information as possible
lasso_regression_improved_model.fit(X_train, y)
xg_reg_improved.fit(X_train, y)
ridge_regression_improved_model.fit(X_train, y)


# # Test on full data

# In[ ]:


# Model predictions
lasso_regression_predict = return_predictions_model(lasso_regression_improved_model,X_test)
xg_reg_predict = return_predictions_model(xg_reg_improved,X_test)
ridge_regression_predict = return_predictions_model(ridge_regression_improved_model,X_test)


# In[ ]:


# Average them together (weighted)
stacked_predictions_weighted = (3*lasso_regression_predict + 2*xg_reg_predict + ridge_regression_predict)/6


# ## Export results

# In[ ]:


# Put into dataframe
d = {'Id': test.index, 'SalePrice': stacked_predictions_weighted}
predictions_df = pd.DataFrame(data=d)
predictions_df.head(5)


# In[ ]:


# Convert 
predictions_df.SalePrice = predictions_df.SalePrice.apply(np.exp)
predictions_df.head(5)


# In[ ]:


# Sensecheck against a) known observations b) predictions made off the training data
plt.figure(figsize=[20,8])
sns.distplot(y_original,hist=False,label= 'Known observations')
sns.distplot(predictions_df.SalePrice,hist=False, label='Unseen data observations')
plt.legend()


# In[ ]:


# Export for testing
predictions_df.to_csv('output.csv', index=False)
predictions_df.head(5)

