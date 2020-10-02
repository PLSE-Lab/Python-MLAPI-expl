#!/usr/bin/env python
# coding: utf-8

# Since this is my first Kaggle submission, I wanted to keep it fairly straightforward, and follow some of the excellent solutions already out there.  In particular, the logic behind this largely follows the following three workbooks: 
# 
# 
# Since this is my first kaggle submission, I wanted to keep things fairly straightforward, and follow the approach of some of the excellent solutions already out there.  In particular, this notebook is largely based on the following three notebooks, but I've tried to build on these by adding some other elements, such as a collinearity matrix, and a standard multiple linear regression for comparison.  
# - [Apapiu](https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models)
# - [Neviadomski](https://www.kaggle.com/neviadomski/house-prices-advanced-regression-techniques/how-to-get-to-top-25-with-simple-model-sklearn)
# - [Juliencs](https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset). 
# 
# Also, there is some very nice exploratory data analysis and plots in this notebook: [xchmiao](https://www.kaggle.com/xchmiao/detailed-data-exploration-in-python).
# 
# Comments welcome!

# ## Import packages

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Stats
from scipy.stats.stats import skew
from scipy.stats.stats import pearsonr


# ## Import data

# In[ ]:


# Test and training set
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Combine into one dataset for the purposes of cleaning, and make sure that index continues
data_full = pd.concat([train, test], keys = ['train', 'test'])#ignore_index = True)


# In[ ]:


# What does the dataset look like?
train.head(3)


# ## Data cleansing

# ### Dealing with nulls

# In[ ]:


# Count the uniques for each column for a given dataframe
def df_uniques(df):
    print('Col name,', 'Number of nulls,', 'Number of unique values', '% of nulls')
    list_of_features = []
    for col in df:
        l = [col, df[col].shape[0] - df[col].count(), df[col].unique().shape[0], '%.3f' %((df[col].shape[0] - df[col].count()) / df[col].shape[0])]
        list_of_features.append(l)
    # Sort by the number of NULLs: 
    list_of_features = sorted(list_of_features, key = lambda x: x[1], reverse = True)
    return list_of_features

df_uniques(train)


# In[ ]:


# The following features have a crazy number of nulls 
# PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage 

# From looking at the data dictionary, these NAs are not necessarily a problem - but "NA" or nUll is misleading, so let's replace them
# Alley: NA = no alley -> replace with "None"
# MiscFeature: other features (e.g. tennis court) - NA = no other feature -> replace with "None"
# Fence: NA = no fence -> replace with "None"
# FireplaceQu: you guessed it -> replace with "None"


# In[ ]:


# Let's get a neat list of the null columns - need to combine both datasets for this
null_columns = [col for col in data_full.columns if data_full[col].isnull().any()]
print(null_columns)


# In[ ]:


# Define a function to replace nulls for many columns: 
def fill_nulls(df, col_list, na_val):
    for col in col_list:
        df[col].fillna(value = na_val, inplace = True)
    return df


# In[ ]:


# Categorical fields with an obvious meaning NA -> 'None'
nulls_to_none = ['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu', 'MasVnrType', 'BsmtCond', 
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 
                 'GarageQual', 'GarageCond', 'KitchenQual']
# Numerical fields with an obvious meaning NA -> 0
nulls_to_zero = ['LotFrontage', 'MasVnrArea', 'BsmtQual', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 
                 'BsmtHalfBath', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'TotalBsmtSF']

# Categorical fields with a less obvious interpretation - guessing that NA means 'None' (there are very few anyway)
nulls_to_zero_2 = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd']
nulls_to_other = ['SaleType', 'Functional']

# Apply to both test and training sets:
for df in [train, test]: 
    fill_nulls(df, nulls_to_none, 'None')
    fill_nulls(df, nulls_to_zero, 0)
    fill_nulls(df, nulls_to_zero_2, 0)
    fill_nulls(df, nulls_to_other, 'Other')
# NB we still have 'data_full' which has not been updated yet


# ### Data types
# Let's make sure everything is in the correct data type.  Pandas will have a go at importing things correctly, but this is good practice to make sure that things haven't gone awry. Ultimately we'll want to use dummy variables for categorical data anyway.

# In[ ]:


# Print out data types
def data_types(df):
    for col in df:
        print(col, type(df[col][1]))   


# In[ ]:


data_types(train)


# In[ ]:


# By pasting the above list into a spreadsheet and cross checking with the data dictionary, we can 
# see which category each field should be

# statsmodel requires all fieldsnames to begin with letters, so let's sort this out now.
train = train.rename(columns = {'1stFlrSF': 'FirstFlrSF','2ndFlrSF': 'SecondFlrSF','3SsnPorch': 'ThreeSsnPorch'})
test = test.rename(columns = {'1stFlrSF': 'FirstFlrSF','2ndFlrSF': 'SecondFlrSF','3SsnPorch': 'ThreeSsnPorch'})
data_full = pd.concat([train, test], keys = ['train', 'test'])

# Makes lists of each type
categories = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
              'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt', 
              'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 
              'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'Heating', 
              'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 
              'GarageFinish', 'GarageCars', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 
              'SaleCondition']
floats = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
          'FirstFlrSF', 'SecondFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
          'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

ints = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
         'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces']

bools = ['CentralAir']

feature_names = categories + floats + ints + bools

# Define a function for converting a list of columns to a particular type: 
def convert_col_type(df, cols, type):
    for col in cols:
        df[col] = df[col].astype(type)


# In[ ]:


# Convert each column for both test and training sets:
for df in [train, test]:
    convert_col_type(df, categories, 'category')
    convert_col_type(df, floats, 'float')
    convert_col_type(df, ints, 'int')
    convert_col_type(df, bools, 'bool')
    
# Re-define the full dataset
data_full = pd.concat([train, test], keys = ['train', 'test'])


# In[ ]:


# Check new data types  
data_types(train)


# ## Collinearity
# First, let's check to see which predictors are correlated; there are many features that essentially encode the same information in different way 

# In[ ]:


# Compute the correlation matrix
corr = data_full.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, #cmap=cmap, vmax=.3,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

# Simpler version (but too small to be useful)
#plt.matshow(data_full.corr())


# In[ ]:


# Which predictors are mostly closely correlated with SalePrice?
corr['SalePrice'].sort_values(ascending = False)


# The most highly correlated predictors relate to size: OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, FirstFlrSF. This intuitively makes sense.

# ## Motivating plots

# In[ ]:


# What's the distribution of prices?
sales_price = train['SalePrice']
graph = sns.distplot(sales_price)


# In[ ]:


# Let's log-tranform this: 
sales_prices_log = np.log1p(sales_price)
graph = sns.distplot(sales_prices_log)


# In[ ]:


# This looks much better, so let's replace the SalePrice with the log-transformed version (will need to exponentiate predictions)
train['SalePrice'] = np.log1p(train['SalePrice'])
# Re-define the full dataset - and work on this until we are ready to split out test and train sets again
data_full = pd.concat([train, test], keys = ['train', 'test'])


# In[ ]:


# Let's look at the plots of the important features identified above with SalePrice
fig, axs = plt.subplots(ncols=3, nrows=4, figsize = (20,10))
sns.regplot(x='OverallQual', y='SalePrice', data=data_full, ax=axs[0,0])
sns.regplot(x='GrLivArea', y='SalePrice', data=data_full, ax=axs[0,1])
sns.regplot(x='GarageCars',y='SalePrice', data=data_full, ax=axs[0,2])
sns.regplot(x='GarageArea',y='SalePrice', data=data_full, ax=axs[1,0])
sns.regplot(x='TotalBsmtSF',y='SalePrice', data=data_full, ax=axs[1,1])
sns.regplot(x='FirstFlrSF',y='SalePrice', data=data_full, ax=axs[1,2])
sns.regplot(x='FullBath',y='SalePrice', data=data_full, ax=axs[2,0])
sns.regplot(x='TotRmsAbvGrd',y='SalePrice', data=data_full, ax=axs[2,1])
sns.regplot(x='YearBuilt',y='SalePrice', data=data_full, ax=axs[2,2])
sns.regplot(x='MasVnrArea',y='SalePrice', data=data_full, ax=axs[3,0])
sns.regplot(x='Fireplaces',y='SalePrice', data=data_full, ax=axs[3,1])
sns.regplot(x='BsmtFinSF1',y='SalePrice', data=data_full, ax=axs[3,2])
fig.tight_layout()


# Many of these are also skewed to the left, so let's log-transform any variables with a skewness greater than 1

# In[ ]:


skewed_features = data_full[floats].apply(lambda x: skew(x.dropna()))
skewed_features = skewed_features[skewed_features > 1]
skewed_features.sort_values(ascending = False)


# In[ ]:


skewed_features = skewed_features.index


# In[ ]:


# Now let's log-transform the skewed features
for col in skewed_features:
   data_full[col] = np.log1p(data_full[col])


# ## Standardising numeric features

# In[ ]:


# Standardise numeric features (normalise)
numeric_features = data_full.loc[:,floats]
numeric_features_st = (numeric_features - numeric_features.mean())/numeric_features.std()


# In[ ]:


data_full.loc[:,floats] = numeric_features_st


# ### Split test-train sets again

# In[ ]:


# split out the test and train sets again
train = data_full.ix['train']
test = data_full.ix['test']


# ## Linear regression

# In[ ]:


# For the purposes of a multiple regression, let's use statsmodel rather than scikit learn, as it gives us
# more information, such as p-values, and hence, which regressors are important.
import statsmodels.formula.api as smf

# create a fitted model with the features that are floats: 
#lm = smf.ols(formula='SalePrice ~ LotFrontage + LotArea + MasVnrArea + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + TotalBsmtSF + FirstFlrSF + SecondFlrSF + LowQualFinSF + GrLivArea + GarageArea + WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSsnPorch + ScreenPorch + PoolArea + MiscVal + OverallQual + OverallCond + BsmtFullBath + BsmtHalfBath + FullBath + BedroomAbvGr + KitchenAbvGr + TotRmsAbvGrd + Fireplaces', data=train).fit()
formula = 'SalePrice ~ ' + ' + '.join(feature_names)
lm = smf.ols(formula=formula, data=train).fit()

# print the coefficients
lm.summary()


# In[ ]:


# Best features
lm.pvalues.sort_values(ascending = False, inplace=False).tail(10)


# In[ ]:


# Worst features
lm.pvalues.sort_values(ascending = False, inplace=False).head(10)


# It looks like the most important predictors relate to the type of roofing material (perhaps correlated with certain neighbourhoods, or architecture styles, that are in turn correlated with demographic factors), and features related to the size of the house (LotArea, GrLivArea, OverallQual), and the zone of the area (MSZoning).  This intuitively makes sense.  
# 
# On the other hand, features that are fairly useless relate to remodelling of the house, and exterior surface features.
# 
# Since there are so many features, it would make sense to either remove these by one of the following: 
# - backward elimination
# - principal component analysis
# - regularisation to penalise the extra features.  This avoids over-fitting.  In particular, lasso regularisation performs some feature selection for us.
# 
# In this notebook, I'll take the latter approach.

# ## Splitting the testing and training sets again; define dummy variables for categories

# In[ ]:


# Features - remove the thing we're trying to predict!
features = data_full.drop('SalePrice', axis = 1)

# Create dummy variables - for each categorical data, make several boolean flags
features = pd.get_dummies(features)

# Make matrices to pass to scikit learn:
X_train = features[:train.shape[0]]
X_test = features[train.shape[0]:]
y = train['SalePrice']

# Verify that the number of features has been increased due to the dummy variables:
print('Number of features in original dataset, including categorical fields: ', train.shape[1], 
      '\nNumber of features, including dummy variables for categorical fields: ', X_train.shape[1])


# ## Ridge regularisation (L2 regularisation)

# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_squared_error


# In[ ]:


# Define root-mean-square-error function - use 10-fold cross-validation
# You have to use neg_mean_squared_error because mean_squared_error will be deprecated in future
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse

# Invoke Ridge regularisation
model_ridge = Ridge()


# In[ ]:


# Tune parameters - the only parameter is alpha - the larger alpha, the larger the penalty for extra predictors
alphas = [0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
# Work out the RMSE for each value of the alphas above: 
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)


# In[ ]:


# Let's plot the RMSE as a function of alpha
matplotlib.rcParams['figure.figsize'] = (7,3)
cv_ridge.plot(title = 'RMSE as a function of alpha (Ridge regularisation)')
plt.xlabel('alpha')
plt.ylabel('RMSE')


# We want to chose the value of $\alpha$ that minimises the chart above. The extreme cases are $\alpha = 0$, which corresponds to no penalty for each extra predictor, and $\alpha\to\inf$ which corresponds to a null model.  We want a balance between flexibility and over-fitting, which represents the minimium of this chart.

# In[ ]:


cv_ridge.min()


# In[ ]:


# This looks like it correpsonds to alpha = 30, so let's fit the model with that.
model_ridge = Ridge(alpha = 30)
model_ridge.fit(X_train, y)


# In[ ]:


# What are the important coefficients here?
coef_ridge = pd.Series(model_ridge.coef_, index = X_train.columns)
important_coef_ridge = pd.concat([coef_ridge.sort_values().head(10), coef_ridge.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
important_coef_ridge.plot(kind = "barh")
plt.title('Important coefficients in the Lasso Model')


# In[ ]:


# How many features were eliminated? 
print("Ridge picked " + str(sum(coef_ridge != 0)) + " features and eliminated the other " + str(sum(coef_ridge == 0)) + " features")


# In[ ]:


# Let's see what the correlation matrix looks like now: 
c = coef_ridge[coef_ridge != 0]
corr = features[c.index].corr()
plt.matshow(corr)


# This is much bigger (more features due to dummy variables) but it looks like lasso has eliminated a lot of the correlated variables that we saw above in the correlation matrix. 

# In[ ]:


#let's look at the residuals as well:
def plot_residuals(model, X_train, y):
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    preds = pd.DataFrame({"preds":model.predict(X_train), "true":y})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")


# In[ ]:


plot_residuals(model_ridge, X_train, y)


# These look pretty good - nicely clustered around 0.

# ## Lasso regularisation (L1 regularisation)
# The advantage of Lasso regularisation is that it performs some feature selection.  We'll use Lasso cross-validation to choose the $\alpha$ for us.

# In[ ]:


model_lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1],
                     max_iter = 50000, cv = 10).fit(X_train, y)
# Coefficients of each predictor:
coef_lasso = pd.Series(model_lasso.coef_, index = X_train.columns)


# In[ ]:


# What are the important coefficients here?
important_coef_lasso = pd.concat([coef_lasso.sort_values().head(10), coef_lasso.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
important_coef_lasso.plot(kind = "barh")
plt.title('Important coefficients in the Lasso Model')


# A lot of these are on the list of features picked out by the normal multiple regression without regularisation. Not the hugely negative coefficient for Clay Tile - there is only one house with this feature, so this is not actually a particuarly important feature.

# In[ ]:


data_full['RoofMatl'].value_counts()


# In[ ]:


data_full[data_full['RoofMatl'] == 'ClyTile']['SalePrice']


# In[ ]:


# How many features were eliminated? 
print("Lasso picked " + str(sum(coef_lasso != 0)) + " features and eliminated the other " + str(sum(coef_lasso == 0)) + " features")


# In[ ]:


# Let's have a look at the residuals of this too.    
plot_residuals(model_lasso, X_train, y)


# ## Elastic net regularisation

# In[ ]:


from sklearn.linear_model import ElasticNetCV


# In[ ]:


model_elastic = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
model_elastic.fit(X_train, y)
alpha = model_elastic.alpha_
ratio = model_elastic.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )


# In[ ]:


coef_elastic = pd.Series(model_elastic.coef_, index = X_train.columns)


# In[ ]:


# How many features were eliminated? 
print("Elastic picked " + str(sum(coef_elastic != 0)) + " features and eliminated the other " + str(sum(coef_elastic == 0)) + " features")


# In[ ]:


plot_residuals(model_elastic, X_train, y)


# These look almost identical to the residuals for the Lasso, and the number of features is larger, so it's probably more prone to over-fitting.  I'm sticking with the Lasso regularisation.

# # Make predictions

# In[ ]:


ridge_preds = np.expm1(model_ridge.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
elastic_preds = np.expm1(model_elastic.predict(X_test))


# In[ ]:


preds = {'ridge': ridge_preds, 'lasso': lasso_preds, 'elastic': elastic_preds}


# In[ ]:


from datetime import datetime


# In[ ]:


def make_export_table(model):
    kaggle_export = pd.DataFrame({
        'id': test['Id'],
        'SalePrice': preds[model]
    },
    columns = ['id', 'SalePrice'])
    return kaggle_export


# In[ ]:


for model in ['ridge', 'lasso', 'elastic']:
    filebasename = 'kaggle_export'
    timestamp = datetime.today().strftime('%Y%m%d-%H%M%S')
    filename = filebasename + timestamp + model
    table = make_export_table(model)
    table.to_csv(filename, index = False)


# These submissions give the following scores on the public leaderboard. This puts me in the top 31% of submissions as of 3 May 2017.
# - Elastic - 0.12414
# - Lasso - 0.12377
# - Ridge - 0.12431
# As others have noted in their notebooks, these scores shows that you can do reasonably well with pretty straightforward models. 
# 
# Comments welcome!
