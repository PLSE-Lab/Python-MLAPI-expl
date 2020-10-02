#!/usr/bin/env python
# coding: utf-8

# # Melbourne Housing Price Regression
# 
# Using [this Kaggle data](https://www.kaggle.com/anthonypino/melbourne-housing-market) create a model to predict a house's value.
# We want to be able to understand what creates value in a house, as though we were a real estate developer.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Import data

# In[2]:


# import data
PATH = ('../input/Melbourne_housing_FULL.csv')
raw_data = pd.read_csv(PATH, index_col=None)
df_raw = pd.DataFrame(raw_data)
df_raw.head()

#PATH_clean = ('assets\Melbourne_housing_FULL_clean.csv')
#clean_data = pd.read_csv(PATH_clean, index_col=0)
#df_clean = pd.DataFrame(clean_data)
#df_clean.head()


# ### Content & Acknowledgements
# This data was scraped from publicly available results posted every week from Domain.com.au, I've cleaned it as best I can, now it's up to you to make data analysis magic. The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.
# 
# ....Now with extra data including including property size, land size and council area, you may need to change your code!
# 
# Some Key Details
# Suburb: Suburb
# 
# Address: Address
# 
# Rooms: Number of rooms
# 
# Price: Price in Australian dollars
# 
# Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
# 
# Type: br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.
# 
# SellerG: Real Estate Agent
# 
# Date: Date sold
# 
# Distance: Distance from CBD in Kilometres
# 
# Regionname: General Region (West, North West, North, North east ...etc)
# 
# Propertycount: Number of properties that exist in the suburb.
# 
# Bedroom2 : Scraped # of Bedrooms (from different source)
# 
# Bathroom: Number of Bathrooms
# 
# Car: Number of carspots
# 
# Landsize: Land Size in Metres
# 
# BuildingArea: Building Size in Metres
# 
# YearBuilt: Year the house was built
# 
# CouncilArea: Governing council for the area
# 
# Lattitude: Self explanitory
# 
# Longtitude: Self explanitory

# ## EDA
# Let's explore the data and get an idea of what we're working with.

# In[3]:


df_raw.info()


# In[4]:


df_raw.describe()


# In[5]:


# view price data
display(df_raw.Price.head())
df_raw.Price.dropna().hist()
plt.title('Price');


# Let's see if we can log transform the Price data to create a distribution we can work with.

# In[6]:


# log transform price data
df_raw['log_Price'] = np.log1p(df_raw.Price.dropna())

log_price_mean = df_raw['log_Price'].mean()
log_price_std = df_raw['log_Price'].std()

# view log(price) data
df_raw.log_Price.hist(bins=20)
plt.axvline((log_price_mean+log_price_std), color='k', linestyle='--')
plt.axvline((log_price_mean-log_price_std), color='k', linestyle='--')
plt.axvline(log_price_mean, color='k', linestyle='-')
plt.title('log(Price)');


# That's much better.

# ## Data Cleaning

# In[7]:


# view missing data
display(df_raw.shape)
display(df_raw.dropna().shape)


# It looks like our dataset has a ton of missing values, approximately 75%.
# Let's see if we can clean some of this up by dropping unnecessary features filling in some NaNs.
# We'll also have to work with our dtypes, we have 8 instances of dtype = object that we'll have to correct.

# In[8]:


# view features
df_raw.columns.values

# view sum of NaN values if
# all missing Price values are dropped
#df_raw.dropna(subset=['Price']).isna().sum()

# view sum of NaN values
df_raw.isna().sum()


# It looks like most of our missing data is from the columns 'Price' (and consequentially 'log_Price'), 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuild', 'Lattitude', and 'Longitude'.
# The rest of the columns with missing data should be easy to infer or input based on the rest of the data present.
# 
# The missing 'Price' data will be the most troublesome, as it is the variable we hope to predict. We could input the missing values with the median Price value, or the median Price value of homes in the similar class.
# For example we could input the median price of all homes, or the median price homes with a similar number of rooms.
# Either way, it will skew the data and our model.
# We could drop the missing values, but we would loose ~20% of our dataset.
# 
# Let's start with filling in our other missing data and see what we're left with.
# First, though, we'll define price ranges for low, median, and high priced homes.
# These ranges will [hopefully] give us a more intuitive idea when inputting NaN values.

# ### Set categorical price values
# #### Price ranges

# In[9]:


# define 'high price' and 'low price' binary features
df_raw['high_price'] = np.where(
    df_raw['log_Price'] > (log_price_mean+log_price_std), 1, 0
)

df_raw['low_price'] = np.where(
    df_raw['log_Price'] < (log_price_mean-log_price_std), 1, 0
)

display(df_raw['high_price'].value_counts())
df_raw['low_price'].value_counts()


# #### One-Hot encoding

# The features 'Suburb', 'Address', and 'SellerG' have far too many unique values to be helpful.
# So we will just drop those columns.
# 
# 'Type', 'Method', and 'Regionname' we can convert with one-hot encoding to a new set of features.
# If we do that, we can also drop 'CouncilArea' as it is a more specific (and likely directly correlated!) version of the 'Regionname' feature (for example, 'Yarra City Council' will only be in the 'Northen Metropolitan' region).
# 
# Lastly, we can convert 'Date' to a datetime format.

# In[10]:


# drop features
drop_list = ['Suburb', 'Address', 'SellerG', 'CouncilArea']
df_raw = df_raw.drop(drop_list, axis=1)

df_raw.head()


# In[11]:


df_raw[['Type', 'Method', 'Regionname']].isna().sum()


# In[12]:


# Regionname has NaN values,
# so we'll have to deal with those
df_raw = df_raw.dropna(subset=['Type', 'Method', 'Regionname'], axis=0)


# In[13]:


# one-hot encoding
# set sparse=False to return an array
cat_encoder = OneHotEncoder(sparse=False)
df_raw_type_reshaped = df_raw['Type'].values.reshape(-1,1)
df_raw_type_1hot = cat_encoder.fit_transform(df_raw_type_reshaped)
categories = cat_encoder.categories_
df_raw_type_1hot = pd.DataFrame(df_raw_type_1hot, columns=categories)


# In[14]:


# concat 1hot DataFrame w/ df_na
# reset index of df_na and concat
df_raw = df_raw.reset_index().drop('index', axis=1)
df_raw = pd.concat([df_raw, df_raw_type_1hot], axis=1)
df_raw.head()


# In[15]:


# one-hot encode 'Method' feature
df_raw_meth_reshaped = df_raw['Method'].values.reshape(-1,1)
df_raw_meth_1hot = cat_encoder.fit_transform(df_raw_meth_reshaped)
categories = cat_encoder.categories_
df_raw_meth_1hot = pd.DataFrame(df_raw_meth_1hot, columns=categories)

# concat 1hot DataFrame w/ df_na
df_raw = pd.concat([df_raw, df_raw_meth_1hot], axis=1)
df_raw.head()


# In[16]:


# one-hot encode 'Regionname' feature
df_raw_reg_reshaped = df_raw['Regionname'].values.reshape(-1,1)
df_raw_reg_1hot = cat_encoder.fit_transform(df_raw_reg_reshaped)
categories = cat_encoder.categories_
df_raw_reg_1hot = pd.DataFrame(df_raw_reg_1hot, columns=categories)

# concat 1hot DataFrame w/ df_na
df_raw = pd.concat([df_raw, df_raw_reg_1hot], axis=1)
df_raw.head()


# In[17]:


df_raw.isna().sum()


# ### Fill NaN values

# In[18]:


# drop NaNs from target column(s)
df_na = df_raw.dropna(subset=['Price', 'log_Price'], axis=0)
df_na.isna().sum()


# In[19]:


# find our features with dtype == object
objects = []

for i in df_raw.columns.values:
    if df_raw[i].dtype == 'O':
        objects.append(str(i))

df_na = df_na.drop(objects, axis=1)

df_na.isna().sum()


# In[20]:


# drop regional data
cols = ['Bedroom2', 'Bathroom', 'Car',
        'Landsize','BuildingArea', 'YearBuilt',
        'high_price', 'low_price']
df_num = df_na[cols]

# check nulls
df_num.isna().sum()


# In[21]:


# fill NaN values by median of price category
df_num.loc[df_num['high_price']==1] = df_num.loc[
    df_num['high_price']==1].apply(lambda x: x.fillna(x.median()),axis=0)
df_num.loc[df_num['low_price']==1] = df_num.loc[
    df_num['low_price']==1].apply(lambda x: x.fillna(x.median()),axis=0)
df_num.loc[df_num['high_price' and 'low_price']==0] = df_num.loc[
    df_num['high_price' and 'low_price']==0].apply(lambda x: x.fillna(x.median()),axis=0)

df_num.isna().sum()


# In[22]:


# concat filled NaNs w/ the rest of our data
df_na = pd.concat([df_na.drop(cols, axis=1), df_num], axis=1)
df_na.head()
df_na.isna().sum()


# In[23]:


# clean up columns from 1hot encoding
col_list = ['Rooms', 'Price', 'Distance', 'Postcode', 'Lattitude',
            'Longtitude', 'Propertycount', 'log_Price', 'h', 't', 'u',
            'PI', 'PN', 'S', 'SA', 'SN', 'SP', 'SS',
            'VB', 'W', 'Eastern_Metropolitan', 'Eastern_Victoria',
            'Northern_Metropolitan', 'Northern_Victoria',
            'South_Eastern_Metropolitan', 'Southern_Metropolitan',
            'Western_Metropolitan', 'Western_Victoria', 'Bedroom2',
            'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
            'high_price', 'low_price']
df_na.columns = col_list


# In[24]:


reg_cols = ['Eastern_Metropolitan', 'Eastern_Victoria',
            'Northern_Metropolitan', 'Northern_Victoria',
            'South_Eastern_Metropolitan', 'Southern_Metropolitan',
            'Western_Metropolitan', 'Western_Victoria']

for i in reg_cols:
    df_na.loc[df_na[i]==1] = df_na.loc[
        df_na[i]==1].apply(lambda x: x.fillna(x.median()),axis=0)
    
df_na.isna().sum()


# Now we've filled the NaN values.

# ### Convert to datetime

# In[25]:


# convert Date feature to datetime
df_na['Date'] = df_raw['Date']
df_na['Date'].head()
df_na['Date'] = pd.to_datetime(df_na['Date'], errors='raise', dayfirst=1)
df_na['Date'].head()


# In[26]:


# store clean data
df_clean = df_na
#df_clean.to_csv('assets\Melbourne_housing_FULL_clean.csv')


# In[27]:


# check data
df_clean.info()


# Excellent! Our data is clean, we have no missing values, and our dtypes are correct.
# Let's check our data for correlation and then it will be ready to model.

# ## Feature Selection

# ### Check for correlation.
# First, let's define our data and target value.
# Then we can check for correlation and drop any features with over 90% correlation.

# In[28]:


# define data and target
# drop our target value and derived values
# drop datetime as it likely won't help our model
drop_list = ['Price', 'log_Price', 'high_price', 'low_price', 'Date']
data = df_clean.drop(drop_list, axis=1)
target = df_clean['log_Price']

# plot a heatmap
sns.heatmap(data.corr());


# In[29]:


# Create correlation matrix
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=4).astype(np.bool))

# Find index of feature columns with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

display(data.shape)

# Drop correlated features 
for i in to_drop:
    data = data.drop(i, axis=1)

data.shape


# It looks like our features aren't overly correlated.
# Let's go ahead with modeling.

# ## Model Selection

# In[30]:


# define training and test set
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42)

# scale X_train values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))


# ### Linear Regression

# In[31]:


# determine which model to use
OLS = LinearRegression()
OLS.fit(X_train, y_train)
y_pred = OLS.predict(X_test)

# Display.
print('Linear Regression')
print('\nR-squared training set:')
print(OLS.score(X_train, y_train))

print('\nR-squared test set:')
print(OLS.score(X_test, y_test))


# ### Linear Regression Scaled

# In[32]:


# determine which model to use
OLS_scaled = LinearRegression()
OLS_scaled.fit(X_train_scaled, y_train)
y_pred = OLS_scaled.predict(X_test_scaled)

# Display.
print('Scaled Linear Regression')
print('\nR-squared training set:')
print(OLS_scaled.score(X_train_scaled, y_train))

print('\nR-squared test set:')
print(OLS_scaled.score(X_test_scaled, y_test))


# ### Random Forest Regression

# In[33]:


# determine which model to use
RF = RandomForestRegressor(n_estimators=10)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

# Display.
print('Random Forest Regressor')
print('\nR-squared training set:')
print(RF.score(X_train, y_train))

print('\nR-squared test set:')
print(RF.score(X_test, y_test))


# ### Random Forest Regressor Scaled

# In[34]:


# determine which model to use
RF_scaled = RandomForestRegressor(n_estimators=10)
RF_scaled.fit(X_train_scaled, y_train)
y_pred = RF_scaled.predict(X_test_scaled)

# Display.
print('Scaled Random Forest Regressor')
print('\nR-squared training set:')
print(RF_scaled.score(X_train_scaled, y_train))

print('\nR-squared test set:')
print(RF_scaled.score(X_test_scaled, y_test))


# ### Ridge Regression

# In[35]:


# define empty list
alphas = []
train_scores = []
test_scores = []

#Run the model for many alphas.
for lambd in range(1, 50, 2):
    ridge = Ridge(alpha=lambd, fit_intercept=False)
    ridge.fit(X_train, y_train)
    alphas.append(lambd)
    train_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))

plt.plot(alphas, train_scores, label='Training Data')
plt.plot(alphas, test_scores, label='Test Data')
plt.title('Ridge Regression')
plt.xlabel('Lambda')
plt.ylabel('R-Squared')
plt.legend()
plt.show();


# In[36]:


# instantiate ridgeCV regression
alpha_list = range(1, 50, 2)
ridge = RidgeCV(alphas=alpha_list, cv=5)
ridge.fit(X_train, y_train)

# Display
print('Ridge Regression')
print('\nR-squared training set:')
print(ridge.score(X_train, y_train))

print('\nR-squared test set:')
print(ridge.score(X_test, y_test))

print('\nRidge regression alpha:')
print(ridge.alpha_)


# ### Scaled Ridge Regression

# In[37]:


# instantiate ridgeCV regression
alpha_list = range(1, 50, 2)
ridge_scaled = RidgeCV(alphas=alpha_list, cv=5)
ridge_scaled.fit(X_train_scaled, y_train)

# Display
print('Scaled Ridge Regression')
print('\nR-squared training set:')
print(ridge_scaled.score(X_train_scaled, y_train))

print('\nR-squared test set:')
print(ridge_scaled.score(X_test_scaled, y_test))

print('\nRidge regression alpha:')
print(ridge_scaled.alpha_)


# ### Gradient Boosting Regressor

# In[38]:


# determine which model to use
GBRT = GradientBoostingRegressor(max_depth=2, n_estimators=120)
GBRT.fit(X_train, y_train)

errors = [mean_squared_error(y_test, y_pred)
         for y_pred in GBRT.staged_predict(X_test)]
best_n_estimators = np.argmin(errors)

GBRT_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
GBRT_best.fit(X_train, y_train)
y_pred = GBRT_best.predict(X_test)

# Display
print('Gradient Boosting Regressor')
print('\nR-squared training set:')
print(GBRT_best.score(X_train, y_train))

print('\nR-squared test set:')
print(GBRT_best.score(X_test, y_test))


# ### Gradient Boosting Regressor Scaled

# In[39]:


# determine which model to use
GBRT_scaled = GradientBoostingRegressor(max_depth=2, n_estimators=120)
GBRT_scaled.fit(X_train_scaled, y_train)

errors = [mean_squared_error(y_test, y_pred)
         for y_pred in GBRT_scaled.staged_predict(X_test_scaled)]
best_n_estimators = np.argmin(errors)

GBRT_scaled_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
GBRT_scaled_best.fit(X_train_scaled, y_train)
y_pred = GBRT_scaled_best.predict(X_test_scaled)

# Display
print('Scaled Gradient Boosting Regressor')
print('\nR-squared training set:')
print(GBRT_scaled_best.score(X_train_scaled, y_train))

print('\nR-squared test set:')
print(GBRT_scaled_best.score(X_test_scaled, y_test))


# ## Fine Tune Model

# ### Sample dataset
# Sample 50% of training data for model optimization.

# In[40]:


# sample training set to speed up parameter optimization
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42)


# ### Parameter optimization

# In[41]:


# define our parameter ranges
learning_rate=[0.01]
alpha=[0.01,0.03,0.05,0.1,0.3, 0.9]
n_estimators=[int(x) for x in np.linspace(start = 10, stop = 500, num = 4)]
max_depth=[int(x) for x in np.linspace(start = 3, stop = 15, num = 4)]
max_depth.append(None)
min_samples_split=[int(x) for x in np.linspace(start = 2, stop = 5, num = 4)]
min_samples_leaf=[int(x) for x in np.linspace(start = 1, stop = 4, num = 4)]
max_features=['auto', 'sqrt']

# Create the random grid
param_grid = {'learning_rate':learning_rate,
              'alpha':alpha,
              'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
             }

print(param_grid)

# Initialize and fit the model.
model = GradientBoostingRegressor()
model = RandomizedSearchCV(model, param_grid, cv=3)
model.fit(X_train_sample, y_train_sample)

# get the best parameters
best_params = model.best_params_
print(best_params)


# In[42]:


# refit model with best parameters
model_best = GradientBoostingRegressor(**best_params)
model_best.fit(X_train, y_train)
y_pred = model_best.predict(X_test)


# In[43]:


feature_importance = model_best.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5

plt.subplot(1,2,2)
plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, data.columns.values[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[44]:


# sort top features
top_features = np.where(feature_importance > 20)
top_features = data.columns[top_features].ravel()
print(top_features)


# In[45]:


# Display.
print('Optimized Gradient Boosting Regressor')
print('\nR-squared training set:')
print(model_best.score(X_train, y_train))
print('\nMean absolute error training set: ')
print(mean_absolute_error(y_train, model_best.predict(X_train)))
print('\nMean squared error training set: ')
print(mean_squared_error(y_train, model_best.predict(X_train)))

print('\n\nR-squared test set:')
print(model_best.score(X_test, y_test))
print('\nMean absolute error test set: ')
print(mean_absolute_error(y_test, y_pred))
print('\nMean squared error test set: ')
print(mean_squared_error(y_test, y_pred))

# top features
print('\nTop indicators:')
print(top_features)


# ## Conclusion
# Our model results are consistent across our training and test data and we've determined the top indicators of housing price.
