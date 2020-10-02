#!/usr/bin/env python
# coding: utf-8

# <h1>Importing data and detecting outliers</h1>
# 
# First, import the necessary models and functions.

# In[ ]:


import warnings  
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from xgboost import XGBRegressor


# Import the training and testing data.

# In[ ]:


X_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col="Id")
X_train.dropna(axis=0, subset=['SalePrice'], inplace=True)


# Plot living area vs. sale price to spot outliers in either dimension. Extremes in home size or final sale price will distort patterns in the dataset.

# In[ ]:


plt.figure(figsize=(16,6))
sns.scatterplot(x=X_train['GrLivArea'], y=X_train['SalePrice'])


# Similar to above, but now we plot lot area vs. sale price.

# In[ ]:


plt.figure(figsize=(16,6))
sns.scatterplot(x=X_train['LotArea'], y=X_train['SalePrice'])


# Remove the outliers and start digging into missing values that require imputation.

# In[ ]:


X_train = X_train[(X_train['SalePrice'] > 50000) & (X_train['SalePrice'] < 750000) & (X_train['LotArea'] < 110000) & (X_train['GrLivArea'] < 4000)]
y = X_train['SalePrice']
X_train.drop(['SalePrice'], axis=1, inplace=True)
train_size = len(X_train)

def investigate_missing(df):
    for col in df:
        missing = df[col].isnull().sum()
        if missing > 0:
            print("{}: {} missing --- type: {}".format(col, missing, df[col].dtype))
            
investigate_missing(pd.concat([X_train, X_test]))


# <h1>Feature processing</h1>
# 
# Now comes the time to cleanup a lot of the features and possibly engineer new ones.
# 
# We start by converting some numeric features to strings. Months of the year being encoded as integers doesnt make much sense.
# 
# We then look into ```LotFrontage```, which has *483* missing values. ```Neighborhood``` is present in every observation, so for each house with missing ```LotFrontage```, we can lookup which ```Neighborhood``` it belongs to and impute the median ```LotFrontage``` for that ```Neighborhood```.
# 
# ```MiscFeature``` is missing in the majority of observations, but after some investigation turns out to contain information on sheds - namely if the the house has one. We make a new feature ```Shed?``` and drop ```MiscFeature```.
# 
# ```MasVnrType``` contains redundant values **None** and **NA**, so we get rid of the **None**.
# 
# ```GarageYrBlt``` is difficult to impute. There are a lot of missing values, which likely stems from the home not having a garage. So what do we impute? 0? The mean? The build year of the home? These don't make much sense. In the end, we drop ```GarageYrBlt```. The main information encoded here is whether or not the home has a garage, and this information is captured in the other garage predictors.
# 
# ```Condition2``` and ```Functional``` have very little relation to sale price, so we drop them, too.
# 
# In terms of feature engineering, only one came to mind: quality square footage. Raw square footage is a good predictor, but it has some disadvantages. A big house can be of poor quality (think an old rundown mansion), the same way a smaller house can be of very fine quality. Combining ```OverallQual``` and ```GrLivArea``` into one feature, ```QualitySF```, is useful to us.
# 
# The rest of the imputation is done on a by-case basis, explained in comments in the code below.
# 
# For any feature describing area, we take the square root to get a better linear predictor.
# 
# Next, we apply a box-cox transformation to all skewed features.
# 
# We do one final imputation pass for all features in case we missed any, filling 0 in numerical features and **NA** in categorical ones.
# 
# Finally, we use ```RobustScaler``` on numerical features and we one-hot encode the categoricals.
# 
# The resulting ```DataFrame``` is cleaned and processed and ready for modelling.

# In[ ]:


def featureProcessing(df):
    
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)
    
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    df['Shed?'] = np.where(df['MiscFeature']=='Shed', 1, 0)
    df.drop('MiscFeature', axis=1, inplace=True)
    
    df['MasVnrType'].replace(['None'], 'NA')
    df.drop(['GarageYrBlt', 'Condition2', 'Functional'],  axis=1, inplace=True)
    
    df['QualitySF'] = df['GrLivArea'] * df['OverallQual']
    
    # Numerical columns for which we want to impute missing values with 0
    fill_num_col=[
        'MasVnrArea'
    ]
    
    # Categorical columns for which we want to impute missing values with 'NA'
    fill_cat_col=[
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'FireplaceQu',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'PoolQC',
        'Fence'  
    ]
    
    # Categorical columns for which we want to impute missing values with their mode
    mode_cat_col=[
        'Alley',
        'MasVnrType',
        'Electrical'
    ]
    
    for feat in fill_num_col:
        df[feat].fillna(0, inplace=True)
    for feat in fill_cat_col:
        df[feat].fillna('NA', inplace=True)
    for feat in mode_cat_col:
        df[feat].fillna(df[feat].mode(), inplace=True)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    area_features = [feat for feat in X_train.columns if "SF" in feat or "Area" in feat]
    
    numericals = df.select_dtypes(include=numerics).columns.values
    categoricals = df.select_dtypes(include=['object']).columns.values
    
    for feat in area_features:
        df[feat] = df[feat].map(lambda x: np.sqrt(x))
    
    skewness = df.select_dtypes(include=numerics).apply(lambda x: skew(x))
    skew_index = skewness[abs(skewness) >= 0.5].index
    for col in skew_index:
        df[col] = boxcox1p(df[col], 0.15) 
    for feat in numericals:
        df[feat].fillna(0, inplace=True)
        df[feat] = RobustScaler().fit_transform(df[feat].apply(float).values.reshape(-1,1))
    for feat in categoricals:
        df[feat].fillna('NA', inplace=True)
        dummies = pd.get_dummies(df[feat])
        dummies.columns = [col + feat for col in dummies.columns.values]
        df.drop(feat, axis=1, inplace=True)
        df = df.join(dummies)

    return df


X_full = featureProcessing(pd.concat([X_train, X_test]))

X_train = X_full[:train_size]
X_test = X_full[train_size:]


# <h1>Modelling</h1>
# 
# We are now ready to start modelling our data. We will use ```ElasticNet```, ```Ridge```, ```Lasso```, and ```XGBoost``` in our ensemble. But first, we need to tune the hyperparameters for these models.
# 
# For ```ElasticNet``` we need to tune ```alpha``` and ```l1_ratio```.
# 
# For ```Ridge``` and ```Lasso```, we need to tune ```alpha```.
# 
# For ```XGBoost```, we need to tune ```n_estimators``` and ```learning_rate```.
# 
# For each model, we do this using a grid search with k-fold cross validation.
# 
# **Important**: Since the scoring metric used for this competition is root mean squared *log* error, we use ```np.log1p(y)``` as our target, not just ```y```. We just have to remember to apply the reverse transformation ```np.expm1()``` when we submit our predictions.

# In[ ]:


#Tunes the respective hyperparameters

'''

def RMSLE_cross_val(model):
    return (-cross_val_score(model, X_train, np.log1p(y), scoring="neg_mean_squared_error", cv=5)).mean()

#Hyperparameter vector space for ElasticNet
alphas_en = np.arange(0.0005, 0.01, 0.0005)
l1_ratios = np.arange(0.1, 0.9, 0.025)
best_alpha_en, best_l1 = 0.0001, 0.1

#Hyperparameter vector space for Ridge
a_r = np.arange(1,10, 1, dtype=float)
b_r = np.arange(-6, 5, 1, dtype=float)
best_a_r, best_b_r = 1, -6

#Hyperparameter vector space for Lasso
a_l = np.arange(1,10, 1, dtype=float)
b_l = np.arange(-6, 5, 1, dtype=float)
best_a_l, best_b_l = 1, -6

#Hyperparameter vector space for XGBoost
n_estimators = np.arange(100, 1200, 100)
learning_rates = np.arange(0.005, 0.15, 0.005)
best_n, best_learn = 100, 0.005

best_score = float('inf')
print("ELASTICNET")
for alpha in alphas_en:
    for l1_ratio in l1_ratios:
        score = RMSLE_cross_val(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=0))
        print(alpha, l1_ratio, score)
        if score < best_score:
            best_alpha_en, best_l1, best_score = alpha, l1_ratio, score
print("BEST PARAMETERS: alpha = {}, l1-ratio = {} --- SCORE: {}\n:".format(best_alpha_en, best_l1, best_score))

best_score = float('inf')
print("RIDGE")
for a in a_r:
    for b in b_r:
        alpha = a * 10**b
        score = RMSLE_cross_val(Ridge(alpha=alpha, max_iter=10000, random_state=0))
        print(alpha, score)
        if score < best_score:
            best_a_r, best_b_r, best_score = a, b, score
print("BEST PARAMETERS: alpha = {} --- SCORE = {}\n:".format(best_a_r * 10**best_b_r, best_score))

best_score = float('inf')
print("LASSO")
for a in a_l:
    for b in b_l:
        alpha = a * 10**b
        score = RMSLE_cross_val(Lasso(alpha=alpha, max_iter=10000, random_state=0))
        print(alpha, score)
        if score < best_score:
            best_a_l, best_b_l, best_score = a, b, score
print("BEST PARAMETERS: alpha = {} --- SCORE: {}\n:".format(best_a_l * 10**best_b_l, best_score))

best_score = float('inf')
print("XGBOOST")
for n in n_estimators:
    for lr in learning_rates:
        score = RMSLE_cross_val(XGBRegressor(n_estimators=n, learning_rate=lr, objective="reg:squarederror", n_jobs=-1, random_state=0))
        print(n, lr, score)
        if score < best_score:
            best_n, best_learn, best_score = n, lr, score
print("BEST PARAMETERS: n_estimators = {}, learning_rate = {} --- SCORE: {}\n:".format(best_n, best_learn, best_score))

'''


# <h1>Training the models and blending</h1>
# 
# We have obtained the optimal values for our hyperparameters, so we start training the models.
# 
# We take a weighted average of the predictions from these 4 models to use as our final ensemble prediction.
# 
# It should be noted that this is quite a crude blending technique, a more sophisticated approach would yield more accurate predictions.

# In[ ]:


model_EN = ElasticNet(alpha=0.0025, l1_ratio=0.15, max_iter=10000, random_state=0)
model_EN.fit(X_train, np.log1p(y))
EN_preds = model_EN.predict(X_test)

model_R = Ridge(alpha=30, max_iter=10000, random_state=0)
model_R.fit(X_train, np.log1p(y))
R_preds = model_R.predict(X_test)

model_L = Lasso(alpha=0.0006, max_iter=10000, random_state=0)
model_L.fit(X_train, np.log1p(y))
L_preds = model_L.predict(X_test)

model_XGB = XGBRegressor(n_estimators=1000, learning_rate=0.055, objective="reg:squarederror", n_jobs=-1, random_state=0)
model_XGB.fit(X_train, np.log1p(y))
XGB_preds = model_XGB.predict(X_test)

ensemble_preds = 0.75*EN_preds + 0.05*R_preds + 0.05*L_preds + 0.15*XGB_preds 


# We apply ```npexp1()``` to our predictions and export them for submission.

# In[ ]:


preds = np.expm1(ensemble_preds)

output = pd.DataFrame({'Id': X_test.index,
                      'SalePrice': preds})

output.to_csv('submission.csv', index=False)

