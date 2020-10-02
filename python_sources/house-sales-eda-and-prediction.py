#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">House Sales EDA and Prediction</font></center></h1>
# 
# <br>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
#  - <a href='#21'>Load packages</a>  
#  - <a href='#22'>Load the data</a>   
# - <a href='#3'>Data exploration</a>   
#  - <a href='#31'>Check for missing data</a>  
#  - <a href='#32'>Features visualization</a>   
# - <a href='#4'>Build a baseline model</a>  
# - <a href='#5'>Model refinement</a> 
# - <a href='#6'>Submission</a>  
# - <a href='#7'>Model with cross-validation</a> 
# - <a href='#8'>Submission (2)</a> 
# - <a href='#9'>Blending</a>
# - <a href='#10'>Submission (3)</a>
# - <a href='#11'>Suggestions for further improvement</a>
# - <a href='#12'>References</a>      

# # <a id='1'>Introduction</a>  
# 
# This Kernel will take you through the process of **analyzing the data** to understand the **predictive values** of various **features** and the possible correlation between different features, **selection of features** with predictive value, **features engineering** to create features with higher predictive value, creation of a **baseline model**, succesive **refinement** of the model  through selection of features and, at the end, **submission** of the best solution found. 
# 
# 
# The dataset used for this tutorial contains sales information for houses in a certain US region.
# 
# The objective of the competition is to predict with good accuracy the sale **price** for each house in the test data.
# 
# The metric used for the competition is **RMSE**.
# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 

# # <a id='2'>Prepare the data analysis</a>   
# 
# 
# Before starting the analysis, we need to make few preparation: load the packages, load and inspect the data.
# 

# ## <a id='21'>Load packages</a>
# 
# We load the packages used for the analysis. There are packages for data manipulation, visualization and models.

# 

# In[ ]:


import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id='22'>Load the data</a>  
# 
# Let's see first what data files do we have in the root directory. 

# In[ ]:


PATH="../input/"
os.listdir(PATH)


# In[ ]:


train_df = pd.read_csv(PATH+"train.csv")
test_df = pd.read_csv(PATH+"test.csv")


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# # <a id='3'>Data exploration</a>  
# 
# We check the shape of train and test dataframes and also show a selection of rows, to have an initial image of the data.
# 
# 

# In[ ]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Both **train** and **test** files contains the following values:  
# 
# * **id** - the index of the transaction (in the dataset);  
# * **date** - the date of the transaction;  
# * **bedrooms** - the number of bedrooms;  
# * **bathrooms** - the number of the bathrooms;    
# * **sqft_living** - the number of square feet of the living area;    
# * **sqft_lot** - the number of square feet of the lot area;  
# * **floors** - the number of floors of the house;  
# * **waterfront** - flag to indicate if the house is on the waterfront;  
# * **view** - flag to indicate if the house has a view;  
# * **condition** - categorical value for the condition of the house;  
# * **grade** - categorical value for the grade of the property;  
# * **sqft_above** - the number of square feet of the living area above the ground;  
# * **sqft_basement** - the number of square feet of the living area below the ground;  
# * **yr_built** - year when the property was built;  
# * **yr_renovated** - year when the property was renovated; 
# * **zipcode** - zipcode for the property location;  
# * **lat** - latitude of the property location;  
# * **long** - longitude of the property location;  
# * **sqft_living15** - the number of square feet of the living area '15';  
# * **sqft_lot15** - the number of square feet of the lot area '15';  
# 
# 
# The **train** data has as well the target value, **price**. This is the sale price of the property. 
# 
# The competition objective is to estimate with highest accuracy the price of properties in the test set.
# 
# It is important, before going to create a model, to have a good understanding of the data. We will therefore explore the various features.

# ## <a id='31'>Check for missing data</a>  
# 
# Let's create a function that check for missing data in the two datasets (train and test).

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))


# In[ ]:


missing_data(train_df)


# In[ ]:


missing_data(test_df)


# There are no missing data in the train and test sets.

# ## <a id='32'>Features visualization</a>  
# 
# Let's explore each individual feature.  First, let's show the info about each feature.

# In[ ]:


train_df.info()


# Besides the date (in string format), all other data is numeric (either float or integer).  Also, there is no missing data. 
# Float values are the bathrooms (that have also fractional values), the floors (same as previous), lat/long (real numbers) and the price (same as previous).
# The rest, categorical or numeral, are integers.
# 
# Let's check now the distribution of each numeric column.

# In[ ]:


train_df.describe()


# We plot first the price distribution grouped by each categorical feature.

# In[ ]:


categorical_columns = ['waterfront', 'view', 'condition', 'grade']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(16,10))
for col in categorical_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'])
    plt.xlabel(col, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# Let's see also the comparison between train and test set distribution of categorical columns.

# In[ ]:


def plot_stats(feature):
    temp = train_df[feature].dropna().value_counts().head(50)
    df1 = pd.DataFrame({feature: temp.index,'Number of samples': temp.values})
    temp = test_df[feature].dropna().value_counts().head(50)
    df2 = pd.DataFrame({feature: temp.index,'Number of samples': temp.values})    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
    s = sns.barplot(x=feature,y='Number of samples',data=df1, ax=ax1)
    s.set_title("Train set")
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    s = sns.barplot(x=feature,y='Number of samples',data=df2, ax=ax2)
    s.set_title("Test set")
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()    


# In[ ]:


categorical_columns = ['waterfront', 'view', 'condition', 'grade']

for col in categorical_columns:
    plot_stats(col)


# We plot then the price distribution grouped by each numerical feature.

# In[ ]:


numerical_columns = ['bedrooms', 'bathrooms', 'floors']
i = 0
plt.figure()
fig, ax = plt.subplots(1,3,figsize=(18,4))
for col in numerical_columns:
    i += 1
    plt.subplot(1,3,i)
    sns.boxplot(x=train_df[col],y=train_df['price'])
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# We plot now the relative distribution of price and square feet (area).

# In[ ]:


area_columns = ['sqft_living','sqft_lot','sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

i = 0
plt.figure()
fig, ax = plt.subplots(3,2,figsize=(16,15))
for col in area_columns:
    i += 1
    plt.subplot(3,2,i)
    plt.scatter(x=train_df[col],y=train_df['price'],c='magenta', alpha=0.2)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('price', fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# Let's check the geographical info now.

# In[ ]:


geo_columns = ['lat','long']

i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(16,6))
for col in geo_columns:
    i += 1
    plt.subplot(1,2,i)
    plt.scatter(x=train_df[col],y=train_df['price'], c=train_df['zipcode'], alpha=0.2)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('price', fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show();


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(16,16))
ax=fig.add_subplot(1,1,1, projection="3d")
ax.scatter(train_df['lat'],train_df['long'],train_df['price'],c=train_df['zipcode'],alpha=.8)
ax.set(xlabel='\nLatitude',ylabel='\nLongitude',zlabel='\nPrice')


# Let's plot the distribution of price by zipcode.
# 

# In[ ]:


print("There are {} unique zipcodes.".format(train_df['zipcode'].nunique()))


# In[ ]:


plt.figure(figsize=(18,4))
sns.boxplot(x=train_df['zipcode'],y=train_df['price'])
plt.xlabel('zipcode', fontsize=8)
locs, labels = plt.xticks()
plt.tick_params(axis='x', labelsize=8, rotation=90)
plt.show();


# Let's plot the zipcode on lat/long.

# In[ ]:


plt.figure(figsize=(16,16))
plt.scatter(x=train_df['lat'],y=train_df['long'], c=train_df['zipcode'], cmap='Spectral')
plt.xlabel('lat', fontsize=12); plt.ylabel('long', fontsize=12)
plt.show();


# Let's look into more details about the date feature.

# In[ ]:


for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.weekofyear
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = pd.to_numeric(df['date'].dt.is_month_start)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = pd.to_numeric(df['dayofweek']>=5)


# And let's build some more engineering features.

# In[ ]:


for df in [train_df, test_df]:
    df['med_lat'] = np.round(df['lat'],1) 
    df['med_long'] = np.round(df['long'],1) 
    df['build_old'] = 2019 - df['yr_built']
    df['sqft_living_diff'] = df['sqft_living'] - df['sqft_living15']
    df['sqft_lot_diff'] = df['sqft_lot'] - df['sqft_lot15']
    df['bedroom_bathroom_ratio'] = df['bedrooms'] / df['bathrooms']


# In[ ]:


train_df.head()


# In[ ]:


date_columns = ['year', 'month', 'dayofweek', 'quarter']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(12,12))
for col in date_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=False)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# In[ ]:


date_columns = ['year', 'month', 'dayofweek', 'quarter']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(16,12))
for col in date_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=True)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# In[ ]:


date_columns = ['dayofyear', 'weekofyear']
i = 0
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(18,12))
for col in date_columns:
    i += 1
    plt.subplot(2,1,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=False)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# In[ ]:


date_columns = ['dayofyear', 'weekofyear']
i = 0
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(18,12))
for col in date_columns:
    i += 1
    plt.subplot(2,1,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=True)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# In[ ]:


date_columns = ['is_month_start', 'is_weekend']
i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,6))
for col in date_columns:
    i += 1
    plt.subplot(1,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=False)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# In[ ]:


date_columns = ['is_month_start', 'is_weekend']
i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,6))
for col in date_columns:
    i += 1
    plt.subplot(1,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=True)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


# Let's represent now the Pearson corelation matrix for all the numerical features.

# In[ ]:


features = ['bedrooms','bathrooms','floors',
            'waterfront','view','condition','grade',
            'sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15',
            'yr_built','yr_renovated',
            'lat', 'long','zipcode', 
            'date', 'dayofweek', 'weekofyear', 'dayofyear', 'quarter', 
            'is_month_start', 'month', 'year', 'is_weekend',
            'price']

mask = np.zeros_like(train_df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(18,18))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(train_df[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="Blues", 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75});


# # <a id='4'>Build a baseline model</a>  
# 
# 
# We start with a very basic model, with only two features.
# 
# But first, we will separate train dataset in train and valid sets.
# 

# In[ ]:


#We are using 80-20 split for train-test
VALID_SIZE = 0.2
#We also use random state for reproducibility
RANDOM_STATE = 2019

train, valid = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )


# In[ ]:


predictors = ['sqft_living', 'grade']
target = 'price'


# In[ ]:


train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


# In[ ]:


RFC_METRIC = 'mse'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier


# In[ ]:


model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)


# In[ ]:


model.fit(train_X, train_Y)


# In[ ]:


preds = model.predict(valid_X)


# Let's plot the features importance. This shows the relative importance of the predictors features for the current model. With this information, we are able to select the features we will use for our gradually refined models.

# In[ ]:


def plot_feature_importance():
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': model.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance',fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()   


# In[ ]:


plot_feature_importance()


# In[ ]:


print("RF Model score: ", model.score(train_X, train_Y))


# Let's evaluate the `rmse` score for training set and for valid set.

# In[ ]:


def rmse(preds, y):
    return np.sqrt(mean_squared_error(preds, y))


# In[ ]:


print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))


# # <a id='5'>Model refinement</a>  
# 
# We will now add succesivelly features and verify if the validation `rmse` error improves.

# In[ ]:


predictors = ['sqft_living', 'grade', 'sqft_above']
target = 'price'
train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values
model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
model.fit(train_X, train_Y)
preds = model.predict(valid_X)


# In[ ]:


plot_feature_importance()


# In[ ]:


print("RF Model score: ", model.score(train_X, train_Y))
print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))


# Let's add one more feature.

# In[ ]:


predictors = ['sqft_living', 'sqft_lot',
              'sqft_above', 'sqft_living15',
              'waterfront', 'view', 'condition', 'grade',
             'bedrooms', 'bathrooms', 'floors',
             'zipcode', 
              'month', 'dayofweek', 
              'med_lat', 'med_long',
              'build_old', 'sqft_living_diff', 'sqft_lot_diff',
             ]
target = 'price'
train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values
model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
model.fit(train_X, train_Y)
preds = model.predict(valid_X)
plot_feature_importance()
print("RF Model score: ", model.score(train_X, train_Y))
print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))


# # <a id='6'>Submission</a>  
# 
# We prepare now the submission file.

# In[ ]:


test_X = test_df[predictors] 
predictions_RF = model.predict(test_X)
submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = predictions_RF
submission.to_csv('submission.csv', index=False)


# #  <a id='7'>Model with cross-validation</a> 
# 
# Let's use a slightly improved model with cross-validation.

# In[ ]:


param = {'num_leaves': 51,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "metric": 'rmse',
         "verbosity": -1,
         "nthread": 4,
         "random_state": 42}


# In[ ]:


predictors = ['sqft_living', 'sqft_lot',
              'sqft_above', 'sqft_living15',
              'waterfront', 'view', 'condition', 'grade',
             'bedrooms', 'bathrooms', 'floors',
             'zipcode', 
              'month', 'dayofweek', 
              'med_lat', 'med_long',
              'build_old', 'sqft_living_diff', 'sqft_lot_diff',
             ]
target = 'price'


# In[ ]:


#prepare fit model with cross-validation
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_df))
predictions_lgb_cv = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()


# In[ ]:


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['price'].values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][predictors], label=train_df.iloc[trn_idx][target])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx][predictors], label=train_df.iloc[val_idx][target])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][predictors], num_iteration=clf.best_iteration)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = predictors
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions_lgb_cv += clf.predict(test_df[predictors], num_iteration=clf.best_iteration) / folds.n_splits
    
strRMSE = "RMSE: {}".format(rmse(oof, train_df[target]))
print(strRMSE)


# In[ ]:


def plot_feature_importance_cv():
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False).index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(12,6))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


# In[ ]:


plot_feature_importance_cv()


# # <a id='8'>Submission (2)</a>  
# 
# We prepare now the submission file.

# In[ ]:


submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = predictions_lgb_cv
submission.to_csv('submission_cv.csv', index=False)


# # <a id='9'>Blending</a>  
# 
# Let's combine the predictions obtained until now, using a weighted sum of the two models predictions.

# In[ ]:


predictions_blending = predictions_RF * 0.55 + predictions_lgb_cv * 0.45


# # <a id='10'>Submission (3)</a>  

# In[ ]:


submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = predictions_blending
submission.to_csv('submission_blending.csv', index=False)


# # <a id='11'>Suggestions for further improvement</a>  
# 
# To further develop this Kernel and obtain better performance in the competition, try the followings:
# * experiment with different algorithms (linear regressor, catbost, xgb, lgb etc.)
# * add more features to your model;
# * after adding each new feature, test if the cross-validation score is improving (your rmse for cross-valid should decrease);  
# * start to experiment more with engineered features; see the Reference section of this Kernel for suggestions;  
# * use hyperparameter optimization for your model to set the best hyperparameters;  
# 
# **Important to remember**: before submitting a new solution, try to get better score (or better cross-validation score) for your model;  do not use the public score on the leaderboard as the unique criteria for evaluating your model performance, since this is calculated with only 30% of the test set.
# 
# **Happy Kaggling**!
# 

# # <a id='12'>References</a>  
# 
# [1]  Root mean square error,  https://en.wikipedia.org/wiki/Root-mean-square_deviation   
# [2]  Example of using cross validation and scalling, https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction   
# [3] Tutorial for classification, https://www.kaggle.com/gpreda/tutorial-for-classification   
# [4] Regression models, https://www.kaggle.com/toraaglobal/soilpropertyprediction   
# [5] Regression models, https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard  
# [6] Regression models, https://www.kaggle.com/gpreda/elo-world-high-score-without-blending  
# 
