#!/usr/bin/env python
# coding: utf-8

# This kernal was made for all the dreamers out there that took the path of bravey to become the greatest Data Scientiest they could be!
# 
# **I hope you will find value in the content that I'm about to share, don't forget to share the love by upvoting this project of mine.** it would be a great sign of support, love and apprecation. A great shoutout for kaggle as well from making all of this possible. 
# 
# Hurrah!!
# 
# **Importing and Exploring**
# 
# First thing's first. let's do some importing of the libraries that we'll use (i assure you the project will get more entertaining as you progress)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's import more libraries that we'll be using and try to take a peak at the some of the training data:

# In[ ]:


# importing some libraries for visulizations
import matplotlib.pyplot as plt
import seaborn as sns

# importing sklearn to select the model that will be fitting out data into
# we will train_test_split to divide the data
# we will use cross_val_score to determine best accuracy 
from sklearn.model_selection import train_test_split, cross_val_score

# import the data into dataframes using pandas library
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# Now, let's explore the data a bit before we dig into the bigger stuff:
# 
# always, **ALWAYS**!! check the shape of your data before you start!!

# In[ ]:


train.shape, test.shape


# hmm.. as predicted it seems that we have 1 more row and 1 more column for the training data.
# Thank God we checked before it was too late ;)
# 
# okay, let's figure out which column it is:

# In[ ]:


cols = {}
uniqueCols =[]
for col in test.columns:
    if col not in cols:
        cols[col]=1
    else:
        cols+=1
for col in train.columns:
    if col not in cols:
        uniqueCols.append(col)

print( uniqueCols)
    


# yeah, that makes sense. since we need to predict the 'SalePrice' for the test data it can't be a column there :)
# 
# okay, moving on..
# 
# let's drop the 'Id' columns since the are unnecessary for the prediction process:
# 

# In[ ]:


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# 
# A good practice when trying to understand dependent variables (in this case it's 'SalePrice') is to look at it's normal distribution and try to understand it's nature.
# 
# we will look also at it's skewness (if it's equal to 0 it means that this variable is evenly distributed) and kurtosis (the standard value should be 3)

# In[ ]:


sns.distplot(train['SalePrice'], bins=20, rug=True)

print("Skewness: %0.2f" %train['SalePrice'].skew())
print("Kurtosis: %0.2f" %train['SalePrice'].kurt())


# Another good practice (and also the best first move) when doing DS projects is to look for all sorts of correlations between all features.

# In[ ]:


corrmat = train.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);


# Whoa.. now that's a messy graph!
# 
# When we have TOO MANY features, it is best to filter correlations first. in this case we will settle for 0.5 correlation or above.
# 

# In[ ]:


corrmat = train.corr()
# extracting the relevant features
filteredCorrMat_features = corrmat.index[abs(corrmat['SalePrice'])>=0.5]
plt.figure(figsize=(12,12))
# performing corr on the chosen features and presenting it on the heatmap
sns.heatmap(train[filteredCorrMat_features].corr(),annot=True,cmap='winter')


# In this way, we selected only the most important features that will serve us as the best predictors for SalePrice.
# 
# Furthermore, we find that the columns ** 'OverallQaul', 'GrLivArea' ** have the highest corrlations with SalePrice.
# 
# It is also very important to notice correlations amongst other features like:
# * ** 'GrLivArea' ** and 'TotalRmsAbvGrd' (corr= 0.83)
# *  'GarageCars' and 'GarageArea' (corr= 0.88)
# * 'lstFlrSF' and 'TotalBsmtSF' (corr= 0.82)
# 
# It seems like OverallQaul serves as the most reliable feature for predicting SalePrice, but don't believe me, let's just see it visually:
# 

# In[ ]:


sns.barplot(train.OverallQual,train.SalePrice)


# Before we dive into feature engineering, let's join our training data and test data so that we won't get lost later and stay consistent with changes across the data.

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# **Pre-processing** and **Feature Engineering**
# 
# **Stage 1: Handling Missing Data**
# 
# Now we approach one of the most important sub sections of DS project. 
# 
# What do we do about missing data?
# Do we ommit it?
# Do we replace it with the mean? the median?
# 
# There are many considerations when we are dealing with missing data. 
# As a first step, let's see which data is missing and it's weight in percentage.

# In[ ]:


totalMissing = all_data.isnull().sum().sort_values(ascending=False)
percentage = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)

missingData = pd.concat([totalMissing,percentage],axis=1,keys=['Total','Percentage'])
missingData.head(20)


# as always, it's best to see things visually before moving forward

# In[ ]:


plt.subplots(figsize=(15,20))
plt.xticks(rotation='90')
sns.barplot(x=totalMissing.index[:24],y=percentage[:24])
plt.xlabel('features')
plt.ylabel('percentage of missing data')
plt.title('percent of missing data by feature')
plt.show()


# Now let's drop any feature that has more that 50% missing data. In this case, the features 'PoolQC', 'MiscFeature', 'Alley' and 'Fence' don't seem to add much either way. perhaps that why this data is mostly missing in the first place. 
# 
# Since most of this data is missing and since such data does not seem to be if high correlation with our dependent variable. let's go ahead and drop them!

# In[ ]:


# columns to be dropped
columnsToDrop = missingData[missingData['Percentage']>50].index

all_data = all_data.drop(columnsToDrop, axis=1)
# test = test.drop(columnsToDrop, axis=1)
print(all_data.shape)


# ** Handling categorcial missing data**
# 
# We will replace missing data for the catigorical features with None
# 
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these **categorical** basement-related features, NaN means that there is no basement.
# 
# FireplaceQu, GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2','BsmtFullBath', 'BsmtHalfBath',
            'GarageType', 'GarageFinish', 'GarageQual', 'BsmtUnfSF','BsmtFinSF1','BsmtFinSF2',
            'GarageCond', 'FireplaceQu', 'MasVnrType', 'Exterior2nd'):
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')


# **Handling numerical missing data**
# 
# Now, lets take care of missing data for the numerical features.
# 
# 

# In[ ]:


#GarageYrBlt replacing missing data with 0
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

# NA most likely means no masonry veneer for these houses. We can fill in 0
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# let's drop YrSold since it's also not correlated with 'SalePrice'
all_data = all_data.drop('YrSold', axis=1)

# Electrical has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# 'RL' is by far the most common value. So we can fill in missing values with 'RL'
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
# Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling.
all_data = all_data.drop(['Utilities'], axis=1)

# data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#  Replacing missing data with 0 (Since missing in this case would imply 0.)
for col in ('TotalBsmtSF', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
#  Replacing missing data with the most common
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# Now let's check if there are any missing values left

# In[ ]:


all_data.isnull().sum().sort_values(ascending=False) #check


# Would you look at that.
# Ain't that beautiful :)

# **Stage 2: Outliers! **
# 
# In statistics, an outlier is an observation point that is distant from other observations. usually the distance is measured by standard deviations. such points are usually produced by some sort of error or simply do not represent any real data and just get in the way to make our predictions less accurate. 
# 
# The approach we're going to go with is simply remove data that's below the 0.05 percentile or above the 0.9 percentile (check out this link to better understand quantiles and percentiles: http://www.statisticshowto.com/quantile-definition-find-easy-steps/). 

# In[ ]:


from pandas.api.types import is_numeric_dtype
def remove_outliers(df):
    low = .05
    high = .9
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

remove_outliers(all_data).head()


# huh.. doesn't see like we have any outliers in the chosen quantile.
# 
# Nonetheless, i would like to explore the feature 'GrLivArea' and see if i could visually spot outliers.

# In[ ]:


plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers (those bastards). Therefore, we can safely delete them.

# In[ ]:


#Deleting outliers
tempTrain = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(x = tempTrain['GrLivArea'], y = tempTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# Outliers removal is not always safe. We decided to delete these two as they are very huge and really bad ( extremely large areas for very low prices). There are probably other outliers in the training data that we need to handle. but removing outliers always comes with a price. For the time being we will settle for the above explination and demonstration of outliers.
# 
# **however, notice that this is just a demonstration and we did not in fact change our data (all_data). we will however deal with outliers later using StandardScaler/RobustScaler**
#  
# 
# **Stage 3: Target Variable**
# 
# SalePrice is the variable we need to predict. So let's do some analysis on this variable first (remember the skewness?).
# 

# In[ ]:


from scipy import stats
from scipy.stats import norm

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function 
# mu is the mean across the population (more accurately, given data)
# and sigma is the standard deviation across the population
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot (probability plot)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# We can clearly see that the 'SalePrice' distrution plot is right skewed (also indicative from the probability plot). 
# 
# Since we will be working with linear models, our next step would be to transform our distrubtion to look more normally distributed. We will do that by appling log(1+x) to all elements of 'SalePrice'.
# 
# let's begin :)
# 

# In[ ]:


#Appling log(1+x) to all elements of 'SalePrice'
y_train = train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# Ain't that beautiful :)
# 
# it seems that we have come a long way with our data.
# 
# Before we continue with feature processing, there's this one thing that i would like to be aware of. 
# It's important to realize that till now, our data (all the features that we have been working with) is simply gathered and put into a table. thinking outside the box leads us to one important understanding. Can we combine features in order to create a new one that's potentially more correlatioed with our target variable?
# 
# of course we can! take for example the ground area of the whole house, includeing the basement, the first floor, second floor..
# the total square foor area of the house is a dominant feature for the prediction of the price of the house. 
# 
# let's take care of it then ;)

# In[ ]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# Up next, let's see what's up with skewed features( hmm.. screwed features..)
# 
# we'll focus on the numerical features and how skewed they are.
# 

# In[ ]:


from scipy.stats import skew 

# extracting numerical features
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
numeric_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :numeric_features})
skewness.head(10)

print(all_data.shape)


# Now we will use the Box-Cox transformation to deal with the highly skewed values (any features with skewness value > 3).

# In[ ]:


highly_skewed = ['PoolArea','LotArea','KitchenAbvGr','ScreenPorch']

from scipy.special import boxcox1p
lam = 0.15
for feat in highly_skewed:
    all_data[feat] = boxcox1p(all_data[feat], lam)


# Now, let's get all the dummies :)
# 
# (this is basically converting categorical variables into dummy/indicator variables)

# In[ ]:


all_data = pd.get_dummies(all_data)


# Remember when we combined the training data and the test data to make one big all_data dataframe. well it's time to split them once more.

# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]

# train.drop('SalePrice',axis=1,inplace=True)
# train['SalePrice']
print(train.shape, test.shape)


# phew!! that was a long jounrey. however, the interesting part is just about to start ;)
# 
# 
# **Modeling**
# First step's first. let's import all the classes that we will be working with:

# In[ ]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from xgboost import XGBRegressor


# Now let's split our data into training data and test data. We do this so we can validate our results later on.

# In[ ]:


# train= train.drop(train.index[[0,1]],axis=0)
print(y_train.shape, train.shape)


# In[ ]:



X_train,X_test,y_train2,y_test = train_test_split(train.values,y_train,test_size = 0.3,random_state= 0)
X_train.shape,X_test.shape,y_train2.shape,y_test.shape


# What i would like to do next is define the root mean squared error function for calculating the accuracy of the different modeling algorithms we are going to use.
# 
# **Notice** that we are using cross validation technique with 5 folds.

# In[ ]:



# Scoring - Root Mean Squared Error
def rmse_CVscore(model,X,y):
    return np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))


# There are many modeling algorithms we can use, let's check sevearl and see their accuracy scores.
# 
# we'll start with **LASSO Regression**
# (since this model is senstive to outliers, we can use the RobustScaler to deal with them. the RobustScaler is a standardization technique that allows us to standardize our data using the mean and the standard deviation to be able to compare between data points)

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.05, random_state=1))

score = rmse_CVscore(lasso,X_train,y_train2)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# **ElasticNet** regression model (which basically combines Ridge regression and Lasso regression):

# In[ ]:



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3))

score = rmse_CVscore(ENet,X_train,y_train2)
print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# **Gradient Boosting Regression**
# In order to take care of outliers for this model we can use the huber loss function.

# In[ ]:



GBoost = GradientBoostingRegressor(n_estimators=1000,max_depth=4,
                                   learning_rate=0.05,
                                   max_features='sqrt',
                                   loss="huber",random_state =5)
score = rmse_CVscore(GBoost,X_train,y_train2)
print("\nGradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#  **XGBRegressor** 

# In[ ]:


# create pipeline
# my_pipeline = make_pipeline(
#     SimpleImputer(),
# XGBR = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state =7,max_depth=3)
# )

# score = rmse_CVscore(XGBR,X_train,y_train2)
# print("\nXGBRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# **LightGBM **

# In[ ]:



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=1000)

score = rmse_CVscore(model_lgb, X_train, y_train2)
print("\nLightGBM Regressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# As you can see, sometimes the more complicated the model, the poorer the results.

# What we can do now is a strategy called averaging base models. we combine some of the used models and average their results to get a more accurate result.
# 
# let's do this in a class:
# 

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # creating clones of the original models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # fitting our data to the models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #making predictions on our fitted models and averaging them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# In[ ]:


averaged_models = AveragingModels(models = (ENet,GBoost, lasso, model_lgb))

score = rmse_CVscore(averaged_models,X_train, y_train2)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Awesome!! it seems that we have a great estimate of accuracy for our model.
# 
# **Final Training and Prediction**
# 

# In[ ]:


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
print("test shape: {}, train shape: {}".format(test.shape, y_train.shape))

# train=train.drop(train.index[[0,4]],axis=0)

averaged_models.fit(train, y_train)
train_pred = averaged_models.predict(train)
avg_pred = np.expm1(averaged_models.predict(test))

print(rmse(y_train, train_pred))


# Now that's some HOT results!!!
# 
# let's go ahead and submit our hard work :)

# In[ ]:


# test['Id'].shape
# avg_pred.shape
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = avg_pred
sub.to_csv('submission.csv',index=False)
#train.shape
#test.shape


# Thank you for your interest in this project. I hope it brought value to your Data science endouvers and i hope you had some fun along the way!
# 
# This project was possible with the help of the following kernals:
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# https://www.kaggle.com/tecknomart/basic-data-science-skillset-we-must-have
# https://www.kaggle.com/bsivavenu/house-price-calculation-methods-for-beginners
# 
# Cheers!!
