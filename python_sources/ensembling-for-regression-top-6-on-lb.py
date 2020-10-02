#!/usr/bin/env python
# coding: utf-8

# In this notebook, let us try and explore the data given for **House Prices: Advanced Regression Techniques**. Before we dive deep into the data, let us know a little more about the competition.
# 
# **What decides a House Price?**
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But the ral world experiments proves that price negotiations are much more dependent on other **Factors** rather than the number of bedrooms or a white-picket fence.
# 
# **Objective:**
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# # Contents
# 1. [Importing Packages](#p1)
# 2. [Loading Data](#p2)
# 3. [Imputing Null Values](#p3)
# 4. [Feature Engineering](#p4)
# 5. [Creating, Training, Evaluating, Validating, and Testing ML Models](#p5)
# 6. [Submission](#p6)

# **Let's Do Some Real "Work"**
# 
# We are creating this Notebook to illustrate that how you can "Approach almost any Regression Problem" 
# 
# **A Comprehensive Checklist for Solving Any Regression Problem:**
# 
# * Data Fetching
# * Understanding the Data
# * Checking the Skwewness of the Output Variable
# * Performing Log Transformation (if required)
# * Exploratory Data Analysis
# * Analysing Correlation
# * Finding out Important Predictors
# * Feature Engineering: -
#      1. Missing Values
#      2. Outliers
#      3. Categorical Feature Encoding     
# * Creating Folds and Defining Fold Map
# * Defining Models
# * Fitting the Model and Running Cross Validation
# * Stacking and Ensembling
# * Hyperparameter Optimization
# 
# The above mentioned checklist is very importnant for solving any regression based problem. The similar kind of checklist can also be prepared for other Machine Learning based problems. I will cover them in my upcoming kernals.
# 
# Let's begin!
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import some of the necessary libraries
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
import seaborn as sns
sns.set(rc={'figure.figsize':(15,12)})
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let us first import the training and the test data.

# In[ ]:


train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# Let's do a little exploration on the training file.

# In[ ]:


train_df.head(10)


# In[ ]:


train_df.columns


# In[ ]:


len(train_df.columns)


# In[ ]:


train_df['SalePrice'].describe()


# In[ ]:


train_df.shape


# **Thus,**
# 
# * Total Observations in the Training Data : - 1460
# * Total Features in the Training Data : - 79, Excluding Id Column and Dependent Variable i.e. SalePrice
#     
# **Description of the Training Data: -**
# 
# 1. Mean Value      -->  180921.195890
# 2. Std. Deviation  -->  79442.502883
# 3. Min Value       -->  34900.00
# 4. Max Value       -->  755000.00

# Now let's have the Training and Testing ID's  aved in a dataframe for future references. As you know for any machine learning based problem the Id doesn't make a feature, so we are going to drop the ID column from our train and test dataframe.

# In[ ]:


train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)
#Deleting outliers
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)


# Here, **SalePrice** is what we have to Predict. So we will first start with Checking the Distribution of the Variable and Let's see how much Skewness it has got.

# In[ ]:


sns.set(rc={'figure.figsize':(18,8)})
sns.distplot(train_df['SalePrice'],fit=norm)

(mu, sig) = norm.fit(train_df['SalePrice'])
#Now plot the distribution
plt.legend(['Normal Distribution Curve ($\mu=$ {:.2f} & $\sigma=$ {:.2f} )'.format(mu, sig)])
plt.ylabel('Frequency')
plt.show()


# **By Analyzing the Graph we can see the following:**
# 
# * Deviation from the Normal Distribution.
# * Have Positive Skewness.
# * Show Peakedness
# 
# Also let's see the measurement of Skewness: 

# In[ ]:


print("Skewness of Sale Price is: ",train_df['SalePrice'].skew())


# **Looks like our Data is Skewed Towards Right.**
# 
# * We are normalising the data by simply takithe Natural Log and then adding 1.
# 
# **Why do we need to make the Data Normal?**
# 
# Since Machine Learning or Data Science is nothing but Glorified Statistics at the end of the day and most of the algorithms assumes that the data is that the data is normal and it calculates various stats assuming this. So the more the data is close to normal, the better it fits the assumption.
# 
# **Log Transformation: -**

# In[ ]:


train_df['SalePrice'] = np.log(train_df['SalePrice']+1)
sns.distplot(train_df['SalePrice'],fit=norm)

(mu, sig) = norm.fit(train_df['SalePrice'])
#Now plot the distribution
plt.legend(['Normal Distribution Curve ($\mu=$ {:.2f} & $\sigma=$ {:.2f} )'.format(mu, sig)])
plt.ylabel('Frequency')
plt.show()


# **This is the Normalised Data**

# In[ ]:


print("Skewness of Sale Price is: ",train_df['SalePrice'].skew())


# **Now, Let's do Some More Feature Analysis**
# 
# The First and Foremost Important thing is to See that What are the Relevant Features. Not all features might be useful for our prediction and Having all the unnecessary features is going to make our model complex and we don't want the dimensionality to be huge!

# Let's Generate the Correlation Matrix [](http://)

# In[ ]:


corremap = train_df.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corremap, vmax=0.9, square=True)


# **From the Correlation Map, We Got: **
# 
# * OverallQual, GrLivArea, TotalBsmtSF, GarrageCars, GarrageArea, 1stFlrSF, YearBuilt, FullBath are the most important Predictors.
# * We can see from the above graph that how significantly they are related to our output variable "SalePrice"

# **Let's have an Eagle's Eye View! **

# In[ ]:


sns.set()
columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF','GarageCars', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt','SalePrice']
sns.pairplot(train_df[columns], size = 2)
plt.show();


# **According to our tarot card, these are the variables most correlated with 'SalePrice'**
# 
# * 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'. 
# * 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, as we discussed in the last sub-point,    the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. 
# * 'TotalBsmtSF' and '1stFloor' also seem to be twin brothers. 
# * 'FullBath'?? Really?
# * 'TotRmsAbvGrd' and 'GrLivArea', twin brothers again.
# * It seems that 'YearBuilt' is slightly correlated with 'SalePrice'. Honestly, it scares me to think about 'YearBuilt' because I start     feeling that we should do a little bit of time-series analysis to get this right.

# Now Let's See the Relationship Between the Predictors and the "SalePrice"
# 
# **Let's See who makes the Best Couple?**

# In[ ]:


var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(var,'SalePrice');


# It seems that 'SalePrice' and 'GrLivArea' are really in love with each other, with a **linear relationship**

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(var,'SalePrice');


# 'TotalBsmtSF' also have a great bond with 'SalePrice' but this seems a much more emotional relationship! Everything is ok and suddenly, in a strong linear (exponential?) reaction, everything changes. We can call it a **Mood Swing!** Moreover, it's clear that sometimes 'TotalBsmtSF' closes in itself and gives no credit to 'SalePrice'.

# **Let's Analyze others as well!**

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GarageArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(var,'SalePrice');


# In[ ]:


#scatter plot grlivarea/saleprice
var = '1stFlrSF'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(var, 'SalePrice');


# **Analyzing the Categorical Predictors**

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.boxplot(x=var,y='SalePrice',hue=var,data=data)


# In[ ]:


#box plot overallqual/saleprice
f, ax = plt.subplots(figsize=(20, 16))
plt.xticks(rotation='90')
var = 'YearBuilt'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.boxplot(x=var,y='SalePrice',data=data)


# In[ ]:


#box plot overallqual/saleprice
var = 'GarageCars'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.violinplot(x=var,y='SalePrice',data=data,palette='rainbow', hue = 'GarageCars')


# In[ ]:


var = 'FullBath'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.violinplot(x=var,y='SalePrice',data=data,palette='rainbow', hue = 'FullBath')


# **Now that we are done with most of the Feature Analysis, Let's Beging with the Feature Engineering!**

# Let's Concat the Training and the Test Data to a Complete Dataframe. This has to be done because both our training and testing data might contain missing values, outliers and may require categorical features to be handelled. 
# 
# Therefore rather than doing them seperatly we can do it together to save our time, so that we can waste the time somewhere else!

# In[ ]:


train_df.shape


# In[ ]:


ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
comp_data = pd.concat((train_df, test_df)).reset_index(drop=True)
comp_data.drop(['SalePrice'], axis=1, inplace=True)
print("Comp_data size is : {}".format(comp_data.shape))


# **Our Feature Engineering begins with Handling Missing Data!**

# In[ ]:


missing_val = comp_data.isnull().sum().sort_values(ascending=False)
missing_val_df = pd.DataFrame({'Feature':missing_val.index, 'Count':missing_val.values})
missing_val_df = missing_val_df.drop(missing_val_df[missing_val_df.Count == 0].index)
missing_val_df


# The above dataframe illustrates the count of Missing values corresponding to each observation.

# **Numbers our not my Stuff! Let's See some Graphs!**

# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='60')
plt.title('Count of Missing Data Per Feature', fontsize=15)
sns.barplot(x = 'Feature', y = 'Count', data = missing_val_df,
            palette = 'cool', edgecolor = 'b')


# **Now it looks Beautiful!**

# So, there are 2920 observations in are Complete dataframe, i,.e 1460 in both train and test dataframe. Now if we see the top three predictors from our missing value dataframe we see that most of them are close to 2920 that means,most of the observations from those predictors are not present.

# **Let's Handle the Missing Values!**

# **2. Handling Missing Values in Numerical Data**

# In[ ]:


comp_data["LotFrontage"] = comp_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    comp_data[col] = comp_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    comp_data[col] = comp_data[col].fillna(0)
comp_data['MSZoning'] = comp_data['MSZoning'].fillna(comp_data['MSZoning'].mode()[0])
comp_data["MasVnrArea"] = comp_data["MasVnrArea"].fillna(0)
comp_data['Electrical'] = comp_data['Electrical'].fillna(comp_data['Electrical'].mode()[0])
comp_data['SaleType'] = comp_data['SaleType'].fillna(comp_data['SaleType'].mode()[0])
comp_data['KitchenQual'] = comp_data['KitchenQual'].fillna(comp_data['KitchenQual'].mode()[0])
comp_data['Exterior1st'] = comp_data['Exterior1st'].fillna(comp_data['Exterior1st'].mode()[0])
comp_data['Exterior2nd'] = comp_data['Exterior2nd'].fillna(comp_data['Exterior2nd'].mode()[0])


# **3. Handling Missing Values in Categorical Data**

# In[ ]:


for col in ('PoolQC', 'MiscFeature', 'Alley'):
    comp_data[col] = comp_data[col].fillna('None')
comp_data["MasVnrType"] = comp_data["MasVnrType"].fillna("None")
comp_data["Fence"] = comp_data["Fence"].fillna("None")
comp_data["FireplaceQu"] = comp_data["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    comp_data[col] = comp_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    comp_data[col] = comp_data[col].fillna('None')
comp_data['MSSubClass'] = comp_data['MSSubClass'].fillna("None")
comp_data['SaleType'] = comp_data['SaleType'].fillna(comp_data['SaleType'].mode()[0])
comp_data = comp_data.drop(['Utilities'], axis=1)
comp_data['OverallCond'] = comp_data['OverallCond'].astype(str)
comp_data["Functional"] = comp_data["Functional"].fillna("Typ")  


# **Let's Handle the Categorical Features!**

# In[ ]:


comp_data['MSSubClass'] = comp_data['MSSubClass'].apply(str)
comp_data['YrSold'] = comp_data['YrSold'].astype(str)
comp_data['MoSold'] = comp_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for col in columns:
    labl = LabelEncoder() 
    labl.fit(list(comp_data[col].values)) 
    comp_data[col] = labl.transform(list(comp_data[col].values))     
print('Shape all_data: {}'.format(comp_data.shape))
comp_data['TotalSF'] = comp_data['TotalBsmtSF'] + comp_data['1stFlrSF'] + comp_data['2ndFlrSF']


# In[ ]:


comp_data.shape


# **Looking at Skewed Features**

# In[ ]:


from scipy.stats import norm, skew
numeric_feats = comp_data.dtypes[comp_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = comp_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# **Box Cox Transformation on Skewed Features**

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    comp_data[feat] = boxcox1p(comp_data[feat], lam)


# Creating Dummy Columns for Label Encoding the Categorical Features.

# In[ ]:


comp_data = pd.get_dummies(comp_data)
print(comp_data.shape)


# Now that we are done with our Feature Engineering,We can now seperate thhe Training and Test Data from the complete data.

# In[ ]:


train_df = comp_data[:ntrain]
test_df = comp_data[ntrain:]


# In[ ]:


train_df.shape


# **Let's Create the Folds for Our Cross Validation Model**

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse= np.sqrt(-cross_val_score(model, train_df.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# **Let's Define the Models and Do a Scoring!**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge,ElasticNet,Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# **Analyze the Above Results Well!**

# In[ ]:


from sklearn.metrics import mean_squared_error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# **Final Training and Prediction**

# * XGBoost

# In[ ]:


model_xgb.fit(train_df, y_train)
xgb_train_pred = model_xgb.predict(train_df)
xgb_pred = np.expm1(model_xgb.predict(test_df))
print(rmsle(y_train, xgb_train_pred))


# * LightGBM

# In[ ]:


model_lgb.fit(train_df, y_train)
lgb_train_pred = model_lgb.predict(train_df)
lgb_pred = np.expm1(model_lgb.predict(test_df.values))
print(rmsle(y_train, lgb_train_pred))


# **And Finally! Let's Make the Submission Dataframe!**
# **You have Done it!**

# We are going to generate our final submission by Ensembling the XGBoost and LightGBM Results

# In[ ]:


ensemble = xgb_pred*0.5 + lgb_pred*0.5


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = ensemble

print("Creating Submission File")
submission.to_csv("submission.csv", index=False)


# **Conclusion**
# 
# That's it! We reached the end of our exercise. We Saw how a Simple Model gave us Great Results!
# 
# If you liked the Kernal then don't forget to hit the Upvote! :) Also, Suggestion are always Welcomed! Post your Doubts and Suggestions on the Comment Section.

# **References**
# 
# * [COMPREHENSIVE DATA EXPLORATION WITH PYTHON](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
