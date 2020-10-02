#!/usr/bin/env python
# coding: utf-8

# # House_Price_Prediction : Regression_Model
# 
# ![](http://investmarbellaproperty.com/wp-content/uploads/2019/04/4c96032809c10d54e3e216015aecf32a_XL-1-1080x675.jpg)

# ### Data Fields (or Variables)
# 
# Here's a brief version of what you'll find in the data description file.
# 
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

# # Project Work Flow
# * Problem Defintion : 
# It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
# 
# * Loading Packages or Import Libraries
# * Gathering Data or Data Collection
# * Exploratory Data Analysis(EDA)
#     - Data Analysis
#     - Data Visualization
#     - Data Pre-processing
#     - Data Wraggling
# * Training and Testing the model
# * Evaluation
# * Submission

# # Loading Packages or Import Libraries
# Loading and Inspecting Data

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


import scipy.stats as stats
from scipy import stats
from scipy.stats import pointbiserialr, spearmanr, skew, pearsonr


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn import linear_model


# In[ ]:


house_train = pd.read_csv("../input/train.csv")
house_test = pd.read_csv("../input/test.csv")


# In[ ]:


house_train.shape, house_test.shape


# In[ ]:


house_train.head()


# In[ ]:


house_test.head()


# ## Exploratory Data Analysis

# ### Data Analysis

# #### Analysis : Sales Price (Target or Dependent Variable)

# In[ ]:


# "Descriptive Statistics": Summary of Target Variable
house_train['SalePrice'].describe()


# In[ ]:


# Let's plot histogram to check data is normally distributed or not?
fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(house_train['SalePrice'])


# ##### The distribution does not look normal, it is positively skewed, some outliers can also be seen. Simple log transformation might change the distribution to normal. 

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % house_train['SalePrice'].skew())
print("Kurtosis: %f" % house_train['SalePrice'].kurt())


# In[ ]:


house_train.info()


# In[ ]:


house_train.describe(include='all')


# Quite a lot of variables. Many categorical variables, which makes analysis more complex. And a lot of missing values. Or are they merely missing values? There are many features for which NaN value simply means an absense of the feature (for example, no Garage).

# #### Lets' check corelation of target varible with other variables

# In[ ]:


#correlation matrix
c_mat = house_train.corr()
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(c_mat, square=True)


# In[ ]:


# Highly Correlated Features or Variables
c_mat = house_train.corr()
top_corr_features = c_mat.index[abs(c_mat["SalePrice"])>0.4]
plt.figure(figsize=(10,10))
g = sns.heatmap(house_train[top_corr_features].corr(),annot=True)


# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(house_train[cols], size = 2.5)
plt.show()


# #### Relationship with numerical variables

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


# #### Relationship with categorical features

# In[ ]:


#Box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


# The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.

# In[ ]:


var = 'YearBuilt'
data = pd.concat([house_train['SalePrice'], house_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


# #### Pivotal Features

# In[ ]:


house_train[['OverallQual','SalePrice']].groupby(['OverallQual'],
as_index=False).mean().sort_values(by='OverallQual', ascending=False)


# In[ ]:


house_train[['GarageCars','SalePrice']].groupby(['GarageCars'],
as_index=False).mean().sort_values(by='GarageCars', ascending=False)


# In[ ]:


house_train[['Fireplaces','SalePrice']].groupby(['Fireplaces'],
as_index=False).mean().sort_values(by='Fireplaces', ascending=False)


# ### Let's Check Missing or Null Values in data
# Missing values in the training data set can affect prediction or classification of a model negatively.
# 
# But filling missing values with mean/median/mode or using another predictive model to predict missing values is also a prediction which may not be 100% accurate, instead you can use models like Decision Trees and Random Forest which handle missing values very well.

# In[ ]:


house_train.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


#plot of missing value features
plt.figure(figsize=(12, 8))
sns.heatmap(house_train.isnull())
plt.show()


# In[ ]:


house_test.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


#plot of missing value features
plt.figure(figsize=(12, 8))
sns.heatmap(house_test.isnull())
plt.show()


# In[ ]:


total = house_train.isnull().sum().sort_values(ascending=False)
percent = (house_train.isnull().sum()/house_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


total = house_test.isnull().sum().sort_values(ascending=False)
percent = (house_test.isnull().sum()/house_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# #### Imputting missing values
# 

# In[ ]:


#Create a list of column to fill NA with "None" or 0.
to_null = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'BsmtFullBath', 'BsmtHalfBath',
           'PoolQC', 'Fence', 'MiscFeature']
for col in to_null:
    if house_train[col].dtype == 'object':

        house_train[col].fillna('None',inplace=True)
        house_test[col].fillna('None',inplace=True)
    else:

        house_train[col].fillna(0,inplace=True)
        house_test[col].fillna(0,inplace=True)


# In[ ]:


#Fill NA with common values.
house_test.loc[house_test.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
house_test.loc[house_test.MSZoning.isnull(), 'MSZoning'] = 'RL'
house_test.loc[house_test.Utilities.isnull(), 'Utilities'] = 'AllPub'
house_test.loc[house_test.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
house_test.loc[house_test.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
house_test.loc[house_test.Functional.isnull(), 'Functional'] = 'Typ'
house_test.loc[house_test.SaleType.isnull(), 'SaleType'] = 'WD'
house_train.loc[house_train['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
house_train.loc[house_train['LotFrontage'].isnull(), 'LotFrontage'] = house_train['LotFrontage'].mean()
house_test.loc[house_test['LotFrontage'].isnull(), 'LotFrontage'] = house_test['LotFrontage'].mean()


# There are several additional cases: when a categorical variable is None, relevant numerical variable should be 0. For example if there is no veneer (MasVnrType is None), MasVnrArea should be 0.

# In[ ]:


house_train.loc[house_train.MasVnrType == 'None', 'MasVnrArea'] = 0
house_test.loc[house_test.MasVnrType == 'None', 'MasVnrArea'] = 0
house_test.loc[house_test.BsmtFinType1=='None', 'BsmtFinSF1'] = 0
house_test.loc[house_test.BsmtFinType2=='None', 'BsmtFinSF2'] = 0
house_test.loc[house_test.BsmtQual=='None', 'BsmtUnfSF'] = 0
house_test.loc[house_test.BsmtQual=='None', 'TotalBsmtSF'] = 0


# In[ ]:


#Let's check again is there any missing values present in data or not
house_train.columns[house_train.isnull().any()]
plt.figure(figsize=(10, 5))
sns.heatmap(house_train.isnull())


# In[ ]:


house_test.loc[house_test.GarageCars.isnull(), 'GarageCars'] = 0
house_test.loc[house_test.GarageArea.isnull(), 'GarageArea'] = 0


# In[ ]:


#Let's check again is there any missing values present in data or not
house_test.columns[house_test.isnull().any()]
plt.figure(figsize=(10, 5))
sns.heatmap(house_test.isnull())


# In[ ]:


total = house_test.isnull().sum().sort_values(ascending=False)
percent = (house_test.isnull().sum()/house_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)


# ### Outliers
# Treat outliers from the dataset. There are some values that are quite different from the rest. It makes sense to delete these variables as the Linear regression methods are very sensitive to outliers.

# ## Data Visualization

# At first I'll look into data correlation, then I'll visualize some data in order to see the impact of certain features.

# In[ ]:


corr = house_train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=1)


# It seems that only several pairs of variables have high correlation. But this chart shows data only for pairs of numerical values. I'll calculate correlation for all variables.

# In[ ]:


threshold = 0.8 # Threshold value.
def correlation():
    for i in house_train.columns:
        for j in house_train.columns[list(house_train.columns).index(i) + 1:]: #Ugly, but works. This way there won't be repetitions.
            if house_train[i].dtype != 'object' and house_train[j].dtype != 'object':
                #pearson is used by default for numerical.
                if abs(pearsonr(house_train[i], house_train[j])[0]) >= threshold:
                    yield (pearsonr(house_train[i], house_train[j])[0], i, j)
            else:
                #spearman works for categorical.
                if abs(spearmanr(house_train[i], house_train[j])[0]) >= threshold:
                    yield (spearmanr(house_train[i], house_train[j])[0], i, j)


# In[ ]:


corr_list = list(correlation())
corr_list


# This is a list of highly correlated features. They aren't surprising and none of them should be removed.

# In[ ]:


#It seems that SalePrice is skewered, so it needs to be transformed.
sns.distplot(house_train['SalePrice'], kde=False, color='c', hist_kws={'alpha': 0.9})


# In[ ]:


#As expected price rises with the quality.
sns.regplot(x='OverallQual', y='SalePrice', data=house_train, color='Orange')


# In[ ]:


#Price also varies depending on neighborhood.
plt.figure(figsize = (12, 6))
sns.boxplot(x='Neighborhood', y='SalePrice',  data=house_train)
xt = plt.xticks(rotation=30)


# In[ ]:


#There are many little houses.
plt.figure(figsize = (12, 6))
sns.countplot(x='HouseStyle', data=house_train)
xt = plt.xticks(rotation=30)


# In[ ]:


#And most of the houses are single-family, so it isn't surprising that most of the them aren't large.
sns.countplot(x='BldgType', data=house_train)
xt = plt.xticks(rotation=30)


# In[ ]:


#Most of fireplaces are of good or average quality. And nearly half of houses don't have fireplaces at all.
pd.crosstab(house_train.Fireplaces, house_train.FireplaceQu)


# In[ ]:


sns.factorplot('HeatingQC', 'SalePrice', hue='CentralAir', data=house_train)
sns.factorplot('Heating', 'SalePrice', hue='CentralAir', data=house_train)


# Houses with central air conditioning cost more. And it is interesting that houses with poor and good heating quality cost nearly the same if they have central air conditioning. Also only houses with gas heating have central air conditioning.

# In[ ]:


#One more interesting point is that while pavement road access is valued more, for alley they quality isn't that important.
fig, ax = plt.subplots(1, 2, figsize = (12, 5))
sns.boxplot(x='Street', y='SalePrice', data=house_train, ax=ax[0])
sns.boxplot(x='Alley', y='SalePrice', data=house_train, ax=ax[1])


# In[ ]:


#We can say that while quality is normally distributed, overall condition of houses is mainly average.
fig, ax = plt.subplots(1, 2, figsize = (12, 5))
sns.countplot(x='OverallCond', data=house_train, ax=ax[0])
sns.countplot(x='OverallQual', data=house_train, ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(2, 3, figsize = (16, 12))
ax[0,0].set_title('Gable')
ax[0,1].set_title('Hip')
ax[0,2].set_title('Gambrel')
ax[1,0].set_title('Mansard')
ax[1,1].set_title('Flat')
ax[1,2].set_title('Shed')
sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Gable'], jitter=True, ax=ax[0,0])
sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Hip'], jitter=True, ax=ax[0,1])
sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Gambrel'], jitter=True, ax=ax[0,2])
sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Mansard'], jitter=True, ax=ax[1,0])
sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Flat'], jitter=True, ax=ax[1,1])
sns.stripplot(x="RoofMatl", y="SalePrice", data=house_train[house_train.RoofStyle == 'Shed'], jitter=True, ax=ax[1,2])


# These graphs show information about roof materials and style. Most houses have Gable and Hip style. And material for most roofs is standard.

# In[ ]:


sns.stripplot(x="GarageQual", y="SalePrice", data=house_train, hue='GarageFinish', jitter=True)


# Most finished garages gave average quality.

# In[ ]:


sns.pointplot(x="PoolArea", y="SalePrice", hue="PoolQC", data=house_train)


# It is worth noting that there are only 7 different pool areas. And while for most of them mean price is ~200000-300000$, pools with area 555 cost very much. Let's see.

# In[ ]:


#There is only one such pool and sale condition for it is 'Abnorml'.
house_train.loc[house_train.PoolArea == 555]


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (12, 5))
sns.stripplot(x="SaleType", y="SalePrice", data=house_train, jitter=True, ax=ax[0])
sns.stripplot(x="SaleCondition", y="SalePrice", data=house_train, jitter=True, ax=ax[1])


# Most of the sold houses are either new or sold under Warranty Deed. And only a little number of houses are sales between family, adjoining land purchases or allocation.

# ## Data preparation

# In[ ]:


#MSSubClass shows codes for the type of dwelling, it is clearly a categorical variable.
house_train['MSSubClass'].unique()


# In[ ]:


house_train['MSSubClass'] = house_train['MSSubClass'].astype(str)
house_test['MSSubClass'] = house_test['MSSubClass'].astype(str)


# Transforming skewered data and dummifying categorical.

# In[ ]:


for col in house_train.columns:
    if house_train[col].dtype != 'object':
        if skew(house_train[col]) > 0.75:
            house_train[col] = np.log1p(house_train[col])
        pass
    else:
        dummies = pd.get_dummies(house_train[col], drop_first=False)
        dummies = dummies.add_prefix("{}_".format(col))
        house_train.drop(col, axis=1, inplace=True)
        house_train = house_train.join(dummies)
        
for col in house_test.columns:
    if house_test[col].dtype != 'object':
        if skew(house_test[col]) > 0.75:
            house_test[col] = np.log1p(house_test[col])
        pass
    else:
        dummies = pd.get_dummies(house_test[col], drop_first=False)
        dummies = dummies.add_prefix("{}_".format(col))
        house_test.drop(col, axis=1, inplace=True)
        house_test = house_test.join(dummies)


# Maybe a good idea would be to create some new features, but I decided to do without it. It is time-consuming and model is good enough without it. Besides, the number of features if quite high already.

# In[ ]:


#This is how the data looks like now.
house_train.head()


# In[ ]:


# Spilit training and testing dataset
X_train = house_train.drop('SalePrice',axis=1)
Y_train = house_train['SalePrice']
X_test  = house_test


# ## Training and Testing the Model

# In[ ]:


#Function to measure accuracy.
def rmlse(val, target):
    return np.sqrt(np.sum(((np.log1p(val) - np.log1p(np.expm1(target)))**2) / len(target)))


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, Y_train, test_size=0.30)


# In[ ]:


ridge = Ridge(alpha=10, solver='auto').fit(Xtrain, ytrain)
val_ridge = np.expm1(ridge.predict(Xtest))
rmlse(val_ridge, ytest)


# In[ ]:


ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_cv.fit(Xtrain, ytrain)
val_ridge_cv = np.expm1(ridge_cv.predict(Xtest))
rmlse(val_ridge_cv, ytest)


# In[ ]:


las = linear_model.Lasso(alpha=0.0005).fit(Xtrain, ytrain)
las_ridge = np.expm1(las.predict(Xtest))
rmlse(las_ridge, ytest)


# In[ ]:


las_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
las_cv.fit(Xtrain, ytrain)
val_las_cv = np.expm1(las_cv.predict(Xtest))
rmlse(val_las_cv, ytest)


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2) #the params were tuned using xgb.cv
model_xgb.fit(Xtrain, ytrain)
xgb_preds = np.expm1(model_xgb.predict(Xtest))
rmlse(xgb_preds, ytest)


# In[ ]:


forest = RandomForestRegressor(min_samples_split =5,
                                min_weight_fraction_leaf = 0.0,
                                max_leaf_nodes = None,
                                max_depth = None,
                                n_estimators = 300,
                                max_features = 'auto')

forest.fit(Xtrain, ytrain)
Y_pred_RF = np.expm1(forest.predict(Xtest))
rmlse(Y_pred_RF, ytest)


# So linear models perform better than the others. And lasso is the best.
# 
# Lasso model has one nice feature - it provides feature selection, as it assignes zero weights to the least important variables.

# In[ ]:


coef = pd.Series(las_cv.coef_, index = X_train.columns)
v = coef.loc[las_cv.coef_ != 0].count() 
print('So we have ' + str(v) + ' variables')


# In[ ]:


#Basically I sort features by weights and take variables with max weights.
indices = np.argsort(abs(las_cv.coef_))[::-1][0:v]


# In[ ]:


#Features to be used. I do this because I want to see how good will other models perform with these features.
features = X_train.columns[indices]
for i in features:
    if i not in X_test.columns:
        print(i)


# RoofMatl_ClyTile
# 
# There is only one selected feature which isn't in test data. I'll simply add this column with zero values.[](http://)

# In[ ]:


X_test['RoofMatl_ClyTile'] = 0


# In[ ]:


X = X_train[features]
Xt = X_test[features]


# In[ ]:


Xtrain1, Xtest1, ytrain1, ytest1 = train_test_split(X, Y_train, test_size=0.33)


# Let's see whether something changed.

# In[ ]:


ridge = Ridge(alpha=5, solver='svd').fit(Xtrain1, ytrain1)
val_ridge = np.expm1(ridge.predict(Xtest1))
rmlse(val_ridge, ytest1)


# In[ ]:


las_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10)).fit(Xtrain1, ytrain1)
val_las = np.expm1(las_cv.predict(Xtest1))
rmlse(val_las, ytest1)


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2) #the params were tuned using xgb.cv
model_xgb.fit(Xtrain1, ytrain1)
xgb_preds = np.expm1(model_xgb.predict(Xtest1))
rmlse(xgb_preds, ytest1)


# In[ ]:


forest = RandomForestRegressor(min_samples_split =5,
                                min_weight_fraction_leaf = 0.0,
                                max_leaf_nodes = None,
                                max_depth = 100,
                                n_estimators = 300,
                                max_features = None)

forest.fit(Xtrain1, ytrain1)
Y_pred_RF = np.expm1(forest.predict(Xtest1))
rmlse(Y_pred_RF, ytest1)


# In[ ]:


las_cv1 = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
las_cv1.fit(X, Y_train)
lasso_preds = np.expm1(las_cv1.predict(Xt))


# In[ ]:


#I added XGBoost as it usually improves the predictions.
model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.1)
model_xgb.fit(X, Y_train)
xgb_preds = np.expm1(model_xgb.predict(Xt))


# In[ ]:


preds = 0.7 * lasso_preds + 0.3 * xgb_preds


# In[ ]:


submission = pd.DataFrame({
        'Id': house_test['Id'].astype(int),
        'SalePrice': preds
    })
submission.to_csv('home.csv', index=False)


# But the result wasn't very good. I thought for some time and then decided that the problem could lie in feature selection - maybe I selected bad features or Maybe random seed gave bad results. I decided to try selecting features based on full training dataset (not just on part of the data).

# In[ ]:


model_lasso = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 100))
model_lasso.fit(X_train, Y_train)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
v1 = coef.loc[model_lasso.coef_ != 0].count()
print('So we have ' + str(v1) + ' variables')


# In[ ]:


indices = np.argsort(abs(model_lasso.coef_))[::-1][0:v1]
features_f=X_train.columns[indices]


# In[ ]:


print('Features in full, but not in val:')
for i in features_f:
    if i not in features:
        print(i)
print('\n' + 'Features in val, but not in full:')
for i in features:
    if i not in features_f:
        print(i)


# A lot of difference between the selected features. I suppose that the reason for this is that there was too little data relatively to the number of features in the first case. So I'll use the features obtain with the analysis of the whole train dataset.

# In[ ]:


for i in features_f:
    if i not in X_test.columns:
        X_test[i] = 0
        print(i)
X = X_train[features_f]
Xt = X_test[features_f]


# Now all necessary features are present in both train and test.

# In[ ]:


model_lasso = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
model_lasso.fit(X, Y_train)
lasso_preds = np.expm1(model_lasso.predict(Xt))


# 

# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X, Y_train)
xgb_preds = np.expm1(model_xgb.predict(Xt))


# ### Submission

# In[ ]:


solution = pd.DataFrame({"id":house_test.Id, "SalePrice":0.7*lasso_preds + 0.3*xgb_preds})
solution.to_csv("House_price.csv", index = False)


# ![](http://investmarbellaproperty.com/wp-content/uploads/2019/04/Torremolinos-1024x640.jpg)

# I hope this kernal is useful to you to learn exploratory data analysis and regression problem.
# 
# If find this notebook help you to learn, Please Upvote.
# 
# Thank You!!
