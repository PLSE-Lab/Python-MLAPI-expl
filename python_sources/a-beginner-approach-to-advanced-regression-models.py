#!/usr/bin/env python
# coding: utf-8

# <h1>Introduction</h1>

# This notebook is a very basic and simple approach to this regression problem. This is a perfect starter competition for new comers in this field. I have used elements from the excellent kernel provided by Serigne you can find it <a href="https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard">here</a>.
# 
# Please feel free to leave any comments that will help me to further improve this kernel and do upvote if you like it.
# 
# This notebook is divided into six major parts:
# <ol>
#     <li>Introduction</li>
#     <li>Competition Description</li>
#     <li>Data Description</li>
#     <li>Exploratory Data Analyis or EDA (in short)</li>
#     <li>Data Pre-Processing</li>
#     <li>Modeling</li>
# </ol>
# Following the famous data science mantra we will spending the majority of our time in EDA and preprocessing compared to Modeling in a 80:20 ratio.
# For the modeling part I have used Random Forest Regressor,XGBoost Regressor and LightGBM Regressor. You can find the documentation for XGBoost <a href="https://xgboost.readthedocs.io/en/latest/">here</a> and the documentation for LightGBM <a href="https://lightgbm.readthedocs.io/en/latest/">here</a>.

# <h2>Competition Description</h2>

# This competition contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa through these variables we are challenged to predict the final price of each home.

# <h2>Data Description</h2>

# The files given are:
# 
# 1. train.csv: This is the dataset that we are gonna use to train our model to give predictions. SalePrice is theproperty's sale price in dollars. This is the target variable that we are trying to predict.
# 2. test.csv:The test set will be used to see how well our model performs on unseen data. For the test set, we do not provide the Target variable i.e, SalePrice. It is our job to predict these outcomes. For each passenger in the house, we use the model that we trained to predict the price of the house.
# 3. data_description.txt - This gives us the full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here.
# 4. sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms. This serves as an example of how our submission should look like.

# <h2>Importing the Libraries</h2>
# 
# This is where the actual fun begins. We start off by importing all the libraries that we will need later on. We will be using Numpy and pandas for data analysis and matplotlib (Matlab for python), seaborn for data visualisation. Below I have given the steps to load the data onto varibles.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This is how we assign the datasets to variables in python using pandas.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# <h2>Exploratory Data Analysis</h2>

# We will use the .head() function to display the first five columns of the dataset to get a feel of the dataset.

# In[ ]:


train.head()


# In[ ]:


#info gives us information about index and column data types.
train.info()


# In[ ]:


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop we the 'Id' column since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[ ]:


#This gives us the statistical summary of the dataset
train['SalePrice'].describe()


# A plot to visualise the Target Distribution. Since this plot is right skewed we will later on transform this into a normal distribution in the preprocessing section.

# In[ ]:


sns.set_style("whitegrid")
sns.distplot(train['SalePrice'])


# Here we are plotting the Target varible against the GrLivArea which is Above grade (ground) living area in square feet. From this we can understand that "GrLivArea" and "SalePrice" has a linear relationship.

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# This is a plot of the Target variable against the TotalBsmtSF which is Total square feet of basement area. From the plot we can tell that 'SalePrice' and 'TotalBsmtSF' have a strong linear or exponential relationship.

# In[ ]:


data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))


# This is the plot of the Target Variable against the 'OverallQual' variable which is the Overall material and finish quality. From this we can deduce that as the Overall quality increases so does the house price.

# In[ ]:


data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


# This plot is of the 'YearBuilt' feature against the Target variable. The 'YearBuilt' feature contains the original construction date. It isn't that clear but we can say that 'SalePrice' is more to spend money in new stuff than old.

# In[ ]:


data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


# Let us plot the correlation matrix using heatmap to better understand the data.

# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# From the above heatmap we can find that some predictors are strongly correlated to each other thus causing multicollinearity. Some of them are 'TotalBsmtSF' and '1stFlrSF' and the Garage variables. Now let us plot a correlation matrix against 'SalePrice'. 

# In[ ]:


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# From this we can tell that "OverallQual",'GrLivArea','TotalBsmtSF' are strongly correlated to the target variable.<br> <br>
# 'GarageCars' and 'GarageArea' are strongly correlated to the Target variable too but since we know that the number of cars that fit in a garage is proportional to the 'GarageArea' we can just use one of these variables instead of all the Garage variables.<br><br>
# 'ToalBsmtSF' and '1stFloor' are strongly correlated to each other so we can just use only one of them or combine them. The same thing can be done to 'TotRmsAbvGrd' and 'GrLivArea' 

# <h1>Preprocessing</h1>

# <h3>Outliers</h3>

# From the below plot we can see two huge outliers GrLivArea that are of a low price present in the bottom right corner. Since these are huge outliers it is safe to delete them as it negatively affects our model.

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# As already mentioned in the EDA section when we explored the Target Variable we need to transform this variable to make it more normally distributed. This is because linear models work well with normally distributed data.

# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

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


# We are gonna concatenate the train and test data so that it will be easier to make adjustments to the combined data than to individually do it for train and test dataset.

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# <h3>Missing Values</h3>
# <br>
# Let us look at the percentage of the missing values of the dataset.

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Time to impute the missing values onto the features.

# <ul>
#     <li>We will impute the Nan or null values of 'PoolQC' with None as null values signify that the majority of the houses don't have a Pool.</li>
# </ul>

# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")


# <ul>
#     <li>We will impute the Nan or null values of 'MiscFeature' with None as null values signify that the house does not contain miscellaneous Features.</li>
# </ul>

# In[ ]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")


# <ul>
#     <li>We will impute the Nan or null values of 'Alley' which refers to type of Alley access with None as null values signify that the house does not contain any Alley access.</li>
# </ul>

# In[ ]:


all_data["Alley"] = all_data["Alley"].fillna("None")


# <ul>
#     <li>We will impute the Nan or null values of 'Fence' with None as null values signify that the house does not contain any Fences.</li>
# </ul>

# In[ ]:


all_data["Fence"] = all_data["Fence"].fillna("None")


# <ul>
#     <li>We will impute the Nan or null values of 'FireplaceQu' with None as null values signify that the house does not contain any fireplace.</li>
# </ul>

# In[ ]:


all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")


# <ul>
#     <li>We will impute the Nan or null values of 'LotFrontage' with median values taking into consideration that the area of each street connected to the house property is most likely similar to other houses of the Neighborhood.</li>
# </ul>

# In[ ]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# <ul><li>We replace the 'GarageType', 'GarageFinish', 'GarageQual' and 'GarageCond' missing values into None signifying that those homes most likely don't have any garage for vehicles.</li></ul>

# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')


# <ul><li>We replace the null values of 'GarageYrBlt', 'GarageArea', 'GarageCars' as no garage equals no cars.</li></ul>

# In[ ]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# <ul><li>The missing values of 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath' are filled with zero because the null values most likely signifies no basement.</li></ul>

# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# <ul><li>The missing values'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2 are filled with Nonne as NaN values signifies no basement.</li></ul>

# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# <ul><li>The NA values of "MasVnrType" and "MasVnrArea" refer to no masonry veneer for these houses. Therefore the Area can be imputed with zero and the type can be imputed with None.</li></ul>

# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# <ul><li>The 'MSZoning' refers to the general zoning classification. We fill the NaN values with 'RL' which is the most occurring value.</li></ul>

# In[ ]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# <ul><li>Most of the values of this feature are "AllPub" except for one "NoSeWa" and 2 NA. Since the house with "NoSewa' is present in the training set, this feature is not helpful in predictive modeling so we drop it. </li></ul>

# In[ ]:


all_data = all_data.drop(['Utilities'], axis=1)


# <ul><li>The NA values means typical which is mentioned in the data description.</li></ul>

# In[ ]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# <ul><li>There is only one NA value present here in "Electrical" so we impute it 'SBrkr' since it is the most occurring value.</li></ul>

# In[ ]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# <ul><li>There is only one NA value present here in "KitchenQual" so we impute it 'TA' since it is the most occurring value.</li></ul>

# In[ ]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# <ul><li>There is only one NA value present in both Exterior1st and Exterior2nd so we fill it with the most frequent string.</li></ul>

# In[ ]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# <ul><li>Fill it again with the most frequent which is "WD" in this case.</li></ul>

# In[ ]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# <ul><li>The NA values here refers to No Building class so we fill it with None.</li></ul>

# In[ ]:


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# In[ ]:


all_data.isnull().sum().max()


# With this we no longer have any missing values. 

# <h3>Transforming Categorical Variables</h3>

# We now transform a few numerical variables that are categorical then we perform label encoding onto them.

# In[ ]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# So we create a new feature that plays an important role in predicting house prices which is the sum of total area of basement, first and second floors of each house.

# In[ ]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# We find the skewed features and then perform Box Cox Transformation of highly skewed features.

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# In[ ]:


#Getting dummy categorical features.
all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[ ]:


#Splitting the data back into test and train
train = all_data[:ntrain]
test = all_data[ntrain:]


# <h1>Modeling</h1>

# <h4>Now let us import all the relevant libraries.</h4>

# In[ ]:


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# For cross validation we the cross_val_score function of sklearn.

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# <h3>Random Forest Regressor,GradientBoosting Regressor, XGBoostRegressor and LightGBM Regressor </h3>

# In[ ]:


rfc=RandomForestRegressor(n_estimators=1000)


# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[ ]:


rfc.fit(train,y_train)
rfc_train_pred = rfc.predict(train.values)
rfc_pred = np.expm1(rfc.predict(test.values))


# In[ ]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))


# In[ ]:


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))


# In[ ]:


GBoost.fit(train,y_train)
GBoost_train_pred = GBoost.predict(train)
GB_pred = np.expm1(GBoost.predict(test.values))


# 
# <h2>Submission</h2>

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = GB_pred
sub.to_csv('submission.csv',index=False)

