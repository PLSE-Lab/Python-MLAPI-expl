#!/usr/bin/env python
# coding: utf-8

# ![](http://)

# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. This notebook focuses on detail data exploration and predictions.

# # Learning Objectives
# 
# * Importing  and reading data
# * Checking data informations: head check, shape check, duplicate check,outliers and missing value check
# * Missing values and outliers handling
# * Visualization of categorical and numerical features with targets
# * Finding most correlated features
# * Log transformation 
# * Dropping unnessary features and creating new features
# * Converting ordinal and nominal categorical data(Label and One hot encoding)
# * Data preprocessing for predictive modeling
# * Ensamble modeling
# * Using data analysis tools: numpy, pandas, seaborn,stats, matplotlib

# Importing Basic Data Analysis Tools

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import os #The OS module in Python provides a way of using operating system dependent functionality
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))


# * First things first, load the data files. There are total of 4 different files. They are sample submission file, train data, test data and data descriptions. We will be working on train and test datasets. So let's load them.

# In[ ]:


# Loading  test and train datasets
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
df_train.head()


# Before moving to further analysis, i am going to extract the id columns from both train and test files. Because id column does not make any contribution to predictions. Also submission file requires id columns.

# In[ ]:


# Separate the id columns from both test and train data as id columns
#is needed for submission and also id column does not make any contribution to prediciton

id_train= df_train['Id']
Id_test = df_test['Id']

df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)


# Basic information of the data

# In[ ]:


# Function to print the basic information of the data
def data_info(df):

    print('Shape of the data: ', df.shape)
    
    print('------------########################------------------')
    print('                                                     ')
    print('Information of the data:')
    print(' ', df.info())
    
    print('------------########################------------------')
    print('                                                     ')
    print('Check the duplication of the data:', df.duplicated().sum())


# Train Data

# In[ ]:


data_info(df_train)


# Train data contains 1460 entries and 80 columns including target. Of them 43 are categorical data and 37 columns are numerical columns. There is no duplicate entries

# Sattistical Summary of the Data

# In[ ]:


# Function to find out the Statistical susmmary 
def summary(df):
    print('\n Statistical Summary of Numberical data:\n', df.describe(include=np.number))
    print('------------########################------------------')
    print('\n Statistical Summary of categorical data:\n',df.describe(include='O'))
    
summary(df_train)


# Analysing target
# Ploting target versus features would give an idea how features are related with target.

# In[ ]:


# Boxplot for target
plt.figure(figsize=(12,8))
sns.boxplot(df_train['SalePrice'])


# Above boxplot clearly indicates that there is outliers. It's not safe to remove outliers without thorough investigation. But two data points seemed to be very far from average, we can safely remove them. So sales price above 700K are removed

# In[ ]:


# Remove outliers from target variables
df_train=df_train[df_train['SalePrice']<700000]
df_train.head()


# Now let's look at the distribution of the target

# In[ ]:


# Distribution plot
plt.figure(figsize=(12,8))
sns.distplot(df_train['SalePrice'])


# The distribution is right skewed. ML models work best on normal distribution! Let's compare this with normal distribution and calculate the probability parameters like mu and sigma

# In[ ]:


# Distribution plot
plt.figure(figsize=(12,8))
sns.distplot(df_train['SalePrice'] , fit=norm);

# Probability parameter
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# Since the distribution is not normal distribution, Log transformation will make it normal.

# In[ ]:


#Log tranformation of target column
plt.figure(figsize=(12,8))
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

#Plot the new distriution
sns.distplot(df_train['SalePrice'] , fit=norm);

# probability parameter for normal distribution
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# Now let's check the lower and upper band of the outliers.

# In[ ]:


# Outliers Check
def outlier(df):
    stat=df.describe()
    IQR=stat['75%']-stat['25%']
    upper=stat['75%']+1.5*IQR
    lower=stat['25%']-1.5*IQR
    print('The upper and lower bounds for outliers are {} and {}'.format(upper,lower))


# In[ ]:


outlier(df_train['SalePrice'])


# Further analysis is required to find the causes of outliers. Visualization of numerical and categorical features with target can reveal more information.

# In[ ]:


# Let's separate the numerical and categorical columns
numerical_col=df_train.select_dtypes(include=[np.number])
categorical_col=df_train.select_dtypes(include=[np.object])
num_var=numerical_col.columns.tolist()
cat_var=categorical_col.columns.tolist()


# In[ ]:


# Function to plot target vs categorical data
def cat_plot(df):
    for col in cat_var:
        f, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=col,y='SalePrice', data=df)
        plt.xlabel(col)
        plt.title('{}'.format(col))

cat_plot(df_train)


# We can see that neighborhood, Exterior 1st, Exterior2d and SaleType have great influence on SalePrice. They are highly correlated with SalePricce than other features.

# In[ ]:


# Function to plot target vs numerical data
def num_plot(df):
    for col in num_var:
        f, ax = plt.subplots(figsize=(12, 6))
        plt.scatter(x=col,y='SalePrice', data=df)
        plt.xlabel(col)
        plt.ylabel("SalePrice")
        plt.title('{}'.format(col))


# In[ ]:


num_plot(df_train)


# TotalBsmtSF, 1stFlrSF, 2ndFlrSF and GrLivArea are linearly correlated with SalePrice. Some features reveal suspicious points. They are LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, TotalBsmtSF,1stFlrSF GrLivArea, EnclosedPorch, MiscVal. Removing Outliers is not alawys safe. We might miss important informations.  GarageArea and OpenPorchSF shows kind of similar relationship with SalePrice.

# From looking at the plot we can cleary see that there are some point which are highly suspicious. Let's remove the highly suspicous outliers

# In[ ]:


# Removing suspicious outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)
df_train=df_train.drop(df_train[(df_train['LotFrontage']>250) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)
df_train=df_train.drop(df_train[(df_train['BsmtFinSF1']>1400) & (df_train['SalePrice']<400000)].index).reset_index(drop=True)
df_train=df_train.drop(df_train[(df_train['TotalBsmtSF']>5000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)
df_train=df_train.drop(df_train[(df_train['1stFlrSF']>4000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)


# In[ ]:


#Categorical variables after removing the outliers
new_cat=['GrLivArea','LotFrontage','BsmtFinSF1','TotalBsmtSF','1stFlrSF']


# In[ ]:


# Plotting after removing outliers
for col in new_cat:
        f, ax = plt.subplots(figsize=(12, 6))
        plt.scatter(x=col,y='SalePrice', data=df_train)


# Now let's merge the test and train data so that we can process togather. After processing we will separate train and test data.

# In[ ]:


# merging the data
train_len = len(df_train) # created length of the train data so that after EDA is done we can seperate the train and test data
data= pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)
data.head()


# Missing Value Check

# In[ ]:


# A function for calculating the missing data
def missing_data(df):
    tot_missing=df.isnull().sum().sort_values(ascending=False)
    Percentage=tot_missing/len(df)*100
    missing_data=pd.DataFrame({'Missing Percentage': Percentage})
    
    return missing_data.head(36)

missing_data(data)


# In[ ]:


# missing value in test dataset
missing_data(df_test)


# Let's analyse the missing value columns

# Missing Values on train data
# * 'PoolQC'-NA means no Pool
# * 'MiscFeature'- NA means None
# * 'Alley'- NA means No alley access
# * 'Fence'- NA	means No Fence
# * 'FireplaceQu'-NA Means No Fireplace
# * 'LotFrontage'-Linear feet of street connected to property
# * 'GarageCond'-NA	means No Garage
# * 'GarageType'-NA	means No Garage
# * 'GarageYrBlt'-Year garage was built
# * 'GarageFinish'-NA means No Garage
# * 'GarageQual'- NA meand No Garage
# * 'BsmtExposure'-NA means No Basement
# * 'BsmtFinType2'-NA meand No Basement
# * 'BsmtFinType1'-NA means No Basement
# * 'BsmtCond'-NA mean No Basement
# * 'BsmtQual'-  NA	meand No Basement
# * 'MasVnrArea-Masonry veneer area in square feet
# * 'MasVnrType'-None means None
# * 'Electrical'-Electrical system
# 
# Missing value on test data
# * SaleType -Different types of services offered
# * BsmtFinSF1-Type 1 finished square feet
# * BsmtFinSF2-Type 2 finished square feet
# * Exterior1st-Exterior covering on house
# * Exterior2nd-Exterior covering on house
# * MSZoning-Identifies the general zoning classification of the sale
# * BsmtFullBath-Basement full bathrooms
# * BsmtHalfBath-Basement half bathrooms
# *  SalePrice- True sale price
# *  GarageCars-Size of garage in car capacity
# *  Functional-Home functionality (Assume typical unless deductions are warranted)
# *  KitchenQual-Kitchen quality
# *  Utilities-Type of utilities available
# * TotalBsmtSF-Total square feet of basement area
# * GarageArea- Size of garage in square feet
# * BsmtUnfSF

# The above mentioned columns have missing value. There are different ways of imputing missing values. From our observation we can replace categorical features with None and numerical features with 0.

# In[ ]:


# Features with missing value
miss_col1=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageType', 'GarageYrBlt',
           'GarageFinish', 'GarageQual', 'BsmtExposure','BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 
           'MasVnrArea', 'MasVnrType','SaleType','MSZoning','Utilities','Functional','Exterior1st','Exterior2nd',
           'BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','GarageArea','KitchenQual','GarageCars','BsmtFullBath',
           'BsmtHalfBath','BsmtUnfSF']
# Imputing missing value
for col in miss_col1:
    if data[col].dtype=='O':
        data[col]=data[col].fillna("None")
    else:
        data[col]=data[col].fillna(0)


# 'LotFrontage'-Linear feet of street connected to property. Each house is connected to the street, so most likely the it has similar area to neighbourhood house. 

# In[ ]:


# Imputing missing value with neighborhood value
data['LotFrontage']=data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[ ]:


# Imputing missing value with mode
data['Electrical']=data['Electrical'].fillna(data['Electrical'].mode()[0])


# In[ ]:


missing_data(data)


# Let's see the correlation plot

# In[ ]:


corr= data.corr()
f, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(corr, vmax=.6, square=True)


# Looking at the correlation plot, GarageCars and GarageArea, and TotalBsmtSF have similar relationship with Saleprice. let's do Zoomed heatmap plot to see relationship more closely
# 

# In[ ]:


k = 20 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True,linewidths=0.004, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# By closely looking at the data, it is obvious that OverallQual is the highly correlated feautre followed by GrLivArea, GarageCars, Garage Area and TotalBsmtSF. 1stFlrSF, FullBath and YearBuilt have same correlation with SalePrice. Least correlated feature is BsmtFinSF1.

# Creating new features.
# Since GrLivArea and Garage Area have high influence on Saleprice we can create a new features call total area.

# In[ ]:


# Total area in units of square feet
data['TotSF']=data['TotalBsmtSF']+data['1stFlrSF']+data['2ndFlrSF']
data['TotArea']=data['GarageArea']+data['GrLivArea']


# In[ ]:


plt.scatter(x='TotArea',y='SalePrice', data=data)


# Seems NewTotal TotalArea has very strong linear relationship with SalePrice

# Some of the columns are numerical but they are actually categorical. They are 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold'. These columns are converted into categorical columns

# In[ ]:


cols=['MSSubClass','OverallCond','YrSold','MoSold']

for col in cols:
    data[col] = data[col].apply(str)


# # Converting categorical data to numerical data
# Some features are ordinal and some of them are nominal. For ordinal data let's do the label encoding and for nominal data i will do one hot encoding.

# In[ ]:


categorical_col=data.select_dtypes(include=[np.object])
new_catcol=categorical_col.columns
new_catcol


# In[ ]:


ordinal_cat=['OverallCond','KitchenQual','YrSold','MoSold','Fence','PoolQC','FireplaceQu','GarageQual', 
             'GarageCond','LotShape','LandSlope','HouseStyle','ExterQual','ExterCond','BsmtQual', 
             'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2','HeatingQC','KitchenQual','CentralAir',
             'MSSubClass']

# label Encoding for ordinal data
from sklearn.preprocessing import LabelEncoder
label_encode=LabelEncoder()

for col in ordinal_cat:
    data[col]=label_encode.fit_transform(data[col])


# In[ ]:


data.select_dtypes(include=[np.object]).head()


# In[ ]:


# One hot encoding for nominal data
data=pd.get_dummies(data)


# Separating target and features

# In[ ]:


df_target=data['SalePrice']
df_features=data.drop(columns=['SalePrice'])


# Data Preparation

# In[ ]:


X_train=df_features[:train_len]
Y_train=df_target[:train_len]
X_test=df_features[train_len:]


# # Model Building

# There are lots of models out there. Here i will apply couple of regression model based on data analysis. Data has lots of outliers so i chose models that can handle outliers. First let's import the libraries for model building.

# In[ ]:


# Import MLlibraries
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.utils import shuffle

from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge, LassoLarsIC


# Root mean squared Error matrix is used to evaluate the model. So i created a function to calculate the RMSE

# In[ ]:


# Function to calculate RMSE
n_folds = 5
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Let's try the following model

# Lasso

# In[ ]:


lasso=Lasso()
rmse_cv(lasso).mean()


# Use GridSearch method to find the best parameters

# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]}
grid_search_cv = GridSearchCV(Lasso(random_state=42), params, n_jobs=-1)
grid_search_cv.fit(X_train, Y_train)

print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_score_)


# RandomForest

# In[ ]:


Random=RandomForestRegressor()
rmse_cv(Random).mean()


# In[ ]:


params = {'n_estimators': list(range(50, 200, 25)), 'max_features': ['auto', 'sqrt', 'log2'], 
         'min_samples_leaf': list(range(50, 200, 50))}

grid_search_cv = GridSearchCV(RandomForestRegressor(random_state=42), params, n_jobs=-1)
grid_search_cv.fit(X_train, Y_train)

print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_score_)


# Elastic Net

# In[ ]:


Enet=ElasticNet()
rmse_cv(Enet).mean()


# In[ ]:


params = {'alpha': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]}

grid_search_cv = GridSearchCV(ElasticNet(random_state=42), params, n_jobs=-1)
grid_search_cv.fit(X_train, Y_train)

print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_score_)


# KernelRidge

# In[ ]:


KR=KernelRidge()
rmse_cv(KR).mean()


# In[ ]:


params = {'alpha': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006]}

grid_search_cv = GridSearchCV(KernelRidge(), params, n_jobs=-1)
grid_search_cv.fit(X_train, Y_train)

print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_score_)


# GradientBoosting

# In[ ]:


GBoost = GradientBoostingRegressor()
rmse_cv(GBoost).mean()


# In[ ]:


params = {'n_estimators': [1000,2000,3000,4000,5000,6000]}

grid_search_cv = GridSearchCV(GradientBoostingRegressor(), params, n_jobs=-1)
grid_search_cv.fit(X_train, Y_train)

print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_score_)


# From the above model, we can see that the mean RMSE for lasso, randomforest, elastic Net, Kernel Ridge and Gradient Boosting with defualt parameters are 0.1692, 0.1470, 0.1614, 0.1206 and 0.1222 respectively. Using best parameters we can improve the score.

# In[ ]:


# Models  with best paramenters
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0004, random_state=42))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0006, l1_ratio=.5, random_state=42))
KRR = KernelRidge(alpha=0.0001, kernel='linear', degree=3, coef0=1.0)
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

RanForest=RandomForestRegressor(min_samples_leaf=50, min_samples_split=2, 
                                n_estimators=150, random_state=42)


# RNMSE with best parameters

# In[ ]:


Ml_models=[RanForest,lasso,ENet,KRR,GBoost]
def rmse_score(models):
    for model in models:
        print("RMSE and STD for {} are {:4f} and  {:4f} respectively.".format(model,rmse_cv(model).mean(),rmse_cv(model).std()))
        #print("RMSE and STD for lasso are {:4f} and  {:4f} respectively.".format(rmse_cv(lasso).mean(),rmse_cv(lasso).std()))


# In[ ]:


rmse_score(Ml_models)


# Using the best parameters, RMSE for randomforst regressor, lasso, elastic Net , kernel Ridge and Gradient boosting are 0.174109,0.113024, 0.113532, 0.129478 and 0.117206 respectively

# Fit models with best best parameters

# In[ ]:


LassoFit= lasso.fit(X_train,Y_train)
ENetFit = ENet.fit(X_train,Y_train)
KRRFit = KRR.fit(X_train,Y_train)
GBoostFit = GBoost.fit(X_train,Y_train)
RanForestFit=RanForest.fit(X_train,Y_train)


# In[ ]:


Final_score= (np.expm1(LassoFit.predict(X_test)) + 
              np.expm1(ENetFit.predict(X_test)) + np.expm1(KRRFit.predict(X_test)) 
              + np.expm1(GBoostFit.predict(X_test))+ np.expm1(RanForestFit.predict(X_test))) / 5
Final_score


# In[ ]:


test_prediction = pd.Series(Final_score, name="SalePrice")


# In[ ]:


# Making Submission file
Final_sub= pd.concat([Id_test,test_prediction],axis=1)
Final_sub.to_csv("submission_emrul.csv", index=False)


# In[ ]:


Final_sub.head()


# 
# Thanks a lot for browsing this NoteBook. Any comment or suggestions would be highly appriciated. If you find this notebook is helpfull, please upvote.

# https://www.kaggle.com/agodwinp/stacking-house-prices-walkthrough-to-top-5/notebook
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# https://www.kaggle.com/erick5/predicting-house-prices-with-machine-learning
# https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn
# https://www.kaggle.com/vjgupta/reach-top-10-with-simple-model-on-housing-prices
# 
