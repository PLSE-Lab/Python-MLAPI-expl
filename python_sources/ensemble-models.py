#!/usr/bin/env python
# coding: utf-8

# This is my First Kaggle Notebook that too in Python, I've been exposed to R more than Python, but ive taken a jump into the python world.
# 
# I went through some of the awesome notebooks, which gave me a lot of confidence to try my hand at this:
# 1. [Comprehensive Data Exploration in Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/comments)
# 2. [Stacked Regressions to Predict Housing Prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# 3. [Regularised Linear models](https://www.kaggle.com/apapiu/regularized-linear-models/)
# 4. https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition 
# 
# So here goes, scored, top 39% without any featue engineering, is suppose not that bad i guess for the first time
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from matplotlib.pyplot import figure
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
#figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split

#Import Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

import warnings
# Ignore useless warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Avoid runtime error messages
pd.set_option('display.float_format', lambda x:'%f'%x)

np.random.seed(87)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Reading CSV Files**

# In[ ]:


file_path = '../input/train.csv'
train = pd.read_csv(file_path)
print("Training Dataset Dimensions:" + str(train.shape))
file_path = '../input/test.csv'
test = pd.read_csv(file_path)
print("Test Dataset Dimensions:" + str(test.shape))


# In[ ]:


#sample datapoints
train.sample(5)


# In[ ]:


test.sample(5)


# I think we can safely drop the 'ID' column, it is unnecessary for our model

# In[ ]:


#saving the ID column
train_ID = train['Id']
test_ID = test['Id']

#Droppping the column
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
print("Dropped ID Column")
print("train dataset",train.shape)


# In[ ]:


#Check our training label against all numeric features
#Seleting all Numeric Features
numeric_feats = train.dtypes[train.dtypes != "object"].index
numeric_feats
print(np.ceil(len(numeric_feats)/3))


# In[ ]:


#plotting all numeric features against SalePrice
fig, ax = plt.subplots(figsize=(21,100))
sns.set(style='darkgrid')
length = len(numeric_feats)
for i,feature in enumerate(numeric_feats,1):
    plt.subplot(np.ceil(length/3), 3, i) #nrows,ncols,index
    sns.scatterplot(x = feature, y = 'SalePrice',alpha=0.7,data=train,hue='SalePrice',palette ='nipy_spectral_r',linewidth=0.5, edgecolor='white')
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel(feature, fontsize=13)
plt.show()
    


# In[ ]:


#After looking at the Scatterplot, Capping the Outliers from training set as follows:
#Lotfrontage > 300, GrLivArea>4000, SalePrice > 600000, LotArea >100000, YearBuilt <1880
#MasVnrArea>1200, BsmtFinSF1>2000,BsmtFinSF2>1400, TotalBsmtSF > 4000,1stFlrSF>4000, GarageArea>1200,WoodDeckSf>800, OpenPorchSF>500,EnclosedPorch>400
#ScreenPorch>400, PoolArea>600,MiscVal>8000

#there are only 4 values above 4000 in GrLivArea, cap them to 4000 as they might skew our model
#train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.loc[(train['GrLivArea']>4000),'GrLivArea'] = 4000
train.loc[(train['SalePrice']>600000),'SalePrice'] = 600000
train.loc[(train['LotArea']>100000),'LotArea'] = 100000
train.loc[(train['YearBuilt']<1880),'YearBuilt'] = 1880
train.loc[(train['MasVnrArea']>1200),'MasVnrArea'] = 1200
train.loc[(train['BsmtFinSF1']>2000),'BsmtFinSF1'] = 2000
train.loc[(train['BsmtFinSF2']>1400),'BsmtFinSF2'] = 1400
train.loc[(train['TotalBsmtSF']>4000),'TotalBsmtSF'] = 4000
train.loc[(train['1stFlrSF']>3500),'1stFlrSF'] = 3500
train.loc[(train['GarageArea']>1200),'GarageArea'] = 1200
train.loc[(train['WoodDeckSF']>800),'WoodDeckSF'] = 800
train.loc[(train['OpenPorchSF']>500),'OpenPorchSF'] = 500
train.loc[(train['EnclosedPorch']>400),'EnclosedPorch'] = 400
train.loc[(train['ScreenPorch']>400),'ScreenPorch'] = 400
train.loc[(train['PoolArea']>600),'PoolArea'] = 600
train.loc[(train['MiscVal']>8000),'MiscVal'] = 8000


# In[ ]:


#Lets check all out features again
fig, ax = plt.subplots(figsize=(21,100))
sns.set(style='darkgrid')
length = len(numeric_feats)
for i,feature in enumerate(numeric_feats,1):
    plt.subplot(np.ceil(length/3), 3, i) #nrows,ncols,index
    sns.scatterplot(x = feature, y = 'SalePrice',alpha=0.7,data=train,hue='SalePrice',palette ='nipy_spectral',linewidth=0.5, edgecolor='white')
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel(feature, fontsize=13)
plt.show()
    


# In[ ]:


#save the training label 'SalePrice' seperately
training_label = train['SalePrice'].reset_index(drop=True)


# In[ ]:


train_features = train.drop(['SalePrice'],axis=1)
#creating a full dataset by concatinating both train and test dataset
full_ds = pd.concat([train_features,test],sort=False)
full_ds.shape


# **Checking for Missing Values**

# In[ ]:


# Let's plot these missing values(%) vs column_names, so that we have a visual representation of missing values
#Count
#missing_values_count = (train.isnull().sum().sort_values(ascending=False))
#percentage
missing_values_count = ((full_ds.isnull().sum()/full_ds.isnull().count())*100).sort_values(ascending=False)
plt.figure(figsize=(15,10))
plt.xlabel('Features', fontsize=15)
plt.ylabel('No of missing values in %ge', fontsize=15)
plt.title('Top 10 Variables with missing data', fontsize=15)
sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10],palette="inferno_r",alpha=0.7)


# After looking at the barplot, first question that comes to my mind is, whether it will be prudent to drop the top 4 columns as 80% of the data is missing from those columns. Are these columns going to contribute anything important to our model?
# 
# Nonetheless, Lets treat all the missing values
# 

# In[ ]:


#Lets start with PoolQC
#Categrical Variable which tells us no pools in the house, so missing value means house doesnt have any pools, 
# lets fill missing values with None

full_ds["PoolQC"] = full_ds["PoolQC"].fillna("None")

#MiscFeature: Missing values would mean no miscelleneous features, # lets fill missing values with None
full_ds["MiscFeature"] = full_ds["MiscFeature"].fillna("None")
#Alley: Probably means no Alley Access when there is a missing value,lets fill missing values with None
full_ds["Alley"] = full_ds["Alley"].fillna("None")
#Fence
full_ds["Fence"] = full_ds["Fence"].fillna("None")
#FirePlaceQu
full_ds["FireplaceQu"] = full_ds["FireplaceQu"].fillna("None")
#LotFrontage: Missing values should have a similar Lotfrontage as other houses in the area, we can do a median value of neighburhood 
#and fill the missing values with the median value
full_ds["LotFrontage"] = full_ds.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) 

#GarageYrBlt: Lets add median value from the neighbourhood
full_ds["GarageYrBlt"] = full_ds.groupby("Neighborhood")["GarageYrBlt"].transform(lambda x: x.fillna(round(x.median()))) 
#GarageArea, GarageCars : same as above, assumption that housed in same neighbourhood are similar
full_ds["GarageArea"] = full_ds.groupby("Neighborhood")["GarageArea"].transform(lambda x: x.fillna(round(x.median())))
full_ds["GarageCars"] = full_ds.groupby("Neighborhood")["GarageCars"].transform(lambda x: x.fillna(round(x.median())))

for i in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    full_ds[i] = full_ds[i].fillna('None')
#Basment related variables, missing values likely means no basment
for i in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    full_ds[i] = full_ds[i].fillna(0)
    #for these categorical features, missing values means no Basement
for i in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    full_ds[i] = full_ds[i].fillna('None')

#Masonary Veeneer, if its missing meaning, no veneer and no area which is 0
full_ds["MasVnrType"] = full_ds["MasVnrType"].fillna("None")
full_ds["MasVnrArea"] = full_ds["MasVnrArea"].fillna(0)

#MSZoning: 'RL' is by far the most common value. So we can fill in missing values with 'RL'
full_ds["MSZoning"] = full_ds["MSZoning"].fillna(full_ds['MSZoning'].mode()[0])

#Functional : data description says NA means typical
full_ds['Functional'] = full_ds["Functional"].fillna("typical")

#Electrical : we set most frequent'SBrkr', for the missing value.
full_ds['Electrical'] = full_ds["Electrical"].fillna(full_ds["Electrical"].mode()[0])

#KitchenQual: we set the most frequent value'TA' for the missing value in KitchenQual.
full_ds['KitchenQual'] = full_ds["KitchenQual"].fillna(full_ds["KitchenQual"].mode()[0])
#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
full_ds['Exterior1st'] = full_ds["Exterior1st"].fillna(full_ds["Exterior1st"].mode()[0])
full_ds['Exterior2nd'] = full_ds["Exterior2nd"].fillna(full_ds["Exterior2nd"].mode()[0])

#SaleType : Fill in again with most frequent which is "WD"
full_ds['SaleType'] = full_ds["SaleType"].fillna(full_ds["SaleType"].mode()[0])

#MSSubClass : We can replace missing values with None
full_ds["MSSubClass"] = full_ds["MSSubClass"].fillna("None")


# In[ ]:


#Lets check which variables has maximum missing values
temp = full_ds.isnull().sum().sort_values(ascending=False)
temp[temp>0]


# In[ ]:


#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
full_ds = full_ds.drop(['Utilities'], axis=1)
print("Dropped Utilities")


# In[ ]:


full_ds.dtypes


# In[ ]:


#there are few variables which in my opinion are miscategorised as numeric, I would like to convert them to Categorical variables
#OverallCond , OverallQual, YearBuilt, YearRemodAdded, BsmtFullBath, BsmtHalfBath,FullBath, HalfBath, BedroomAbvGr
#KitchenAbvGr, TorRmsAbvGrd, Fireplaces, GarageCars,MoSold, YrSold, 
temp = ["OverallCond","OverallQual","YearBuilt","YearRemodAdd","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","MoSold","YrSold"]
for label in temp:
    full_ds[label] = full_ds[label].astype('category')

#full_ds["GarageYrBlt"] = full_ds["GarageYrBlt"].astype(str)
print("Converted to Category")
full_ds.dtypes
#full_ds["GarageYrBlt"]=full_ds["GarageYrBlt"].fillna(full_ds["GarageYrBlt"].mode()[0])


# In[ ]:


#converting from Object to category, so that we have consistency in dtypes
categorical_features = full_ds.dtypes[full_ds.dtypes == "object"].index
for label in categorical_features:
    full_ds[label] = full_ds[label].astype('category')
full_ds.dtypes    


# In[ ]:


#Lets check which variables has maximum missing values
temp = full_ds.isnull().sum().sort_values(ascending=False)
temp[temp>0]


# In[ ]:


#Our target variable is SalePrice, let us check if its distributed properly, not touching our full_ds here
from scipy.stats import norm
plt.figure(figsize=(15,10))
sns.distplot(training_label,fit=norm,color='#00C19C')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')


# Our Target Variable looks to be right skewed, We should log transform the variable for our prediction

# In[ ]:


#Log Transform the salePrice
training_label = np.log1p(training_label)
#Plotting Again
plt.figure(figsize=(15,8))
sns.distplot(training_label,fit=norm,color='#00C19C')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')


# **Co-Relation Matrix between each variables**

# In[ ]:


#Lets look at Correltion matrix between variables
corr_mat = train.corr()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap (corr_mat,  square = True,cmap="inferno_r",alpha=0.7,linewidth=0.5,edgecolor='white')


# **#Top 10 co-related variables to SalePrice, (dropping the index 0 as that will be the SalePrice itself )**

# In[ ]:


corr_mat['SalePrice'].sort_values(ascending=False)[1:11] 


# **Log Transforming all  Numerical Features to avoid skewness**

# In[ ]:


#Seleting all Numeric Features
numeric_feats = full_ds.dtypes[full_ds.dtypes != "category"].index
numeric_feats


# In[ ]:



#we compute the skewness and select those variables whose skewness is more than 50%

from scipy.stats import skew
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.50]
skewed_feats = skewed_feats.index
skewed_feats


# In[ ]:


#Applying log transform to all the skewed variables in the full dataset
full_ds[skewed_feats] = np.log1p(full_ds[skewed_feats])
print("Transformed all Numerical Features")


# Creating Dummy Variables for all categorical data

# In[ ]:


#creating Dummy variables from the categorical Data
full_ds = pd.get_dummies(full_ds)
print("created dummy variables")


# In[ ]:


full_ds.sample(5)


# In[ ]:


print (train.shape)
print (test.shape)
print(full_ds.shape)


# In[ ]:


#Creating new training and test set
X_train = full_ds[:train.shape[0]]
X_test = full_ds[train.shape[0]:]
training_label
print("Training Dataset Dimensions:" + str(X_train.shape))
print("Test Dataset Dimensions:" + str(X_test.shape))
print("Training Variable Dimension" + str(training_label.shape))


# 
# **Key features of the model training process:
# **
# * Cross Validation: Using 10-fold cross-validation
# * Models: On each run of cross-validation I fit 7 models (ridge, svr, gradient boosting, random forest, xgboost, lightgbm regressors)
# * Stacking: In addition, I trained a meta StackingCVRegressor optimized using xgboost
# * Blending: All models trained will overfit the training data to varying degrees. Therefore, to make final predictions, I blended their predictions together to get more robust predictions.
# 
# 

# In[ ]:


#setup KFold with 10 splits
kf = KFold(n_splits=10,random_state=23, shuffle=True)


# In[ ]:


#Define our RMSE function: Root Mean Squared Error
def rmse_cross_val (model, X=X_train):
    rmse = np.sqrt(-cross_val_score(model, X, training_label, scoring="neg_mean_squared_error", cv = kf))
    return (rmse)

#we have already scaled back our outliers, even then lets define our RMSLE function: Root Mean Squared log Error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))
print("Defined RMSE function")


# In[ ]:


#LightGBM Regressor
lightgbm = LGBMRegressor(objective='regression',
                         num_leaves=6,
                         learning_rate=0.01,
                         n_estimators=7000,
                         max_bin=200,
                         bagging_fractions=0.8,
                         bagging_freq=4,
                         feature_fractions=0.2,
                         feature_fraction_seed=8,
                         min_sum_hessian_in_leaf=11,
                         verbose=-1,
                         random_state=23)
#XGBoose Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective="reg:linear",
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=22,
                       reg_alpha=0.00006,
                       random_state=23)
                         
    
#RidgeRegressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(),RidgeCV(alphas=ridge_alphas, cv=kf))
#model_ridge = Ridge()
#param = {'alpha': [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}
#ridge_regressor = GridSearchCV(model_ridge,param,scoring='neg_mean_squared_error',cv=5)

#supportvectorregressor
svr = make_pipeline(RobustScaler(),SVR(C=20,epsilon=0.008,gamma=0.0003))

#gradientboostregressor
gbr = GradientBoostingRegressor(n_estimators = 6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=23)
#random forest regressor
rf = RandomForestRegressor(n_estimators=1200,
                           max_depth=5,
                           min_samples_split=5,
                           min_samples_leaf=5,
                           max_features=None,
                           oob_score=True,
                           random_state=23)

#Stack all Models

stack_gen = StackingCVRegressor(regressors=(xgboost,lightgbm,svr,ridge,gbr,rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# In[ ]:


scores={}
score = rmse_cross_val(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['lgb'] = (score.mean(),score.std())


# In[ ]:


score = rmse_cross_val(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['svr'] = (score.mean(),score.std())


# In[ ]:


score = rmse_cross_val(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['ridge'] = (score.mean(),score.std())


# In[ ]:


score = rmse_cross_val(rf)
print("RandomForest: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['rf'] = (score.mean(),score.std())


# In[ ]:


score = rmse_cross_val(gbr)
print("gradientboostingregressor: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['gbr'] = (score.mean(),score.std())


# **Lets train those dragons**

# In[ ]:


print("stack_gen")
stack_gen_model = stack_gen.fit(np.array(X_train),np.array(training_label))


# In[ ]:


print("lightgbm")
lgbm_model_full_data=lightgbm.fit(X_train,training_label)


# In[ ]:


print("xgboost")
xgbm_model_full_data=xgboost.fit(X_train,training_label)


# In[ ]:


print("svr")
svr_model_full_data=svr.fit(X_train,training_label)


# In[ ]:


print("Ridge")
ridge_model_full_data=ridge.fit(X_train,training_label)


# In[ ]:


print("RandomForest")
rf_model_full_data=rf.fit(X_train,training_label)


# In[ ]:


print("gradientBoosting")
gbr_model_full_data=gbr.fit(X_train,training_label)


# In[ ]:


#Blending Models for final prediction
def blended_predictions(X_train):
    return ((0.1 * ridge_model_full_data.predict(X_train))+
            (0.1 * svr_model_full_data.predict(X_train))+
            (0.1 * gbr_model_full_data.predict(X_train))+
            (0.1 * xgbm_model_full_data.predict(X_train))+
            (0.1 * lgbm_model_full_data.predict(X_train))+
            (0.1 * rf_model_full_data.predict(X_train))+
            (0.40 * stack_gen_model.predict(np.array(X_train))))


# In[ ]:


#Getting final predictions
blended_score = rmsle(training_label, blended_predictions(X_train))
scores['blended'] = (blended_score,0)
print("RMSLE Score on training data:")
print(blended_score)


# In[ ]:


#Identifying best model
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'],color='green')
for i, score in enumerate(scores.values()):
    ax.text(i,score[0]+0.002,'{:.6f}'.format(score[0]),horizontalalignment='left',size='large',color='purple',weight='semibold')
plt.ylabel('Score(RMSE)',size=20)
plt.xlabel('Model',size=20)
plt.title("Scores of Models",size=20)
plt.show()


# our blended model outperforms all other models with a rmsle of 0.085. Lets use this model for our final predictions

# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")
submission.shape


# In[ ]:


blendPred = blended_predictions(X_test)


# In[ ]:


#Append Predictions from blended model
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test)))


# In[ ]:


# Scale predictions
submission['SalePrice'] *= 1.001619
submission.to_csv("submission_regression.csv", index=False)

