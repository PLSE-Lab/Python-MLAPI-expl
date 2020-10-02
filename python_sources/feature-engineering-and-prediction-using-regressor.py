#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import gc
from scipy.stats import norm
import matplotlib.ticker as mticker
import scipy.stats as stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


traindf = pd.read_csv("../input/train.csv")
testdf = pd.read_csv("../input/test.csv")


# Printing shapes of train and test data  

# In[ ]:


print(traindf.shape,testdf.shape)


# In[ ]:


for c in traindf.columns:
    if c not in testdf.columns.values:
        print(c)


# We will combine both train and test data sets into single data set which help later to eliminate all the null values in one task and also our insights of features should contains all the data  

# In[ ]:


all_data = traindf.drop('SalePrice',axis=1).append(testdf).reset_index(drop=True)


# In[ ]:


all_data.info()


# In[ ]:


(mu, sigma) = stats.norm.fit(traindf["SalePrice"])
print("mu:{:.3f} and sigma:{:.3f}".format(mu,sigma))
plt.figure(figsize=(12,7))
sns.distplot(traindf['SalePrice'],fit=stats.norm)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution SalePrice")
plt.legend(['Normal dist.( $\mu=$ {:.3f} and $\sigma=$ {:.3f} )'.format(mu,sigma)],loc='best')
plt.show()


# we can see that distribution of our target feature("SalePrice") is skewed left side, since most of statistical models  are applicable on Normal distribution therefor we should convert the feature distribution in normal distribution which is obtained by taking logarithimic values of the data, we will do that later 
# 
# Now we will see two important parameters for mesuring normal distribution, skewness and kurtosis, skewness which measures how symmetric is the data around the mean value and kurtosis tell the pointyness or flatness of the curve  as we can see from the plot, it is skewed left and also pointy 

# In[ ]:


fig = plt.figure()
stats.probplot(traindf['SalePrice'],plot=plt)
plt.show()


# In[ ]:


print('Skewness: %f' % traindf['SalePrice'].skew())
print('Kurtosis: %f' % traindf['SalePrice'].kurt())


# perfect normal distribution curve have zero kurtosis and zero skewness 
# 1. kurt > 0 => pointier than normal distribution . 
# 2. kurt < 0 => represent flatness of the curve 
# 

# In[ ]:


(mu,sigma) = stats.norm.fit(np.log(traindf['SalePrice']))
fig = plt.figure()
sns.distplot(np.log(traindf['SalePrice']),fit=stats.norm)
plt.title("Distribution of SalePrice after Log conversion")
plt.ylabel("Frequency")
plt.legend(["Normal dist.($\mu=$ {:0.3f} and $\sigma=${:.3f})".format(mu,sigma)],loc='best')
plt.show()

plt.figure()
stats.probplot(np.log(traindf['SalePrice']),plot=plt)
plt.show()
print('Skewness: %f' % np.log(traindf['SalePrice']).skew())
print('Kurtosis: %f' % np.log(traindf['SalePrice']).kurt())


# after converting the traget value into log values we can see the curve is more like normal distribution curve and we can see both parameters skew and kurt degraded to near zero value

# we will make two lists one will contain all the numerical variables of the dataset and another one will contain all the categorical variable 

# In[ ]:


number_cols = [c for c in all_data._get_numeric_data().columns]
categorical_cols = all_data.select_dtypes(include=['object']).columns.values


# In[ ]:


print(categorical_cols[:5])
print(number_cols[:5])


# 
# 
# 
# 
# to check how many features contain single constant value, since those features are trash, we will eliminate those feature from our dataset

# In[ ]:


contsant_columns = [c for c in all_data.columns if traindf[c].nunique()==1]
contsant_columns


# **Missing Values handling**

# we should count missing values in each feature, we dont want missing values in our dataset because that will create problem for training our dataset(there are still some algorithms available which support missing values such as KNN) therefor we should handle those missing values in some way. There are multiple ways one can handle the missing values 
# 1. deleting rows which contains missing values, since this method losses the data therefor it is not preferable
# 2. Replacing with mean/mode or median values of the same feature
# 3. predicting the missing values etc.
# 

# In[ ]:


missing_values  = (all_data.isnull().sum()/all_data.shape[0])*100
missing_values = missing_values[missing_values>0]
plt.figure(figsize=(13,5))
missing_values = missing_values.sort_values(ascending=False)
missing_values.plot.bar()
plt.title("MissingValues in Dataset by each column")
plt.xlabel("Columns")
plt.ylabel("Missing Values Ratio")
plt.show()
pd.DataFrame({"Missing Ratio":missing_values})


#  if we see carefully in description data file we can see that many of these feature exactly dont contain missing values because N/A means Not present for example feature "PoolQC" has approx 99% in N/A in data set but that does not mean those are missing values , actually those represent not available Pool for those house data. so we will not eliminate these values, we will replace those values with "None" so that we find another categoy. 

# all those features which has more than 50% missing ratio are not actually missing(see description.txt for clarification) so we will replace these N/A will None

# In[ ]:


columns_cat1 = missing_values[missing_values>40].index
all_data[columns_cat1] = all_data[columns_cat1].fillna("None")


# Lotfrontage is actually a numerical and we will replace with mean value

# In[ ]:


all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data['LotFrontage'].mean())


# for all other garage related variables we can replace N/A with None 

# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')


# In[ ]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# These feature contains N/A values which are replace by either "None" or mean/mode

# In[ ]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typical")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# In[ ]:


for c in all_data.columns:
    if np.any(all_data[c].isnull().sum()>0):
        print(c)


# In[ ]:


sns.countplot(traindf["Utilities"])


#  since most of values are AllPub therefor we can consider this as a constant feature which won't help us in model therefor we can eliminate this feature from the dataset

# In[ ]:


all_data = all_data.drop("Utilities",axis=1)


# Since there are variable which are actually not actually numericals instead those are categorical variables such as OverallQual,MSSubClass and OverallCond. so we shold transform these variable in string data type 

# In[ ]:


cols_cat2 = ["MSSubClass","MoSold","YrSold","OverallQual","OverallCond"] 
for c in cols_cat2:
    all_data[c] = all_data[c].astype(str)


# In[ ]:


all_data['TotalArea'] = all_data['TotalBsmtSF'] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]


# **Some Insights**

# In[ ]:


trainData = all_data[:traindf.shape[0]]
trainData['SalePrice'] = traindf['SalePrice']
fig,ax= plt.subplots(1,3,figsize=(17,5))
sns.swarmplot(x="PavedDrive",y='SalePrice',data=trainData,ax=ax[0])
sns.swarmplot(x="GarageFinish",y ='SalePrice',data=trainData,ax=ax[1])
sns.swarmplot(x="LandSlope",y='SalePrice',data=trainData,ax=ax[2])


# In[ ]:


fig,ax= plt.subplots(2,2,figsize=(15,8))
sns.countplot(x='OverallCond',data=all_data,ax=ax[0,0])
sns.countplot(x="OverallQual",data=all_data,ax=ax[0,1])
sns.stripplot(x='OverallCond',y="SalePrice",data=trainData,ax=ax[1,0])
sns.stripplot(x='OverallQual',y='SalePrice',data=trainData,ax=ax[1,1])
ax[0,0].set_title("Overall Condtion of the house")
ax[0,1].set_title("Overall Quality of material and finish of the house")


# In[ ]:


fig,ax= plt.subplots(2,2,figsize=(15,8))
sns.boxplot(x='MSZoning',y='SalePrice',data=trainData,ax=ax[0,0])
# ax[0,0].set_xticklabels(["Residential Low Density","Residential Medium Density","Commercial","Floating Village Residential","Residential High Density"],rotation=45)

sns.boxplot(x='LotShape',y='SalePrice',data=trainData,ax=ax[0,1])
sns.boxplot(x='LotConfig',y='SalePrice',data=trainData,ax=ax[1,0])
sns.boxplot(x='Foundation',y='SalePrice',data=trainData,ax=ax[1,1])


# In[ ]:


plt.figure(figsize=(15,8))
sns.swarmplot(x='SaleCondition',y='SalePrice',hue='SaleType',data=trainData)


# **Outliers Detection**

# first we will see the scatter plots of numerical variables and will find outliers if present in any variable.

# In[ ]:


Number_cols = [c for c in trainData.drop("Id",axis=1)._get_numeric_data().columns]


# In[ ]:


sns.pairplot(trainData,x_vars=Number_cols,y_vars=["SalePrice"])


# Not clearly but we can see from scatter matrix, there are multiple feature exist in which outliers can be present such as LotFrontage, LotArea and GrLivArea  therfor we will see these variables one by one

# GrLivArea

# In[ ]:


sns.scatterplot(x="GrLivArea",y='SalePrice',data=trainData)


# As we can see there are two values present in the training set which has very low value of saleprice for very high value of area therefor we should remove those values

# In[ ]:


trainData = trainData.drop(trainData[(trainData["GrLivArea"]>4000) & (trainData["SalePrice"]<400000)].index)
sns.scatterplot(x="GrLivArea",y='SalePrice',data=trainData)


#  Nice!

# LotArea

# In[ ]:


sns.scatterplot(x="LotArea",y='SalePrice',data=trainData)


# we should not remove the points lies on the right because their values of saleprice are not so low and there are other factors which are actually affecting the price of that point

# **More feature Engineering**

# In[ ]:


# catg_cols = all_data.select_dtypes(include=["object"]).columns.values
# from sklearn.preprocessing import LabelEncoder
# for c in catg_cols:
#     LE = LabelEncoder()
#     LE.fit(list(all_data[c].values))
#     all_data[c] = LE.transform(list(all_data[c].values))


# we will take all categorical variables and do  oneHotEncode

# **Skewed Feature Testing**

# In[ ]:


skew = []
for c in Number_cols:
    if c not in ['SalePrice'] and abs(all_data[c].skew())>0.75:
        skew.append(c)
print("there are total {} skewed features present in the data set".format(len(skew)))


# Now we will convert these variables into normal dist using log1p method 

# In[ ]:


for c in skew:
    all_data[c] = np.log1p(all_data[c])


# we will get the dummies variable for the categorical variables

# In[ ]:


all_data = pd.get_dummies(all_data)


# In[ ]:


train = all_data[:traindf.shape[0]]
test  = all_data[traindf.shape[0]:]


# In[ ]:


corr_matrix = trainData.corr()
plt.figure(figsize=(24,24))
sns.heatmap(corr_matrix,annot=True,xticklabels=corr_matrix.columns.values,yticklabels=corr_matrix.columns.values)


# In[ ]:


all_data = all_data.drop(["TotalBsmtSF","GrLivArea","GarageCars","TotRmsAbvGrd"],axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler,RobustScaler
scaler =RobustScaler()
trainX= scaler.fit_transform(train.values)


# In[ ]:


testX= scaler.fit_transform(test.values)


# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# In[ ]:


regressor=[]
rd = Ridge(alpha=9)
lr = LinearRegression()
ls = Lasso(alpha=0.1,random_state=2)
#mtl = MultiTaskLasso(alpha=0.1,random_state=2)
en = ElasticNet(alpha=0.1,random_state=42)
rf = RandomForestRegressor(100,max_depth=100)
dtr = DecisionTreeRegressor(max_depth=100,max_features="auto")
svr = SVR()
regressor.extend([rd,lr,ls,en,rf,dtr,svr])


# In[ ]:


from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


# In[ ]:


trainY = np.log(traindf['SalePrice'])
trainx,testx,trainy,testy = train_test_split(trainX,trainY,test_size=0.30,random_state=42)
print(trainx.shape,testx.shape)


# In[ ]:


def rmse(model,n_CV):
        kf = KFold(n_CV,random_state=42,shuffle=True).get_n_splits()
        score = np.sqrt(-cross_val_score(model,trainx,trainy,scoring="neg_mean_squared_error",cv=kf))
        return(score)


# In[ ]:


cv_score = []
rmse_score=[]
for model in regressor:
    score = rmse(model,5)
    mean_score = np.mean(score)
    cv_score.append(score)
    rmse_score.append(mean_score)


# In[ ]:


CV_std = [np.std(x) for x in cv_score]
models_result = pd.DataFrame({"CVmeans":rmse_score,"Algorithms":["Ridge","LinearRegression","Lasso","ElasticNet","RandomForestRegressor","DecisiontreeRegressor","SupportVectorRgression"],"CVstd":[np.std(x) for x in cv_score]})
models_result


# In[ ]:


sns.barplot(x="CVmeans",y="Algorithms",data=models_result,**{"xerr":CV_std})
plt.title("Models performance")
plt.xlabel("avg rmse value for cross validation")
plt.show()


# In[ ]:


rd_params ={"alpha":[0,1,9,50,100,500],"solver":["sag","saga","auto"]}
rf_params = {"max_depth":[100,200,400,500],"bootstrap":[True,False]}
lasso_params = {"alpha":[0.0001,0.0005,0.001,0.01,0.1,1,3,9,50,100,500,1000]}
svm_params = {"kernel":["rbf","poly"],"degree":[3,5],"gamma":["auto","scale"]}


# **Ridge**

# In[ ]:



def rmse_val(actual, predict):
        return -np.sqrt(mean_squared_error(actual,predict))
ridge =Ridge()
my_scorer = make_scorer(rmse_val,greater_is_better=True)
grid_ridge = GridSearchCV(ridge,param_grid=rd_params,cv=5,scoring=my_scorer,n_jobs=-1,verbose=1)
grid_ridge.fit(trainx,trainy)


# In[ ]:


grid_ridge.best_params_


# In[ ]:


grid_ridge.best_score_


# In[ ]:


# ridge =Ridge(alpha=9,solver="auto")
# ridge.fit(trainx,trainy)
# predc = ridge.predict(testx)
# np.mean(rmse(ridge,10))


# **RandomForest**

# In[ ]:


grid_rf = GridSearchCV(rf,param_grid = rf_params,scoring=my_scorer,n_jobs=-1,verbose=1,cv=10)
grid_rf.fit(trainx,trainy)


# In[ ]:


grid_rf.best_params_


# In[ ]:


grid_rf.best_score_


# In[ ]:


rf = RandomForestRegressor(bootstrap=True, max_depth= 200)
np.mean(rmse(rf,10))


# **Lasso**

# In[ ]:


lsg=Lasso(random_state=42)
lasso_grid = GridSearchCV(lsg,param_grid=lasso_params,cv=10,n_jobs=-1,verbose=1,scoring=my_scorer)
lasso_grid.fit(trainx,trainy)


# In[ ]:


lasso_grid.best_params_


# In[ ]:


lsg_1=Lasso(random_state=42,alpha=0.0005)
np.mean(rmse(lsg_1,10))


# **GradientBooster**

# In[ ]:


import  lightgbm as lgb
import xgboost as xgb


# In[ ]:


lgb_model=lgb.LGBMRegressor(objective='regression',num_leaves=10,learning_rate=0.05,n_estimators=720,max_bin = 55,min_data_in_leaf =6)
lgb_model.fit(trainx,trainy)


# In[ ]:


xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
np.mean(rmse(xgb_model,5))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pred = lgb_model.predict(testX)


# In[ ]:


prediction = np.exp(pred)


# In[ ]:


my_submission =pd.DataFrame({"Id":testdf.Id,"SalePrice":prediction})
my_submission.to_csv("output.csv",index=False)


# In[ ]:




