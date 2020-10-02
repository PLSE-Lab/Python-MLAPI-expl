#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display

# remove warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


train = pd.read_csv('../input/train.csv',index_col='Id')
test  = pd.read_csv('../input/test.csv',index_col='Id')


# In[ ]:


print(train.shape)
display(train.head(1))

print(test.shape)
display(test.head(1))


# # Explore Data & Wrangle Data

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[ ]:


train.head(5)


# In[ ]:


train.SalePrice.describe()


# In[ ]:


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


# ## Transform the target variable

# **By taking log of SalePrice the skew reduces to a larger extent**

# In[ ]:


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# ### Analyze Numeric features for correlations with Target variable

# In[ ]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# ## Check the top 10 and Bottom 10 correlated and non-corelate features respectively

# In[ ]:


corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[1:11], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-10:])


# ## Analyze the highly correlated features

# ### Analyze Overall Quality (OverallQual) - highest correlation with SalePrice

# In[ ]:


# How many unique features are there?

train.OverallQual.unique()


# **Even though OverallQual is a numeric feature but it has a rating from 1-10**
# 
# **Let's check how the Overall Quality rating affects the median price of the house**

# In[ ]:



#Define a function which can pivot and plot the intended aggregate function 
def pivotandplot(data,variable,onVariable,aggfunc):
    pivot_var = data.pivot_table(index=variable,
                                  values=onVariable, aggfunc=aggfunc)
    pivot_var.plot(kind='bar', color='blue')
    plt.xlabel(variable)
    plt.ylabel(onVariable)
    plt.xticks(rotation=0)
    plt.show()
    


# In[ ]:


pivotandplot(train,'OverallQual','SalePrice',np.median)


# **The median SalePrice shows and increasing trend with increase in quality rating**

# ### Analyze the GrLivArea: Above grade (ground) living area square feet - Second highest correlation with SalePrice

# In[ ]:


# It is a continous variable and hence lets look at the relationship of GrLivArea with SalePrice using a Regression plot

_ = sns.regplot(train['GrLivArea'], train['SalePrice'])


# **The regression plot clearly indicates that rices increase with increase in GrLivArea however there are some outliers  where for GrLivArea above 4000 the prices are below 20k**
# **Let's Remove the outliers**

# In[ ]:


train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
_ = sns.regplot(train['GrLivArea'], train['SalePrice'])


# **Looks much better now**

# ### Analyze Garage Area 

# In[ ]:


_ = sns.regplot(train['GarageArea'], train['SalePrice'])


# **Some Outliers after GarageArea of 1200**
# **Remove the outliers**

# In[ ]:


train = train[train['GarageArea'] < 1200]
_ = sns.regplot(train['GarageArea'], train['SalePrice'])


# ### Impute the Data for missing values

# * **Before imputing the categorical values it is very important that we impute it on entire (Dev + Test) **
# * ** This is of outmost importance since some of the categories might be missing in the test data  which will create problems in OneHotEncoding later when we run models on test data **
# 

# ### Merge train and test data

# **Let's find the percentage of missing values for the features**

# In[ ]:


# Let us first create a DF for log transformation of SalePrice
train['log_SalePrice']=np.log(train['SalePrice']+1)
saleprices=train[['SalePrice','log_SalePrice']]

saleprices.head(5)


# In[ ]:


train=train.drop(columns=['SalePrice','log_SalePrice'])


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


all_data = pd.concat((train, test))
print(all_data.shape)
all_data.head(5)


# In[ ]:


null_data = pd.DataFrame(all_data.isnull().sum().sort_values(ascending=False))[:50]

null_data.columns = ['Null Count']
null_data.index.name = 'Feature'
null_data


# In[ ]:


(null_data/len(all_data)) * 100


# * ** 99% of Pool Quality Data is missing.In the case of PoolQC, the column refers to Pool Quality. Pool quality is NaN when PoolArea is 0, or there is no pool.**
# * ** Similar is case for Garage column **
# 
# ** But what are the 96% missing Miscelleanous features ?**
# 
# ** Let's find out **

# In[ ]:


print ("Unique values are:", train.MiscFeature.unique())


# ** The options in Miscellenous features explain if there is Shed or Second Garage or some TenC **

# #### Impute Categorical data for missing values and relace by 'None'****

# In[ ]:


for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    all_data[col] = all_data[col].fillna('None')


# #### Impute the numerical features and replace with a value of zero

# In[ ]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)


# In[ ]:


for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])


# **LotFrontage seems like an important feature let's look at the relation with SalePrice**

# In[ ]:


_=sns.regplot(train['LotFrontage'],saleprices['SalePrice'])


# ** Impute the LotFrontage with Median values**

# In[ ]:


all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))


# ## New Features

# * **TotalBsmtSF - Total Basement Square Feet  **
# * **1stFlrSF - First Floor Square Feet **
# * **2ndFlrSF - Second Floor Square Feet **
# 
# **All the above three feature define area of the house and we can easily combine these to form TotalSF - Total Area in square feet**
# 

# In[ ]:


figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
_ = sns.regplot(train['TotalBsmtSF'], saleprices['SalePrice'], ax=ax1)
_ =sns.regplot(train['1stFlrSF'], saleprices['SalePrice'], ax=ax2)
_ = sns.regplot(train['2ndFlrSF'], saleprices['SalePrice'], ax=ax3)
_ = sns.regplot(train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'], saleprices['SalePrice'], ax=ax4)


# In[ ]:


#Impute the entire data set
all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#Let's add two new variables for No nd floor and no basement
all_data['No2ndFlr']=(all_data['2ndFlrSF']==0)
all_data['NoBsmt']=(all_data['TotalBsmtSF']==0)


# **The  BsmtFullBath ,FullBath, BsmtHalfBath can be combined for a TotalBath similar to TotalSF**

# In[ ]:


figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
_ = sns.barplot(train['BsmtFullBath'], saleprices['SalePrice'], ax=ax1)
_ = sns.barplot(train['FullBath'], saleprices['SalePrice'], ax=ax2)
_ = sns.barplot(train['BsmtHalfBath'], saleprices['SalePrice'], ax=ax3)
_ = sns.barplot(train['BsmtFullBath'] + train['FullBath'] + train['BsmtHalfBath'] + train['HalfBath'], saleprices['SalePrice'], ax=ax4)


# In[ ]:


all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['BsmtHalfBath'] + all_data['HalfBath']


# In[ ]:


figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,8)
_ = sns.regplot(train['YearBuilt'], saleprices['SalePrice'], ax=ax1)
_ = sns.regplot(train['YearRemodAdd'], saleprices['SalePrice'], ax=ax2)
_ = sns.regplot((train['YearBuilt']+train['YearRemodAdd'])/2, saleprices['SalePrice'], ax=ax3)


# In[ ]:


all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']


# In[ ]:


# Deleting dominating features over 97%
all_data=all_data.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating'])


# In[ ]:


# treat some numeric values as str which is actually a categorical data
all_data['MSSubClass']=all_data['MSSubClass'].astype(str)
all_data['MoSold']=all_data['MoSold'].astype(str)
all_data['YrSold']=all_data['YrSold'].astype(str)


# In[ ]:


# I found these features might look better without 0 data. (just like the column '2ndFlrSF' above.)
all_data['NoLowQual']=(all_data['LowQualFinSF']==0)
all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)
all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)
all_data['NoGarage']=(all_data['GarageArea']==0)
all_data=all_data.drop(columns=['PoolArea','PoolQC']) # most of the houses has no pools. 
all_data=all_data.drop(columns=['MiscVal','MiscFeature']) # most of the houses has no misc feature.


# ### Group the similar featurtes  related to a House Feature and analyze

# In[ ]:


Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']
Bsmt=all_data[Basement]
Bsmt.head()


# 
# BsmtQual: Evaluates the height of the basement
# 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 		
# BsmtCond: Evaluates the general condition of the basement
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 	
# BsmtExposure: Refers to walkout or garden level walls
# 
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
# 	
# BsmtFinType1: Rating of basement finished area
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 		
# BsmtFinSF1: Type 1 finished square feet
# 
# BsmtFinType2: Rating of basement finished area (if multiple types)
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 
# BsmtFinSF2: Type 2 finished square feet
# 
# BsmtUnfSF: Unfinished square feet of basement area
# 
# TotalBsmtSF: Total square feet of basement area

# In[ ]:


Bsmt['BsmtCond'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


Bsmt.head()


# In[ ]:


Bsmt['BsmtCond'].unique()


# In[ ]:


# Replacing Categorical values to numbers
cond_encoder = LabelEncoder()
Bsmt['BsmtCond']=cond_encoder.fit_transform(Bsmt['BsmtCond'])

exposure_encoder = LabelEncoder()
Bsmt['BsmtExposure'] = exposure_encoder.fit_transform(Bsmt['BsmtExposure'])

finTyp1_encoder = LabelEncoder()
Bsmt['BsmtFinType1'] = finTyp1_encoder.fit_transform(Bsmt['BsmtFinType1'])

finTyp2_encoder = LabelEncoder()
Bsmt['BsmtFinType2'] = finTyp2_encoder.fit_transform(Bsmt['BsmtFinType2'])

qual_encoder = LabelEncoder()
Bsmt['BsmtQual'] = qual_encoder.fit_transform(Bsmt['BsmtQual'])


# In[ ]:


Bsmt.head()


# In[ ]:


Bsmt['BsmtScore']= Bsmt['BsmtQual']  * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']
all_data['BsmtScore']=Bsmt['BsmtScore']


# In[ ]:


Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])
all_data['BsmtFinScore']=Bsmt['BsmtFin']
all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)


# LotFrontage: Linear feet of street connected to property
# 
# LotArea: Lot size in square feet
# 
# LotShape: General shape of property
# 
#        Reg  Regular 
#        IR1  Slightly irregular
#        IR2  Moderately Irregular
#        IR3  Irregular
# 
# LotConfig: Lot configuration
# 
#        Inside   Inside lot
#        Corner   Corner lot
#        CulDSac  Cul-de-sac
#        FR2  Frontage on 2 sides of property
#        FR3  Frontage on 3 sides of property
# 

# In[ ]:


lot=['LotFrontage', 'LotArea','LotConfig','LotShape']
Lot=all_data[lot]


# In[ ]:



Lot.head()


# In[ ]:


garage=['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
Garage=all_data[garage]


# In[ ]:


Garage.head()


# In[ ]:


garcond_encoder = LabelEncoder()
Garage['GarageCond'] = garcond_encoder.fit_transform(Garage['GarageCond'])

garfin_encoder = LabelEncoder()
Garage['GarageFinish'] = garfin_encoder.fit_transform(Garage['GarageFinish'])

garqual_encoder = LabelEncoder()
Garage['GarageQual'] = garqual_encoder.fit_transform(Garage['GarageQual'])

gartyp_encoder = LabelEncoder()
Garage['GarageType'] = gartyp_encoder.fit_transform(Garage['GarageType'])



# In[ ]:


Garage.head()


# In[ ]:


Garage['GarageScore']=(Garage['GarageArea']) * (Garage['GarageCars']) * (Garage['GarageFinish'])*(Garage['GarageQual']) *(Garage['GarageType'])
all_data['GarageScore']=Garage['GarageScore']


# In[ ]:


all_data.head()


# # Data Preprocessesing

# In[ ]:


non_numeric=all_data.select_dtypes(exclude=[np.number, bool])
non_numeric.head()


# In[ ]:



def onehot(col_list):
    global all_data
    while len(col_list) !=0:
        col=col_list.pop(0)
        data_encoded=pd.get_dummies(all_data[col], prefix=col)
        all_data=pd.merge(all_data, data_encoded, on='Id')
        all_data=all_data.drop(columns=col)
    print(all_data.shape)


# In[ ]:


onehot(list(non_numeric))


# In[ ]:


def log_transform(col_list):
    transformed_col=[]
    while len(col_list)!=0:
        col=col_list.pop(0)
        if all_data[col].skew() > 0.5:
            all_data[col]=np.log(all_data[col]+1)
            transformed_col.append(col)
        else:
            pass
    print(f"{len(transformed_col)} features had been tranformed")
    print(all_data.shape)


# In[ ]:


numeric=all_data.select_dtypes(include=np.number)
log_transform(list(numeric))


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train=all_data[:len(train)]
test=all_data[len(train):]

# re-Set the train & test data for ML


# In[ ]:


print(train.shape)
print(test.shape)

# OK. I'm ready


# # Modeling

# In[ ]:


# loading pakages for model. 
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn import linear_model, model_selection, ensemble, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

#Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score,mean_absolute_error


# In[ ]:


def rmse(predict, actual):
    score =mean_squared_error(Ytrain, y_pred)**0.5
    return score
rmse_score = make_scorer(rmse)
rmse_score


# In[ ]:


feature_names=list(all_data)
Xtrain=train[feature_names]
Xtest=test[feature_names]
Ytrain=saleprices['log_SalePrice']


# In[ ]:


def score(model):
    score = cross_val_score(model, Xtrain, Ytrain, cv=5, scoring=rmse_score).mean()
    return score


# **Create a dictionary to keep track of scores for each model and compare later**

# In[ ]:


scores ={}


# ## Simple Linear Regression

# In[ ]:


lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(Xtrain,Ytrain)

#accuracies = cross_val_score(estimator=lr_model,
                         #   X=Xtrain,
                         #   y=Ytrain,
                          #  cv=5,
                          #  verbose=1)

y_pred = lr_model.predict(Xtrain)

print('')
print('####### Linear Regression #######')
meanCV = score(lr_model)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'OLS':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## Lasso Regression

# In[ ]:


model_lasso = Lasso(random_state=42,alpha=0.00035)
lr_lasso = make_pipeline(RobustScaler(), model_lasso)
lr_lasso.fit(Xtrain,Ytrain)



y_pred = lr_lasso.predict(Xtrain)

print('')
print('####### Lasso Regression #######')
meanCV = score(lr_lasso)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'Lasso':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## Ridge Regression

# In[ ]:


lr_ridge = make_pipeline(RobustScaler(), Ridge(random_state=42,alpha=0.002))
lr_ridge.fit(Xtrain,Ytrain)



y_pred = lr_ridge.predict(Xtrain)

print('')
print('####### Ridge Regression #######')
meanCV = score(lr_ridge)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'Ridge':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## ElasticNet Regression

# In[ ]:


lr_elasticnet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.02, l1_ratio=0.7,random_state=42))
lr_elasticnet.fit(Xtrain,Ytrain)



y_pred = lr_elasticnet.predict(Xtrain)

print('')
print('####### ElasticNet Regression #######')
meanCV = score(lr_elasticnet)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'ElasticNet':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## KNN Regression

# In[ ]:


knn = make_pipeline(RobustScaler(),KNeighborsRegressor())
knn.fit(Xtrain,Ytrain)



y_pred = knn.predict(Xtrain)

print('')
print('####### KNN Regression #######')
meanCV = score(knn)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'KNN':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## GradientBoosting Regression

# In[ ]:


model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =42)

model_GBoost.fit(Xtrain,Ytrain)
y_pred = model_GBoost.predict(Xtrain)

print('')
print('####### GradientBoosting Regression #######')
meanCV = score(model_GBoost)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'GradientBoosting':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## RandomForest Regressor

# In[ ]:


forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(Xtrain, Ytrain)

forest_reg.fit(Xtrain,Ytrain)
y_pred = forest_reg.predict(Xtrain)

print('')
print('####### RandomForest Regression #######')
meanCV = score(forest_reg)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'RandomForest':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# ## Grid Search for finding best params for RandomForest

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
  
    {'n_estimators': [70,100], 'max_features': [150]},
   
    {'bootstrap': [True], 'n_estimators': [70,100], 'max_features': [150]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(Xtrain, Ytrain)

print('')
print('####### GridSearch RF Regression #######')
meanCV = score(grid_search)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'GridSearchRF':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)


# In[ ]:


grid_search.best_estimator_


# In[ ]:


scores_list =[]
for k,v in scores.items():
    temp_lst =[]
    temp_lst.append(k)
    temp_lst.extend(v)
    scores_list.append(temp_lst)


# In[ ]:


scores_df =pd.DataFrame(scores_list,columns=['Model','CV_Mean_Score','MSE(RSS)','MAE','RMSE','R2Squared'])


# In[ ]:


scores_df.sort_values(['CV_Mean_Score'])


# In[ ]:


_ =sns.scatterplot(x='Model',y='CV_Mean_Score',data=scores_df,style='Model')


# **Current Winning model looks to be GradientBoosting  on cross validation data**

# ## Predictions

# **Note Even though a model wins on CV score it may not be best predictor. I could not validate data on test as true Y's are not available in kaggle. So the only way would be to predict on good models and try submitting scores for them**

# **We had log transformed the Ytrain and hence it is essential to transform it back to original by taking an exponential of model predictions**

# In[ ]:


Lasso_Predictions=np.exp(lr_lasso.predict(Xtest))-1

GBoost_Predictions=np.exp(model_GBoost.predict(Xtest))-1

KNN_Predictions=np.exp(knn.predict(Xtest))-1

GridSearch_Predictions = np.exp(grid_search.best_estimator_.predict(Xtest))-1


# In[ ]:


submission=pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission['SalePrice'] = Lasso_Predictions
submission.to_csv('Lasso.csv',index=False)

submission['SalePrice'] = GBoost_Predictions
submission.to_csv('GBoost.csv',index=False)

submission['SalePrice'] = KNN_Predictions
submission.to_csv('KNN.csv',index=False)

submission['SalePrice'] = GridSearch_Predictions
submission.to_csv('GidSearch.csv',index=False)


# ## Coefficients of the Model

# **Since Lasso is wininng in predictions here are the coefficients**

# In[ ]:


coef_df = pd.DataFrame(list(zip(train.columns, model_lasso.coef_)),columns=['Feature','Coefficient'])
coef_df


# In[ ]:




