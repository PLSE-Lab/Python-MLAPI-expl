#!/usr/bin/env python
# coding: utf-8

# 1. ### Predict House Prices with ML GradientBoostingRegressor
# Objective: 
# build a regression model that can predict the sales price of the house that match market value. 
# The key here is to make the develop a linear regression models to help us predict housing prices based on most relevant features and greatest impact to the price.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy.stats import norm, skew #for some statistics

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

seed = 5
np.random.seed(seed)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input"))

input_train_file="../input/train.csv"
input_test_file="../input/test.csv"
train_input = pd.read_csv(input_train_file)
test_input = pd.read_csv(input_test_file)

sample_submission = pd.read_csv("../input/sample_submission.csv")


# **Explorer input data**

# Training data

# In[ ]:


print ("Train input data info: ")
print (train_input.info())
print (train_input.describe())


# In[ ]:


print ("Has missing values? ", train_input.isnull().values.any())
print ("\nNumber of missing values :", train_input.isnull().sum().sum())  
print ("\nColumn with missing value: ",train_input.columns[train_input.isnull().any()])


# In[ ]:


print ("Missing value by column name: ") 
print (train_input.isnull().sum())
#print (input_train.isnull().sum().sum())    


# In[ ]:


train_input.head(5)


# Input Test Data

# In[ ]:


print ("Test input data info: ")
print (test_input.info())
print (test_input.describe())


# In[ ]:


print ("Has missing values? ", test_input.isnull().values.any())
print ("\nNumber of missing values :", test_input.isnull().sum().sum())  
print ("\nColumn with missing value: ",test_input.columns[test_input.isnull().any()])


# In[ ]:


test_input.head(5)


# In[ ]:


#train_input.hist()


# #### Analysing label 'SalePrice'   and relationships with features

# In[ ]:


train_input['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(train_input['SalePrice']);


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % train_input['SalePrice'].skew())
print("Kurtosis: %f" % train_input['SalePrice'].kurt())


# **Correlation matrix**

# In[ ]:


#heatmap
corrmat = train_input.corr()
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corrmat, vmax=.8, square=True);


# **Relationship with features**

# hows does number of car garage affect sales price?

# In[ ]:


#scatter plot GarageCars/saleprice
var = 'GarageCars'
data = pd.concat([train_input['SalePrice'], train_input[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# Does number of bedroom affect sales price?

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train_input['SalePrice'], train_input[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# how about living area?

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train_input['SalePrice'], train_input[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# Does year built has impact on sales price?

# In[ ]:


var = 'YearBuilt'
data = pd.concat([train_input['SalePrice'], train_input[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# **Prepare data for model training**
# this is the most important part

# In[ ]:


#drop unused column id
house_train_data = train_input.drop('Id', axis=1)
house_test_data = test_input.drop('Id', axis=1)

#We will find all the columns which have more than 40 % NaN data and drop then
threshold=0.4 * len(house_train_data)
df=pd.DataFrame(len(house_train_data) - house_train_data.count(),columns=['count'])
df.index[df['count'] > threshold]


# In[ ]:


#drop columns not use 
house_train_data = house_train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
house_test_data = house_test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)


# Find all the numeric columns and replace the NaN values with 0 , and for categorical columns ,replace NaN values with 'None'.

# In[ ]:


house_train_data.select_dtypes(include=np.number).columns #will give all numeric columns ,we will remove the SalePrice column 
for col in ('MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold'):
    
    house_train_data[col] = house_train_data[col].fillna(0)
    house_test_data[col] = house_test_data[col].fillna('0')


# In[ ]:


house_train_data.select_dtypes(exclude=np.number).columns
for col in ('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'):
    
    house_train_data[col] = house_train_data[col].fillna('None')
    house_test_data[col] = house_test_data[col].fillna('None')


# In[ ]:


# Verify that there are no null values in the data set
house_train_data[house_train_data.isnull().any(axis=1)]


# In[ ]:


house_test_data[house_test_data.isnull().any(axis=1)]


# In[ ]:


#columns before hot encoding
house_train_data.columns


# In[ ]:


# Combining the two datasets and then doing One Hot Encoding on the combined dataset.
train=house_train_data
test=house_test_data

#Assigning a flag to training and testing dataset for segregation after OHE .
train['train']=1 
test['train']=0

#Combining training and testing dataset
combined=pd.concat([train,test])


# In[ ]:


#Applying One Hot Encoding to categorical data
ohe_data_frame=pd.get_dummies(combined, 
                           columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'],
      )


# In[ ]:


#columns after hot encoding with additional columns created
house_train_data.columns


# In[ ]:


#Splitting the combined dataset after doing OHE .
train_df=ohe_data_frame[ohe_data_frame['train']==1]
test_df=ohe_data_frame[ohe_data_frame['train']==0]

train_df.drop(['train'],axis=1,inplace=True)             #Drop the Flag(train) coloumn from training dataset
test_df.drop(['train','SalePrice'],axis=1,inplace=True)     #Drop the Flag(train),Label(SalePrice) coloumn from test dataset


# In[ ]:


# re-assign values back
house_train_data=train_df
house_test_data=test_df


# In[ ]:


house_train_data.head()


# Data Cleaning is now complete We can now use our data to build our models

# **Prepare data**

# In[ ]:


# features data
X_train = house_train_data.drop('SalePrice', axis=1)

# labels
Y_train = house_train_data['SalePrice']
Y_train = np.log(Y_train+1)

# test data
X_test = house_test_data


# In[ ]:


# find outliner then remove it
from sklearn.linear_model import Ridge, ElasticNet
rr = Ridge(alpha=10)
rr.fit(X_train, Y_train)
np.sqrt(-cross_val_score(rr, X_train, Y_train, cv=5, scoring="neg_mean_squared_error")).mean()


# In[ ]:


y_pred = rr.predict(X_train)
resid = Y_train - y_pred
mean_resid = resid.mean()
std_resid = resid.std()
z = (resid - mean_resid) / std_resid
z = np.array(z)
outliers1 = np.where(abs(z) > abs(z).std() * 3)[0]
outliers1


# In[ ]:


#delete outliners
X_train = X_train.drop([30, 88, 142, 277, 328, 410, 462, 495, 523, 533, 581, 588, 628, 632, 681, 688, 710, 714, 728, 774, 812, 874, 898, 916, 968, 970, 1181, 1182, 1298, 1324, 1383, 1423, 1432, 1453])
Y_train = Y_train.drop([30, 88, 142, 277, 328, 410, 462, 495, 523, 533, 581, 588, 628, 632, 681, 688, 710, 714, 728, 774, 812, 874, 898, 916, 968, 970, 1181, 1182, 1298, 1324, 1383, 1423, 1432, 1453])
print ("dropped outliners")


# **Model Building**

# #### Gradient Boosting for regression.
# 
# GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.

# The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter. 
# 
# Hyper-parameters are parameters that are not directly learnt within estimators. I
# Exhaustive search over specified parameter values for an estimator.
# Important members are fit, predict.

# For this model implementation, let's use Gradient Boosting for regression and find the best parameter for GBR using GridSearchCV as the optimal approach.

# In[ ]:


#GardientBoosting
params = {'n_estimators': 400, 'max_depth': 5, 'min_samples_split': 2,'learning_rate': 0.09, 'loss': 'ls'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)


# In[ ]:


gbr_model.score(X_train, Y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
#cross validation
np.sqrt(-cross_val_score(gbr_model, X_train, Y_train, cv=5, scoring="neg_mean_squared_error")).mean()


# In[ ]:


#Predicting the SalePrice for the test data
y_grad_predict = gbr_model.predict(X_test)
y_grad_predict=np.exp(y_grad_predict)-1
#print(y_grad_predict)


# In[ ]:


#Submission 
#my_submission = pd.DataFrame({'Id': test_input.Id, 'SalePrice': y_grad_predict})
#print(my_submission)
sample_submission["SalePrice"] = y_grad_predict
sample_submission.to_csv("house_price_submission2.csv", index=False)


# In[ ]:


#pre = pd.read_csv("house_price_submission.csv")
#pre.head()


# In[ ]:


print ("the end")


# reference: https://scikit-learn.org/stable/
