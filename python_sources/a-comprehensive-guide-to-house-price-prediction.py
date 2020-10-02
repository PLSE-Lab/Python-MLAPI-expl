#!/usr/bin/env python
# coding: utf-8

# ## About the problem
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# 

# **Key Take aways**
# 
# 1)Creative feature engineering 
# 
# 2)Advanced regression techniques like random forest and gradient boosting**(Which I am currently working)**

# **Loading the necessary Packages**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno #visualize the distribution of NaN values. 
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization


# **Reading the Files**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_df.head()


# In[ ]:


test_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_df.head()


# ## Data Exploration and Analysis

# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(train_df.SalePrice, color="tomato")
plt.title("Target distribution in train")
plt.ylabel("Density");


# **Observation**
# 
# Above Figure shows that the target variable is normally distributed

# **Understanding the distribution of Missing Data**

# In[ ]:


train_df.shape


# In[ ]:


isna_train = train_df.isnull().sum().sort_values(ascending=False)
isna_test = test_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


plt.subplot(2,1,1)
plt_1=isna_train[:20].plot(kind='bar')
plt.ylabel('Train Data')
plt.subplot(2,1,2)
isna_test[:20].plot(kind='bar')
plt.ylabel('Test Data')
plt.xlabel('Number of features which are NaNs')


# **Observation**
# >  From this plot, It is clear that missing data is distributed in equally in both train and test dataset. So we have to figure out a general method to handle the missing values from both the dataset
# 

# ## Handling the Missing Values

# In[ ]:


(train_df.isnull().sum()/len(train_df)).sort_values(ascending=False)[:20]


# In[ ]:


missing_percentage=(train_df.isnull().sum()/len(train_df)).sort_values(ascending=False)[:20]
print(missing_percentage.index[:5])


# Dropping the columns that have more than 30% of missing values

# In[ ]:


missing_percentage


# In[ ]:


train_df=train_df.drop(missing_percentage.index[:5],1)
test_df=test_df.drop(missing_percentage.index[:5],1)


# **Imputing the missing values**

# Finding whether the columns with missing has any pattern or wether they are normally distributed. 

# In[ ]:


missing_percentage.index[5:]


# In[ ]:


#Finding the columns whether they are categorical or numerical
cols = train_df[missing_percentage.index[5:]].columns
num_cols = train_df[missing_percentage.index[5:]]._get_numeric_data().columns
print("Numerical Columns",num_cols)
cat_cols=list(set(cols) - set(num_cols))
print("Categorical Columns:",cat_cols)


# **Numerical values distribution**

# In[ ]:


import matplotlib.pyplot as py
plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(train_df['LotFrontage'].dropna().values)
plt.subplot(332)
sns.distplot(train_df['GarageYrBlt'].dropna().values)
plt.subplot(333)
sns.distplot(train_df['MasVnrArea'].dropna().values)
py.suptitle("Distribution of data before Filling NA'S")


# **Observations**
# 
# 1)LotFrontage is normally distributed hence we can impute it with mean
# 
# 2)GarageYrBlt is skewed so we can either fill it with median
# 
# 3)MasVnrArea is skewed so we can either fill it with median
# 
# To get better results we can localize this imputations generally houses structure will be common for a particular location so I am localizing this data based on Neihbourhood

# In[ ]:


train_df['LotFrontage']=train_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train_df['GarageYrBlt']=train_df.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
train_df['MasVnrArea']=train_df.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


import matplotlib.pyplot as py
plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(train_df['LotFrontage'].values)
plt.subplot(332)
sns.distplot(train_df['GarageYrBlt'].values)
plt.subplot(333)
sns.distplot(train_df['MasVnrArea'].values)
py.suptitle("Distribution of data after Filling NA'S")


# **Just ensuring the distribution of data before and after filling the missing values remain's the same** It's always good to have distrivution same before and after imputing the missing values

# In[ ]:


test_df['LotFrontage']=test_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_df['GarageYrBlt']=test_df.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
test_df['MasVnrArea']=test_df.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# **Categorical Missing value imputation**
# 
# 

# Usually Categorical Variables are imputed with mode but it won't make sense(Houses in Newyork has different features compared to San Francisco) in all cases so in order to make them loaclized based on Neighborhood and we can impute the data

# In[ ]:


for column in cat_cols:
    train_df[column]=train_df.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.mode()))
    test_df[column]=test_df.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.mode()))


# ## Exploratory Data Analysis

# Finding the categorical and numerical feature

# In[ ]:


num_cols = train_df._get_numeric_data().columns
print("Numerical Columns",num_cols)
cat_cols=list(set(cols) - set(num_cols))
print("Categorical Columns:",cat_cols)


# **Neighbourhood wise salesprice distribution**
# 

# In[ ]:


Neighbour=train_df.groupby(['Neighborhood','YearBuilt'])['SalePrice']
Neighbour=Neighbour.describe()['mean'].to_frame()
Neighbour=Neighbour.reset_index(level=[0,1])
Neighbour=Neighbour.groupby('Neighborhood')


# In[ ]:


Neighbour_index=train_df['Neighborhood'].unique()
fig = plt.figure(figsize=(50,12))
fig.suptitle('Yearwise Trend of each Neighborhood')
for num in range(1,25):
    temp=Neighbour.get_group(Neighbour_index[num])
    ax = fig.add_subplot(5,5,num)
    ax.plot(temp['YearBuilt'], temp['mean'])
    ax.set_title(temp['Neighborhood'].unique())
    


# **Label Encoding All the Categorical variables**

# In[ ]:


#Finding the columns whether they are categorical or numerical
cols = train_df.columns
num_cols = train_df._get_numeric_data().columns
print("Numerical Columns",num_cols)
cat_cols=list(set(cols) - set(num_cols))
print("Categorical Columns:",cat_cols)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    train_df[i]=LabelEncoder().fit_transform(train_df[i].astype(str)) 
    test_df[i]=LabelEncoder().fit_transform(test_df[i].astype(str)) 


# ## Feature selection

# In[ ]:


fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(train_df.corr(),ax=ax,annot= False,linewidth= 0.02,linecolor='black',fmt='.2f',cmap = 'Blues_r')
plt.show()


# In[ ]:


#price range correlation
corr=train_df.corr()
corr=corr.sort_values(by=["SalePrice"],ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15,20))
sns.barplot(x=corr.values, y=corr.index.values);
plt.title("Correlation Plot")


# In[ ]:


#Forming a new dataset that has columns having more than 0.15 correlation
index=[]
Train=pd.DataFrame()
Y=train_df['SalePrice']
for i in range(0,len(corr)):
    if corr[i] > 0.15 and corr.index[i]!='SalePrice':
        index.append(corr.index[i])
X=train_df[index]


# **Forming New Features**
# 
# Some of the features represent the same so instead of having them individually we can combine them and get new features
# 
# Thanks for the discussion https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/106618#latest-616788

# **Description of new features**
# 
# 1)**cond*qual** - representative of Overall condition and quality.
# 
# 2)**home_age_when_sold** - Age of Home When sold
# 
# 3)**garage_age_when_sold** -Age of garage when sold
# 
# 4)**TotalSF** - Total Square Foot
# 
# 5)**total_porch_area** - Total Porch Area
# 
# 6)**Totalsqrfootage** - Total Square Foot
# 
# 7)**Total_Bathrooms** - Total Bathrooms
# 
# Now, It makes sense right? 
# These are certain things we would usually consider for when we are planning to buy a house
# 
# 

# In[ ]:


X['cond*qual'] = (train_df['OverallCond'] * train_df['OverallQual']) / 100.0
X['home_age_when_sold'] = train_df['YrSold'] - train_df['YearBuilt']
X['garage_age_when_sold'] = train_df['YrSold'] - train_df['GarageYrBlt']
X['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF'] 
X['total_porch_area'] = train_df['WoodDeckSF'] + train_df['OpenPorchSF'] + train_df['EnclosedPorch'] + train_df['3SsnPorch'] + train_df['ScreenPorch'] 
X['Totalsqrfootage'] = (train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] +train_df['1stFlrSF'] + train_df['2ndFlrSF'])
X['Total_Bathrooms'] = (train_df['FullBath'] + (0.5 * train_df['HalfBath']) +train_df['BsmtFullBath'] + (0.5 * train_df['BsmtHalfBath']))


# In[ ]:


test_df['cond*qual'] = (test_df['OverallCond'] * test_df['OverallQual']) / 100.0
test_df['home_age_when_sold'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['garage_age_when_sold'] =test_df['YrSold'] - test_df['GarageYrBlt']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF'] 
test_df['total_porch_area'] = test_df['WoodDeckSF'] +test_df['OpenPorchSF'] + test_df['EnclosedPorch'] + test_df['3SsnPorch'] + test_df['ScreenPorch'] 
test_df['Totalsqrfootage'] = (test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'] +test_df['1stFlrSF'] + test_df['2ndFlrSF'])
test_df['Total_Bathrooms'] = (test_df['FullBath'] + (0.5 * test_df['HalfBath']) +test_df['BsmtFullBath'] + (0.5 * test_df['BsmtHalfBath']))


# In[ ]:


Old_Cols=['OverallCond','OverallQual','YrSold','YearBuilt','YrSold','GarageYrBlt','TotalBsmtSF','1stFlrSF','2ndFlrSF','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath']


# In[ ]:


Final_cols=[]
for i in X.columns:
    if i not in Old_Cols and i!='SalePrice':
        Final_cols.append(i)
X=X[Final_cols]


# In[ ]:


X.columns


# **Interesting insights**

# In[ ]:


fig = plt.figure(figsize=(20,16))

plt.subplot(2, 2, 1)
plt.scatter(X['home_age_when_sold'],Y)
plt.title("Home Age Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Home Age")

plt.subplot(2, 2, 2)
plt.scatter(X['Total_Bathrooms'],Y)
plt.title("Total_Bathrooms Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Total_Bathrooms")

plt.subplot(2, 2, 3)
plt.scatter(X['TotalSF'],Y)
plt.title("TotalSF Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel('TotalSF')

plt.subplot(2, 2, 4)
plt.scatter(X[ 'cond*qual'],Y)
plt.title("House Condition Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel('cond*qual')

plt.show()


# **Distribution of Data**

# In[ ]:


X.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[ ]:


temp=pd.DataFrame()
temp=X
temp['SalePrice']=Y


# **Finding the outliers**

# In[ ]:


for i in range(0, len(temp.columns), 5):
    sns.pairplot(data=temp,
                x_vars=temp.columns[i:i+5],
                y_vars=['SalePrice'])


# In[ ]:


test_df=test_df[Final_cols]


# In[ ]:


X.isnull().sum()


# **Correlation plot for new dataset formed**

# In[ ]:


temp=X
temp["SalePrice"]=Y
#price range correlation
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(temp.corr(),ax=ax,annot= False,linewidth= 0.02,linecolor='black',fmt='.2f')
plt.show()


# **Observation**
# 
# 1)The newly formed Data has more meaningful columns compared to the original dataset
# 
# 2)**Home age when sold** and **Garage age when sold** having negative correlation with SalePrice make sense as we know, Price of building decreases as it ages.

# ## Modelling

# In[ ]:


Final_cols=[]
for i in X.columns:
    if i not in Old_Cols and i!='SalePrice':
        Final_cols.append(i)
X=X[Final_cols]


# In[ ]:


X.columns


# In[ ]:


test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


test_df.fillna(test_df.mean(), inplace=True)


# **XG Regressor**

# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X,Y)


# In[ ]:


prediction = model_xgb.predict(test_df)
pred_xgb = pd.DataFrame()
pred_xgb['Id']=test['Id']
pred_xgb['SalePrice'] = prediction
pred_xgb.to_csv("../working/submission_xgb.csv", index = False)


# **Gradient Boosting**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X,Y)


# In[ ]:


prediction = GBoost.predict(test_df)
pred_GB = pd.DataFrame()
pred_GB['Id']=test['Id']
pred_GB['SalePrice'] = prediction
pred_GB.to_csv("../working/submission_GB.csv", index = False)


# **Light GB**

# In[ ]:


import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X,Y)


# In[ ]:


prediction = model_lgb.predict(test_df)
pred_LGB = pd.DataFrame()
pred_LGB['Id']=test['Id']
pred_LGB['SalePrice'] = prediction
pred_LGB.to_csv("../working/submission_LGB.csv", index = False)


# **CatBoost Regressor**

# In[ ]:


from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='RMSE',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
cb_model.fit(X, Y)


# In[ ]:


prediction = cb_model.predict(test_df)
pred_CB = pd.DataFrame()
pred_CB['Id']=test['Id']
pred_CB['SalePrice'] = prediction
pred_CB.to_csv("../working/submission_CB.csv", index = False)


# ## Ensembling Weighted average
# 

# In[ ]:


pred_ensemble = pd.DataFrame()
pred_ensemble['Id']=test['Id']
pred_ensemble['SalePrice'] =( 0.6* pred_xgb['SalePrice'] +0.1* pred_CB['SalePrice']+0.2*pred_GB['SalePrice'] +0.1*pred_LGB['SalePrice'])
pred_ensemble.to_csv("../working/submission_ensemble.csv", index = False)


# Thanks **Serigne** for the kernal

# **I will be working extensively on the coming days on the modelling part to get good results**

# Any suggestions please let me know through your comments
