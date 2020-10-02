#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1)


# In[ ]:


#importing data from csv file using pandas
train=pd.read_csv('../input/home-data-for-ml-course/train.csv')
test=pd.read_csv('../input/home-data-for-ml-course/test.csv')

train.head()


# Data Visualization

# In[ ]:


#lets create scatterplot of GrLivArea and SalePrice
sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)


# In[ ]:


#as per above plot we can see there are two outliers which can affect on out model,lets remove those outliers
train=train.drop(train.loc[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,0)
train.reset_index(drop=True, inplace=True)


# In[ ]:


#lest we how its look after removing outliers
sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)


# In[ ]:


#lets create heatmap first of all lest see on which feature SalePrice is dependent
corr=train.drop('Id',1).corr().sort_values(by='SalePrice',ascending=False).round(2)
print(corr['SalePrice'])


# In[ ]:


#here we can see SalePrice mostly dependent on this features OverallQual,GrLivArea,TotalBsmtSF,GarageCars,1stFlrSF,GarageArea 
plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);


# In[ ]:


#now lets create heatmap for top 10 correlated features
cols =corr['SalePrice'].head(10).index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#lets see relation of 10 feature with SalePrice through Pairplot
sns.pairplot(train[corr['SalePrice'].head(10).index])


# In[ ]:


#lets store number of test and train rows
trainrow=train.shape[0]
testrow=test.shape[0]


# In[ ]:


#copying id data
testids=test['Id'].copy()


# In[ ]:


#copying sales priece
y_train=train['SalePrice'].copy()


# In[ ]:


#combining train and test data
data=pd.concat((train,test)).reset_index(drop=True)
data=data.drop('SalePrice',1)


# In[ ]:


#dropping id columns
data=data.drop('Id',axis=1)


# Missing Data

# In[ ]:


#checking missing data
missing=data.isnull().sum().sort_values(ascending=False)
missing=missing.drop(missing[missing==0].index)
missing


# In[ ]:


#PoolQC is quality of pool but mostly house does not have pool so putting NA
data['PoolQC']=data['PoolQC'].fillna('NA')
data['PoolQC'].unique()


# In[ ]:


#MiscFeature: mostly house does not have it so putting NA
data['MiscFeature']=data['MiscFeature'].fillna('NA')
data['MiscFeature'].unique()


# In[ ]:


#Alley,Fence,FireplaceQu: mostly house does not have it so putting NA
data['Alley']=data['Alley'].fillna('NA')
data['Alley'].unique()

data['Fence']=data['Fence'].fillna('NA')
data['Fence'].unique()

data['FireplaceQu']=data['FireplaceQu'].fillna('NA')
data['FireplaceQu'].unique()


# In[ ]:


#LotFrontage: all house have linear connected feet so putting most mean value
data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].dropna().mean())


# In[ ]:


#GarageCond,GarageQual,GarageFinish
data['GarageCond']=data['GarageCond'].fillna('NA')
data['GarageCond'].unique()

data['GarageQual']=data['GarageQual'].fillna('NA')
data['GarageQual'].unique()

data['GarageFinish']=data['GarageFinish'].fillna('NA')
data['GarageFinish'].unique()


# In[ ]:


#GarageYrBlt,GarageType,GarageArea,GarageCars putting 0
data['GarageYrBlt']=data['GarageYrBlt'].fillna(0)
data['GarageType']=data['GarageType'].fillna(0)
data['GarageArea']=data['GarageArea'].fillna(0)
data['GarageCars']=data['GarageCars'].fillna(0)


# In[ ]:


#BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1 
data['BsmtExposure']=data['BsmtExposure'].fillna('NA')
data['BsmtCond']=data['BsmtCond'].fillna('NA')
data['BsmtQual']=data['BsmtQual'].fillna('NA')
data['BsmtFinType2']=data['BsmtFinType2'].fillna('NA')
data['BsmtFinType1']=data['BsmtFinType1'].fillna('NA')

#BsmtFinSF1,BsmtFinSF2 
data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(0)
data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(0)


# In[ ]:


#MasVnrType,MasVnrArea
data['MasVnrType']=data['MasVnrType'].fillna('NA')
data['MasVnrArea']=data['MasVnrArea'].fillna(0)


# In[ ]:


#MSZoning 
data['MSZoning']=data['MSZoning'].fillna(data['MSZoning'].dropna().sort_values().index[0])


# In[ ]:


#Utilities
data['Utilities']=data['Utilities'].fillna(data['Utilities'].dropna().sort_values().index[0])


# In[ ]:


#BsmtFullBath
data['BsmtFullBath']=data['BsmtFullBath'].fillna(0)

#Functional
data['Functional']=data['Functional'].fillna(data['Functional'].dropna().sort_values().index[0])

#BsmtHalfBath
data['BsmtHalfBath']=data['BsmtHalfBath'].fillna(0)

#BsmtUnfSF
data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(0)


# In[ ]:


#Exterior2nd
data['Exterior2nd']=data['Exterior2nd'].fillna('NA')

#Exterior1st
data['Exterior1st']=data['Exterior1st'].fillna('NA')


# In[ ]:


#TotalBsmtSF
data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(0)


# In[ ]:


#SaleType
data['SaleType']=data['SaleType'].fillna(data['SaleType'].dropna().sort_values().index[0])


# In[ ]:


#Electrical
data['Electrical']=data['Electrical'].fillna(data['Electrical'].dropna().sort_values().index[0])


# In[ ]:


#KitchenQual
data['KitchenQual']=data['KitchenQual'].fillna(data['KitchenQual'].dropna().sort_values().index[0])


# In[ ]:


#lets check any missing remain
missing=data.isnull().sum().sort_values(ascending=False)
missing=missing.drop(missing[missing==0].index)
missing


# In[ ]:


#great no missing data


# Feature Engineering

# In[ ]:


#as we know some feature are highly co-related with SalePrice so lets create some feature using these features
data['GrLivArea_2']=data['GrLivArea']**2
data['GrLivArea_3']=data['GrLivArea']**3
data['GrLivArea_4']=data['GrLivArea']**4

data['TotalBsmtSF_2']=data['TotalBsmtSF']**2
data['TotalBsmtSF_3']=data['TotalBsmtSF']**3
data['TotalBsmtSF_4']=data['TotalBsmtSF']**4

data['GarageCars_2']=data['GarageCars']**2
data['GarageCars_3']=data['GarageCars']**3
data['GarageCars_4']=data['GarageCars']**4

data['1stFlrSF_2']=data['1stFlrSF']**2
data['1stFlrSF_3']=data['1stFlrSF']**3
data['1stFlrSF_4']=data['1stFlrSF']**4

data['GarageArea_2']=data['GarageArea']**2
data['GarageArea_3']=data['GarageArea']**3
data['GarageArea_4']=data['GarageArea']**4


# In[ ]:


#lets add 1stFlrSF and 2ndFlrSF and create new feature floorfeet
data['Floorfeet']=data['1stFlrSF']+data['2ndFlrSF']
data=data.drop(['1stFlrSF','2ndFlrSF'],1)


# In[ ]:


#MSSubClass,MSZoning
data=pd.get_dummies(data=data,columns=['MSSubClass'],prefix='MSSubClass')
data=pd.get_dummies(data=data,columns=['MSZoning'],prefix='MSZoning')
data.head()


# In[ ]:


#Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle
data=pd.get_dummies(data=data,columns=['Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle'])
data.head()


# In[ ]:


#OverallQual
data=pd.get_dummies(data=data,columns=['OverallQual'],prefix='OverallQual')


# In[ ]:


#OverallCond
data=pd.get_dummies(data=data,columns=['OverallCond'],prefix='OverallCond')


# In[ ]:


#we have remodel year data so lest one new feature home is remodeled or not
data['Remodeled']=0
data.loc[data['YearBuilt']!=data['YearRemodAdd'],'Remodeled']=1
data=data.drop('YearRemodAdd',1)
data=pd.get_dummies(data=data,columns=['Remodeled'])


# In[ ]:


#creating dummies fo all categorical data
data=pd.get_dummies(data=data,columns=['RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'])


# In[ ]:


#lets add all bath in one feature
data['Bath']=data['BsmtFullBath']+data['BsmtHalfBath']*.5+data['FullBath']+data['HalfBath']*.5
data=data.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],1)


# In[ ]:


#dummies
data=pd.get_dummies(data=data,columns=['BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd'])


# In[ ]:


#here we  has one more outliers lets replace it with 0
data.loc[data['GarageYrBlt']==2207.,'GarageYrBlt']=0


# In[ ]:


#great we have done Feature Engineering


# Feature Scalling

# In[ ]:


#lets import StandardScaler from sklearn for feature scalling
from sklearn.preprocessing import StandardScaler


# In[ ]:


#lets split data using trainrow data and scale data
x_train=data.iloc[:trainrow]
x_test=data.iloc[trainrow:]
scaler=StandardScaler()
scaler=scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[ ]:


#great we have done with feature scalling, now lets do modeling


# In[ ]:


#we will use all the basic algorithm one by one

#1.LinearRegression
from sklearn.linear_model import LinearRegression
reg_liner=LinearRegression()
reg_liner.fit(x_train_scaled,y_train)
reg_liner.score(x_train_scaled,y_train)


# In[ ]:


#2.LogisticRegression
from sklearn.linear_model import LogisticRegression
reg_logistic=LogisticRegression()
reg_logistic.fit(x_train_scaled,y_train)
print(reg_logistic.score(x_train_scaled,y_train))


# In[ ]:


#3.XGBoost one of the powefull ML Algorithm
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(x_train_scaled, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(x_train_scaled, y_train)], 
             verbose=False)
print(my_model.score(x_train_scaled,y_train))


# In[ ]:


#4.DecisionTree
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(criterion='mse',max_depth=3)
tree.fit(x_train_scaled,y_train)
print(tree.score(x_train_scaled,y_train))


# In[ ]:


#5.Support Vector Regression
from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train_scaled,y_train)
print(svm_model.score(x_train_scaled,y_train))


# In[ ]:


#6.Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
gnb=GaussianNB()
mnb=MultinomialNB()
gnb.fit(x_train_scaled,y_train)
mnb.fit(x_train,y_train)
print(gnb.score(x_train_scaled,y_train))
print(mnb.score(x_train,y_train))


# In[ ]:


#7.Random Forest
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=500)
rfr.fit(x_train_scaled,y_train)
print(rfr.score(x_train_scaled,y_train))


# great we have done some algorithms, now we can predict test data.

# <h1>Check out my other Notebooks</h1>
#     <font size='4'><a href="https://www.kaggle.com/vishalvanpariya/basic-visualization-for-beginners" target="_blank">Basic Visualization Techniques</a><br>
# <a href="https://www.kaggle.com/vishalvanpariya/data-explanation-titanic" target="_blank">Titanic EDA</a><br>
# <a href="https://www.kaggle.com/vishalvanpariya/titanic-top-6" target="_blank">Titanic Notebook</a><br>
# <a href="https://www.kaggle.com/vishalvanpariya/nlp-for-beginners" target="_blank">NLP</a><br><font>

# In[ ]:




