#!/usr/bin/env python
# coding: utf-8

# Piyush Singla Kaggle link -> https://www.kaggle.com/mpiyu20/account
# Nikhil Singla Kaggle link -> https://www.kaggle.com/nikhilsharma4

# ## Housing price prediction using ANN

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset

# In[ ]:


df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.columns


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# Processing missing values

# In[ ]:


df_train.isnull().sum().sort_values(ascending =False).head(20)


# Percentage of missing data

# In[ ]:


df_train.isnull().sum().sort_values(ascending =False).head(20)/len(df_train)


# Removing the parameters with missing data 

# In[ ]:


df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1,inplace=True)
df_test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage',],axis=1,inplace=True)


# Treatment of left parameters having missing values

# In[ ]:


import random
missing_values = df_train.columns[df_train.isna().any()].to_list()
for each in missing_values:
    if (df_train[each].dtypes =='float64'):
        minimum= int(df_train[each].quantile(0.25))
        maximum= int(df_train[each].quantile(0.75))
        A=df_train[df_train[each].isnull()].index.tolist()
        for i in A:
            df_train.loc[i,each]=random.randint(minimum,maximum)
        df_train[each]=pd.to_numeric(df_train[each])
   

    elif(df_train[each].dtypes == 'object'):
        if ('True' in str(df_train[each].str.contains('No').unique().tolist())):
            df_train[each].fillna('No',inplace=True)
        elif('True' in str(df_train[each].str.contains('None').unique().tolist())):
            df_train[each].fillna('None',inplace=True)
        elif('True' in str(df_train[each].str.contains('Unf').unique().tolist())):
            df_train[each].fillna('Unf',inplace=True)
        else:
            A=df_train[df_train[each].isnull()].index.tolist()
            unique = df_train[each].unique().tolist()
            unique=pd.Series(unique).dropna().tolist()
            for i in A:
                df_train.loc[i,each]=random.choice(unique)


missing_values = df_test.columns[df_test.isna().any()].to_list()
for each in missing_values:
    if (df_test[each].dtypes =='float64'):
        minimum= int(df_test[each].quantile(0.25))
        maximum= int(df_test[each].quantile(0.75))
        A=df_test[df_test[each].isnull()].index.tolist()
        for i in A:
            df_test.loc[i,each]=random.randint(minimum,maximum)
        df_test[each]=pd.to_numeric(df_test[each])
   

    elif(df_test[each].dtypes == 'object'):
        if ('True' in str(df_test[each].str.contains('No').unique().tolist())):
            df_test[each].fillna('No',inplace=True)
        elif('True' in str(df_test[each].str.contains('None').unique().tolist())):
            df_test[each].fillna('None',inplace=True)
        elif('True' in str(df_test[each].str.contains('Unf').unique().tolist())):
            df_test[each].fillna('Unf',inplace=True)
        else:
            A=df_test[df_test[each].isnull()].index.tolist()
            unique = df_test[each].unique().tolist()
            unique=pd.Series(unique).dropna().tolist()
            for i in A:
                df_test.loc[i,each]=random.choice(unique)


# Removing the Id column

# In[ ]:


df_train.drop(['Id'],axis=1,inplace=True)
df_test.drop(['Id'],axis=1,inplace=True)


# Now checking the correlation between continuous parameters and plotting the heatmap

# # Exploratory data analysis

# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(df_train.corr(),annot=True,cmap='coolwarm')


# In[ ]:


plt.figure(figsize=(10,5))
df_train.corr()['SalePrice'].sort_values().drop('SalePrice').plot(kind='bar')


# For continous parameters, the most correlated values to Sale Price are given in the above plot

# Now, analysing some most correlated parameters using Seaborn 

# In[ ]:


fig = plt.figure(figsize=(15,10));   
ax1 = fig.add_subplot(3,4,1);  
ax2 = fig.add_subplot(3,4,2);
ax3 = fig.add_subplot(3,4,3);  
ax4 = fig.add_subplot(3,4,4);
ax5 = fig.add_subplot(3,4,5);  
ax6 = fig.add_subplot(3,4,6);
ax7 = fig.add_subplot(3,4,7);  
ax8 = fig.add_subplot(3,4,8);
ax9 = fig.add_subplot(3,4,9);  
ax10 = fig.add_subplot(3,4,10);
ax11 = fig.add_subplot(3,4,11);  
ax12 = fig.add_subplot(3,4,12);

sns.boxplot("OverallQual", "SalePrice", data=df_train,ax=ax1)
sns.scatterplot("GrLivArea", "SalePrice", data=df_train, ax=ax2)
sns.boxplot("GarageCars", "SalePrice", data=df_train,ax=ax3)
sns.scatterplot("GarageArea", "SalePrice", data=df_train, ax=ax4)
sns.scatterplot("TotalBsmtSF", "SalePrice", data=df_train,ax=ax5)
sns.scatterplot("1stFlrSF", "SalePrice", data=df_train, ax=ax6)
sns.boxplot("FullBath", "SalePrice", data=df_train,ax=ax7)
sns.boxplot("TotRmsAbvGrd", "SalePrice", data=df_train, ax=ax8)
sns.scatterplot("YearBuilt", "SalePrice", data=df_train,ax=ax9)
sns.scatterplot("YearRemodAdd", "SalePrice", data=df_train, ax=ax10)
sns.boxplot("MasVnrType", "SalePrice", data=df_train,ax=ax11)
sns.boxplot("Fireplaces", "SalePrice", data=df_train, ax=ax12)
plt.tight_layout()


# #### Analysis from the above plot
# Overall Quality
# - SalePrice follows a linear relation with Overall quality of the house(as obvious)
# - When OverallQual = 10, the range of SalePrice is between 350,000 to 500,000
# 
# Garage Cars
# - Evidenty, if the house has size of garage for accomodating 3 cars, the selling price will be more  
# - There are only 5 houses having 4 car space garage, so we can't say much about it, but the prices are lower than 3 car space houses
# 
# FullBath, Total Rooms above ground and Fireplaces
# - More number of these parameter in the house, more is the selling price of the house
# - When the parameter starts to increase the variation in SalePrice also increases because it starts depending on other paramters more to make an accurate prediction.  
# 
# MasVnrType
# - Masonry veneer Stone type has more selling price followed by brick face and common brick
# 
# Scatter Plots
# - all the scatterplots including GRLivArea, GarageArea, 1stFlrSF and TotalBsmtSF follows a linear trend with Sale Price
# 
# 

# In[ ]:


fig = plt.figure(figsize=(15,10));   
ax1 = fig.add_subplot(2,1,1);  
ax2 = fig.add_subplot(2,1,2);
sns.distplot(df_train['YearBuilt'],bins=50,color='black',ax=ax1)
sns.distplot(df_train['YearRemodAdd'],bins=50,color='black',ax=ax2)


# Most of the houses were build in the 2000's and big chunk in 1950's and 1960's. Most of the houses were remodelled/rebuilt in 1950

# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(x='TotRmsAbvGrd',y='GrLivArea',data=df_train)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='FullBath',y='2ndFlrSF',data=df_train,hue='HalfBath',palette="BuGn_r")


# Number of half baths in a house is independent of full baths. As, mostly in all houses the number of half baths is 1

# In[ ]:


plt.figure(figsize=(10,5))
df_train.corr()['OverallQual'].sort_values().drop(['OverallQual','SalePrice']).plot(kind='bar')


# Above plot define the parameters on which the overall quality of the house depends, that are Garage living area, Garage cars, year it was built and No. of full baths

# #### Analysis on catagorical features with dtype object

# In[ ]:


catogorical_features_ = np.array([df_train.columns[df_train.dtypes == 'object'].to_list()])
catogorical_features_


# In[ ]:


df_train['Utilities'].value_counts()


# In[ ]:


# All the records in utilities are mostly AllPub 
df_train.drop('Utilities',axis=1,inplace=True)
df_test.drop('Utilities',axis=1,inplace=True)


# In[ ]:


fig = plt.figure(figsize=(20,15));   
ax1 = fig.add_subplot(4,4,1);  
ax2 = fig.add_subplot(4,4,2);
ax3 = fig.add_subplot(4,4,3);  
ax4 = fig.add_subplot(4,4,4);
ax5 = fig.add_subplot(4,4,5);  
ax6 = fig.add_subplot(4,4,6);
ax7 = fig.add_subplot(4,4,7);  
ax8 = fig.add_subplot(4,4,8);
ax9 = fig.add_subplot(4,4,9);  
ax10 = fig.add_subplot(4,4,10);
ax11 = fig.add_subplot(4,4,11);  
ax12 = fig.add_subplot(4,4,12);
ax13 = fig.add_subplot(4,4,13);  
ax14 = fig.add_subplot(4,4,14);
ax15 = fig.add_subplot(4,4,15);  
ax16 = fig.add_subplot(4,4,16);

sns.boxplot(x="LotShape",y= "SalePrice", data=df_train,ax=ax1)
sns.boxplot("SaleCondition", "SalePrice", data=df_train, ax=ax2)
sns.boxplot("LandSlope", "SalePrice", data=df_train,ax=ax3)
sns.boxplot("Condition1", "SalePrice", data=df_train, ax=ax4)
sns.boxplot("BldgType", "SalePrice", data=df_train,ax=ax5)
sns.boxplot("HouseStyle", "SalePrice", data=df_train, ax=ax6)
sns.boxplot("RoofStyle", "SalePrice", data=df_train,ax=ax7)
sns.boxplot("Exterior1st", "SalePrice", data=df_train, ax=ax8)
sns.boxplot("Exterior2nd", "SalePrice", data=df_train,ax=ax9)
sns.boxplot("ExterQual", "SalePrice", data=df_train, ax=ax10)
sns.boxplot("ExterCond", "SalePrice", data=df_train,ax=ax11)
sns.boxplot("Foundation", "SalePrice", data=df_train, ax=ax12)
sns.boxplot("HeatingQC", "SalePrice", data=df_train,ax=ax13)
sns.boxplot("CentralAir", "SalePrice", data=df_train, ax=ax14)
sns.boxplot("KitchenQual", "SalePrice", data=df_train,ax=ax15)
sns.boxplot("SaleType", "SalePrice", data=df_train, ax=ax16)
plt.tight_layout()


# Lotshape
# - Houses with IR3 type of lots have maximum SalePrice
# 
# SaleConditions
# - It is an important attribute as vatiation in SalePrice is more in it with the change in the parameters.
# - Houses with partial salescondition has maximum sales.
# 
# LandSlope does not play an integral part in estimation of SalePrice
# 
# Condition 1
# - So people prefer to pay more for the lots within the range of 200' to the north-south rail-road. 
# - People prefer to pay less for the lots adjacent to arterial street.
# 
# HouseStyle
# - 2 story and 1 story housing styles have better sales price than the other ones.
# - The other housing styles follow the constant trends for the sales price.
# 
# ExterQual
# - People are willing to pay more for the houses having excellent external quality.
# 
# SaleType
# - The salesprice of the houses which are freshly constructed are maximum. 

# In[ ]:


df_train['Foundation'].value_counts()


# In[ ]:


fig = plt.figure(figsize=(20,10));   
ax1 = fig.add_subplot(1,2,1);  
ax2 = fig.add_subplot(1,2,2);
sns.boxplot("Exterior1st", "SalePrice", data=df_train, ax=ax1)
sns.boxplot("Exterior2nd", "SalePrice", data=df_train,ax=ax2)
plt.tight_layout()


# In[ ]:


fig = plt.figure(figsize=(15,10));   
ax1 = fig.add_subplot(2,1,1);  
ax2 = fig.add_subplot(2,1,2);
sns.distplot(df_train['1stFlrSF'],bins=30,color='black',ax=ax1)
sns.distplot(df_train['2ndFlrSF'],bins=10,color='black',ax=ax2)


# - 1st floor sqaure feet has a normal distribution
# - Houses with 2nd floor has around 0 to 250 square feet of area with some houses of range of 500 to 1000 square feet

# #### Converting catagorical features into dummy variable using pandas get dummies method

# In[ ]:


catogorical_features_ = np.delete(catogorical_features_,np.where(catogorical_features_=='Utilities'))


# In[ ]:


test_match=[]
for i,feature in enumerate(catogorical_features_): 
    test_match.append( (feature,(df_train[feature].nunique()  -  df_test[feature].nunique())))
    if (df_train[feature].nunique()  -  df_test[feature].nunique()) != 0:
        df_train.drop(feature,axis=1,inplace=True)
        df_test.drop(feature,axis=1,inplace=True)


# In[ ]:


print(test_match)


# In[ ]:


catogorical_features_ = np.array([df_train.columns[df_train.dtypes == 'object'].to_list()])
dummies = []
concat_dummies=[]
for i,feature in enumerate(catogorical_features_[0]):
    dummies.append(pd.get_dummies(df_train[feature],drop_first=True))
    df_train = pd.concat([df_train,dummies[i]],axis=1) 


# In[ ]:


df_train.drop(['MSZoning', 'Street', 'LotShape', 'LandContour','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
               'BldgType', 'RoofStyle','MasVnrType', 'ExterQual','Foundation',  'HeatingQC', 'CentralAir',
         'KitchenQual', 'Functional', 'GarageFinish','PavedDrive', 'SaleType', 'SaleCondition','ExterCond',
               'GarageCond',
               'GarageType','GarageYrBlt','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual'],
              axis=1,inplace=True)


# In[ ]:


catogorical_features_ = np.array([df_test.columns[df_test.dtypes == 'object'].to_list()])
dummies = []
concat_dummies=[]
for i,feature in enumerate(catogorical_features_[0]):
    dummies.append(pd.get_dummies(df_test[feature],drop_first=True))
    df_test = pd.concat([df_test,dummies[i]],axis=1) 


# In[ ]:


df_test.drop(['MSZoning', 'Street', 'LotShape', 'LandContour','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
               'BldgType', 'RoofStyle','MasVnrType', 'ExterQual','Foundation',  'HeatingQC', 'CentralAir',
         'KitchenQual', 'Functional', 'GarageFinish','PavedDrive', 'SaleType', 'SaleCondition','ExterCond',
             'GarageCond','GarageType','GarageYrBlt','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual'],axis=1,inplace=True)


# # Artificial Neural Network

# In[ ]:


X_train = np.array(df_train.drop('SalePrice',axis=1))
y_train = np.array(df_train['SalePrice'])
X_test = np.array(df_test)


# In[ ]:


print('Shape of X_train {} \nShape of y_test {}\nShape of X_test {}'.format(X_train.shape,y_train.shape,X_test.shape))


# #### Feature Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


# In[ ]:


y_train = mms.fit_transform(y_train.reshape(-1,1))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[ ]:


regressor = Sequential()
regressor.add(Dense(units=512,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))


# In[ ]:


regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])


# In[ ]:


regressor.fit(X_train,y_train,epochs=100,batch_size=50)


# In[ ]:


losses = regressor.history.history
losses = np.array(pd.DataFrame(losses))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('loss')


# ## Tuning the ANN

# Commenting the code for timing constraint

# In[ ]:


'''
def build_classifier(optimizer,units1,units2,units3,units4):
    regressor = Sequential()
    regressor.add(Dense(units=units1,activation='relu',kernel_initializer='uniform'))
    regressor.add(Dense(units=units2,activation='relu',kernel_initializer='uniform'))
    regressor.add(Dense(units=units3,activation='relu',kernel_initializer='uniform'))
    regressor.add(Dense(units=units4,activation='relu',kernel_initializer='uniform'))
    regressor.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
    regressor.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mean_squared_error'])
    return regressor

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

regressor = KerasRegressor(build_fn=build_classifier)
parameters = {'batch_size':[10,15,25,32],
             'epochs':[100,300,500],
             'optimizer':['adam','rmsprop'],
             'units1':[512,256],
             'units2':[256,128],
             'units3':[256,128],
             'units4':[256,128,64]}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 3)
grid_search = grid_search.fit(X_train, y_train)
'''


# In[ ]:


'''print(grid_search.best_score_)
print('\n')
print(grid_search.best_params_)'''


# In[ ]:


Best = {'batch_size': 15, 'epochs': 500, 'optimizer': 'adam', 'units1': 512, 'units2': 128, 'units3': 128, 'units4': 64}


# In[ ]:


#from keras.models import load_model
#regressor.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
#regressor1 = load_model('my_model.h5')


# ## Training the ANN on best parameters

# In[ ]:


regressor = Sequential()
regressor.add(Dense(units=512,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))
regressor.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])


# For furthur fine tuning add dropout layers and more layers to the model

# In[ ]:


regressor.fit(X_train,y_train,epochs=500,batch_size=15)


# In[ ]:


losses = regressor.history.history
losses = np.array(pd.DataFrame(losses))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('loss')


# In[ ]:


regressor.summary()


# ## Prediction on test set

# In[ ]:


y_pred = regressor.predict(X_test) 
y_pred_original = mms.inverse_transform(y_pred.reshape(-1,1))
y_pred_original = y_pred_original.tolist()
y_pred_original = [pred for i in y_pred_original for pred in i]


# In[ ]:


test_set =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.DataFrame({'Id': test_set['Id'],'SalePrice': y_pred_original})


# In[ ]:


#submission.to_csv('submission.csv', index=False)


# In[ ]:





# 1. **Furthur Imrovement on our submission**
# 
# Training more regressors and combining it with previous one's
# 
# Take more missing values into consideration
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators= 100)
regressor1.fit(X_train,y_train)


# In[ ]:


y_pred1 = regressor1.predict(X_test) 
y_pred_original1 = mms.inverse_transform(y_pred1.reshape(-1,1))
y_pred_original1 = y_pred_original1.tolist()
y_pred_original1 = [pred for i in y_pred_original1 for pred in i]


# In[ ]:


y_pred_final=[]
for i in range(0,1459):
    y_pred_final.append((y_pred_original[i]*0.5)+(y_pred_original1[i]*0.5))


# In[ ]:


submission = pd.DataFrame({'Id': test_set['Id'],'SalePrice': y_pred_final})


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




