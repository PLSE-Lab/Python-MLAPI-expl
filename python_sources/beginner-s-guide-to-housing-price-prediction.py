#!/usr/bin/env python
# coding: utf-8

# ## **House Prices: Advanced Regression Techniques**
#  
#  **Objective:** The purpose of this project is to predict the SalesPrice of houses based on their engineering factors.For Example- No of Garages,Carpet Area,etc. We need to find out what all factors actually influence the price of the houses.

# In[ ]:


#Let us begin with importing with all the necessary library that we'll be using

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #used for data visulization
import seaborn as sns  #used for data visulization

import warnings  #This will ignore all the unnecessary warnings                
warnings.simplefilter("ignore")

import os  #Display the list of files present in the directory.         
print(os.listdir("../input"))

#It is used display the visulization into this notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Loading Data
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()


# ## EDA: Exploratory Data Analysis
# 
# This step is all about understanding the data. Let us now analyse our data and look for areas where we need to work on. Upon investigating I have found couple of interesting things about our data.

# In[ ]:


#We can see that our target column (SalePrice) has a mean value of 180921.
#We can also note the minimum and maximum value of that column
train.describe()


# In[ ]:


#Let us now create a histogram of our Target column.
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.distplot(train['SalePrice'],bins=30,kde=True)


# **Observation:** We can see that the SalePrice reaches a max peak of somewhere around $730000

# In[ ]:


#Let us have a look at all the other columns in our dataset 
train.info()


# In[ ]:


#Lets check whether we have any duplicate data
sum(train['Id'].duplicated()),sum(test['Id'].duplicated())


# In[ ]:


# Drop Id column from train dataset
train.drop(columns=['Id'],inplace=True)


# Let us now Categorize out data based on the datatypes

# In[ ]:


#categorize the data:

num_cols=[var for var in train.columns if train[var].dtypes != 'O']
cat_cols=[var for var in train.columns if train[var].dtypes != 'int64' and train[var].dtypes != 'float64']

print('No of Numerical Columns: ',len(num_cols))
print('No of Categorical Columns: ',len(cat_cols))
print('Total No of Cols: ',len(num_cols+cat_cols))


# ## **Missing Data** 
# 
# Let us now check for columns in our train dataframe which has missing data.
# 

# In[ ]:


#Lets create a heatmap to see which all columns has null values
plt.figure(figsize=(30,12))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')


# **Observation:** We can see there are few columns will null values
# 
# Let us now get details of these columns.

# In[ ]:


#Columns with null values in the Train dataFrame
var_with_na=[var for var in train.columns if train[var].isnull().sum()>=1 ]

for var in var_with_na:
    print(var, np.round(train[var].isnull().mean(),3), '% missing values')


# In[ ]:


#Columns with null values in the Test dataFrame
var_with_na2=[var for var in test.columns if test[var].isnull().sum()>=1 ]

for var in var_with_na2:
    print(var, np.round(test[var].isnull().mean(),3), '% missing values')


# **Observation:** There are many columns with null values in both the dataframes. We'll try to fill these null values with suitables values. Columns with huge null values can be dropped. 

# In[ ]:


#I have categorized the missing columns based on the datatypes and no of null values.

missing_cols=['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
           'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']
num_list=['LotFrontage','MasVnrArea','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
           'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
drop_cols=['PoolQC','Fence','MiscFeature','GarageYrBlt','Alley']


# In[ ]:


combine=[train,test]

for df in combine:
    # Fill missing values in cetgorical variables with None
    for col in missing_cols:
        df[col]=df[col].fillna('None')
    # Fill missing values in numerical variables with 0
    for col in num_list:
        df[col]=df[col].fillna(0)
    # Drop columns with large number of missing values
    df.drop(columns=drop_cols,inplace=True) 


# Now that we have replaced all the null values and dropped columns with major null values. Lets us see our heatmap again!

# In[ ]:


plt.figure(figsize=(30,12))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')  


# **Observations:** It seems to be there is still some null values left. Lets check again what all columns are left out...

# In[ ]:


#Columns with null values in the Train dataFrame
var_with_na=[var for var in train.columns if train[var].isnull().sum()>=1 ]

for var in var_with_na:
    print(var, np.round(train[var].isnull().mean(),3), '% missing values')


# In[ ]:


#Columns with null values in the Test dataFrame
var_with_na2=[var for var in test.columns if test[var].isnull().sum()>=1 ]

for var in var_with_na2:
    print(var, np.round(test[var].isnull().mean(),3), '% missing values')


# Lets fill up these missing values one by one.

# In[ ]:


train['Electrical'].fillna('None', inplace=True)


# In[ ]:


# Fill missing values in categorical variables in test dataset
mode_list=['Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType','MSZoning']
for col in mode_list:
    mode=test[col].mode()
    test[col]=test[col].fillna(mode[0])


# In[ ]:


train.isnull().sum().max(),test.isnull().sum().max()


# In[ ]:


plt.figure(figsize=(30,12))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan') 


# Now we can say there are no null values in both of our dataFrame.

# Let us try the find the correlation between each columns from the Train dataframe.

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.corr(), vmax=.8, square=True,cbar=True,cmap='RdGy');


# From this whole correlation map, I have handpicked some columns which I felt is important when it comes to influencing SalePrice.

# In[ ]:


# Plot the correlation matrix
imp_list=['SalePrice','OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea' ,'GarageArea',
         'GarageCars','LotArea','PoolArea','Fireplaces','1stFlrSF','FullBath']
corrmat = train[imp_list].corr()
plt.subplots(figsize=(10, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10});


# Observation: Correlation wrt Saleprice

# In[ ]:


#Lets plot a boxplot on Overall Quality Vs The salesprice
plt.figure(figsize=(10,8))
sns.boxplot(x='OverallQual',y='SalePrice',data=train)


# **Observation:** There is nothing new here, the price increases with increase in quality

# In[ ]:


#Lets plot a barplot on Built Year Vs The salesprice
plt.figure(figsize=(30,8))
sns.barplot(x=train['YearBuilt'],y=train['SalePrice'],palette='viridis_r')


# **Observation:** There is a gradual increase in saleprice with time, apart from some few spikes inbetween due to inflation or something.

# In[ ]:


#Lets plot a scatterplot on Total Basement Area Vs salesprice
plt.figure(figsize=(10,8))
sns.scatterplot(x=train['TotalBsmtSF'],y=train['SalePrice'], alpha=0.8)


# **Observation:** We can see some outliner present. lets remove them

# In[ ]:


index=train[train['TotalBsmtSF']>4000].index
train.drop(index,inplace=True)


# In[ ]:


#Lets plot a scatterplot on Total Basement Area Vs salesprice
plt.figure(figsize=(10,8))
sns.scatterplot(x=train['TotalBsmtSF'],y=train['SalePrice'], alpha=0.8)


# **Observation:** Most of the houses have a basement area of 500-1500

# In[ ]:


#Lets plot a Jointplot on Carpet Area Vs The salesprice
plt.figure(figsize=(10,8))
sns.scatterplot(x=train['GrLivArea'],y=train['SalePrice'])


# In[ ]:


# Remove outliers from Above Ground Area
index=train[train['GrLivArea']>4000].index
train.drop(index,inplace=True)


# In[ ]:


#Lets plot a Jointplot on Carpet Area Vs The salesprice
plt.figure(figsize=(10,8))
sns.scatterplot(x=train['GrLivArea'],y=train['SalePrice'])


# **Observation:** We can see the GrLivArea and SalePrice have a positive corelation and a large chunk of data is between 800 - 1500 sq feet.

# In[ ]:


#Lets plot a boxplot on Garage Capacity Vs The salesprice
plt.figure(figsize=(10,8))
sns.boxplot(x=train['GarageCars'],y=train['SalePrice'])


# >** Observation:**
# We can see that with the increase in Garage capacity saleprice also increases. But there is something wrong with the Garagecars 4. So lets remove it for simplicity and correct prediction.

# In[ ]:


# Removing outliers manually (More than 4-cars, less than $300k)
train = train.drop(train[(train['GarageCars']>3) & (train['SalePrice']<300000)].index).reset_index(drop=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x=train['GarageCars'],y=train['SalePrice'])


# This looks much better...

# In[ ]:


#Lets plot a line plot on Lot Area vs Salesprice
plt.figure(figsize=(10,6))
sns.lineplot(x=train['LotArea'],y=train['SalePrice'],palette='viridis_r')


# In[ ]:


# Bivariate plot of Quality vs. Area
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
sns.boxplot(data=train,y='TotalBsmtSF',x='OverallQual');
plt.subplot(3,1,2)
sns.boxplot(data=train,y='GarageArea',x='OverallQual');
plt.subplot(3,1,3)
sns.boxplot(data=train,y='GrLivArea',x='OverallQual');


# Observation: We can see that all the 3 factors have a positive correlation wrt to saleprice

# In[ ]:


#Lets plot a scatterplot on Firstfloor sqFt  Vs The salesprice
plt.figure(figsize=(10,8))
sns.jointplot(x=train['1stFlrSF'],y=train['SalePrice'],kind='hex')


# **Observation:** We can see that there is a positive correlation between the 1st Floor carpet area and the saleprice. Seems to be most people prefer to buy houses with carpet area between 800-1200 sq feet.

# ## Model And Prediction

# In[ ]:


# Drop object features
for df in combine:
    df.drop(columns=['MSZoning','SaleCondition','SaleType','PavedDrive','GarageCond','GarageQual','GarageFinish',
                    'GarageType','FireplaceQu','Functional','KitchenQual','Heating','HeatingQC','CentralAir',
                     'Electrical','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',
                     'BsmtFinType1','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Street',
                     'LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
                     'Condition2','BldgType','HouseStyle','BsmtFinType2' ],inplace=True)


# In[ ]:


# Merge the two datasets
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test))


# In[ ]:


# Get dummy variables
all_data=pd.get_dummies(all_data)


# In[ ]:


# Seperate the combined dataset into test and train data
test=all_data[all_data['SalePrice'].isnull()]
train=all_data[all_data['Id'].isnull()]


# In[ ]:


# Check if the new and old sizes are equal
assert train.shape[0]==ntrain
assert test.shape[0]==ntest


# In[ ]:


# Drop extra columns
test.drop(columns='SalePrice',inplace=True)
train.drop(columns='Id',inplace=True)
test['Id']=test['Id'].astype(int)


# In[ ]:


X_train=train.drop(columns='SalePrice')
Y_train=train['SalePrice']
X_test=test.drop(columns='Id')


# In[ ]:


Y_train.head()


# In[ ]:


#from sklearn.ensemble import RandomForestClassifier


# In[ ]:


'''
# Apply Random Forest
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
'''


# In[ ]:


from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_cv.fit(X_train, Y_train)
ridge_cv_preds=ridge_cv.predict(X_test)


# In[ ]:


# Apply XGBRegressor
import xgboost as xgb

xgb = xgb.XGBRegressor(n_estimators=340, learning_rate=0.08, max_depth=2)
xgb.fit(X_train,Y_train)
Y_pred = xgb.predict(X_test)


# In[ ]:


predictions = ( ridge_cv_preds + Y_pred )/2


# In[ ]:


final_df = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": predictions
    })

solution = pd.DataFrame(final_df)
solution.head()


# In[ ]:


# Save the dataframe to a csv file
final_df.to_csv('submission.csv',index=False)


# In[ ]:




