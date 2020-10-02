#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Data Loading**

# Loading the datasets (train.csv and test.csv) 

# In[ ]:


house_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


house_train.head()


# In[ ]:


house_test.head()


# In[ ]:


print('Shape of the taining set') 
print(house_train.shape)
print('\nShape of the testing set') 
print(house_test.shape)


# In[ ]:


print('Information about the taining set\n') 
house_train.info()
print('\nInformation about the testing set\n') 
house_test.info()


# # **Data Cleaning**

# Data cleaning has to be done by identifying number of missing values within the datasets.
# The training set has 1460 values and the testing set has 1459 values.
# We need to find the precentage of missing values present.

# In[ ]:


data = [house_train, house_test]

for dataset in data:
    percentage = round(((dataset.isnull().sum()*100)/(dataset.shape[0])),4).sort_values(ascending=False)
    print(percentage.head(20),'\n')


# From the above we realise that there are 6 features that have most missing values, thus we drop these features.  

# In[ ]:


for dataset in data:
    dataset.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1,inplace=True)


# In[ ]:


for dataset in data:
    percentage = round(((dataset.isnull().sum()*100)/(dataset.shape[0])),4).sort_values(ascending=False)
    print(percentage.head(30),'\n')


# There are 28 features that have minimum missing values we can use the frequently occurring values to substitue these missing values. For this we use the mode function that helps us find the frequently occurring categorical and numerical values. 

# In[ ]:


values = ['GarageYrBlt','GarageFinish','GarageCond','GarageQual','GarageType','BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2',
          'MasVnrType','BsmtHalfBath','MSZoning','Functional','Utilities','BsmtFullBath','Exterior2nd','Exterior1st','KitchenQual','TotalBsmtSF',
          'GarageCars','SaleType','GarageArea','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','MasVnrArea','Electrical']

for dataset in data:
    for feature in values:
        mode_in = dataset[feature].mode()[0]
        #print(mode_in)
        dataset[feature] =  dataset[feature].fillna(mode_in) 


# In[ ]:


for dataset in data:
    percentage = round(((dataset.isnull().sum()*100)/(dataset.shape[0])),4).sort_values(ascending=False)
    print(percentage.head(30),'\n')


# In[ ]:


print("Columns in the training set: ",house_train.shape[1])
print("\nColumns in the testing set: ",house_test.shape[1])


# All the missing values within the datasets have been dealt with and the datasets are clean. Thus we now perform Exploratory Data Analysis (EDA).

# # **Exploratory Data Analysis (Data Visualization)**

# We find the correlation of each feature to the Sale Price and perform visual analysis on them.

# In[ ]:


correlation = house_train.corr()
correlation['SalePrice'].sort_values(ascending=False)[:11]


# We have selected the top 10 positively correlated features to perform our analysis.

# In[ ]:


max_corr = correlation['SalePrice'].sort_values(ascending=False)[:11].index
house_train[max_corr].head()


# In[ ]:


print(house_train.groupby('OverallQual').mean()['SalePrice'])
sns.barplot(x='OverallQual',y='SalePrice',data=house_train)


# We observe that with an increase in the quality of the house the average cost of the house increases.

# In[ ]:


print(house_train.groupby('OverallQual').count()['Id'])
sns.countplot('OverallQual',data=house_train)


# The maximum number of houses that are on sale belong to the house having a quality rating of 5. 

# In[ ]:


sns.jointplot(x='GrLivArea',y='SalePrice',data=house_train,kind='reg')


# The feature Ground Living Area has a positive correlation with the Sale Price of the house. This means that with every increase in the size of the area the cost of the house increases accordingly. If we look carfully we observe that there exists certain outliers. These need to be handled.

# In[ ]:


house_train = house_train[house_train['GrLivArea']<4500]
sns.jointplot(x='GrLivArea',y='SalePrice',data=house_train,kind='reg')


# In[ ]:


sns.jointplot(x='TotalBsmtSF',y='SalePrice',data=house_train,kind='reg')


# The feature Total Basement Square Foot has a positive correlation with the Sale Price of the house. An increase in the Square Foot of the Basement causses an equivalent increase in the Sale Price. The Basement Square Foot mainly lies between 500-2000 Square Foot, along with the Sale Price being between 5000-400000

# In[ ]:


print(house_train.groupby('FullBath').count()['Id'])
sns.countplot('FullBath',data=house_train)


# There are a maximum of 1 and 2 Full Bathroom available for sale.

# In[ ]:


print(house_train.groupby('TotRmsAbvGrd').count()['Id'])
sns.countplot('TotRmsAbvGrd',data=house_train)


# There are many house available that have 6 Rooms Above Ground.

# In[ ]:


sns.distplot(house_train['YearBuilt'],bins=30,kde=False)


# The maximum number of houses were built in the 2010.

# In[ ]:


sns.jointplot(x='GarageArea',y='SalePrice',data=house_train,kind='reg')


# The feature Garage Area has a positive correlation with the Sale Price of the house. An increase in the Area of the Garage causses an equivalent increase in the Sale Price. The Garage Area mainly lies between 250-750 Square Foot, along with the Sale Price being between 100000-200000. There exists certain outliers. These need to be handled.

# In[ ]:


house_train = house_train[house_train['GarageArea'] < 1200]
sns.jointplot(x='GarageArea',y='SalePrice',data=house_train,kind='reg')


# In[ ]:


sns.distplot(house_train['YearRemodAdd'],bins=30,kde=False)


# Houses were remodeled in the year 1950 and 2009.

# In[ ]:


sns.jointplot(x='MasVnrArea',y='SalePrice',data=house_train,kind='reg')


# There is a positive correlation between the following 2 features of the house. On closer observations we notice there exists certain outliers. These need to be handled.

# In[ ]:


house_train = house_train[house_train['MasVnrArea']<1500]
sns.jointplot(x='MasVnrArea',y='SalePrice',data=house_train,kind='reg')


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
gar = house_train.groupby('GarageType').count()['Id']
gar.plot.bar()
plt.xlabel('Garage Type')
plt.ylabel('Total Count')
plt.title('Subplot 1: Count Of Garage')

plt.subplot(1,2,2)
gar = house_train.groupby('GarageType').mean()['SalePrice']
gar.plot.bar()
plt.xlabel('Garage Type')
plt.ylabel('Total Cost')
plt.title('Subplot 2: Cost Of Garage')


# There are many Attached Garage available for sale which is quite economically feasible for the potential buyers. Whereas the most expensive type of garage is the Built in garage. We also notice that the most cost affective Garage is the Car Port, the only drawback to this that is not readily available.

# In[ ]:


table1 = pd.pivot_table(house_train, values=['SalePrice'], index=['Street'],columns=['LotShape'],aggfunc=np.mean)
table1


# In[ ]:


ax = table1.plot.bar(figsize=(8,5))
ax.set_xlabel("Street and Lot Shape")
ax.set_ylabel("Sale Price")


# There are many house present at the pave which have higher sale price as compared to the house at grvl. Among the many house at the pave the houses that are of shape IR2 are much more expensive than the rest of the available house.

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Frequency Of The Sales Price')
plt.xlabel('Sales Price')
sns.distplot(house_train['SalePrice'])


# We observe that the maximum number of houses cost between 100000 - 300000.

# In[ ]:


print(house_train['CentralAir'].value_counts())
sns.countplot('CentralAir',data = house_train)


# About 1360 houses have and centralized air conditioning facility.

# In[ ]:


sale = house_train.groupby('SaleCondition').mean()['SalePrice']
sale.plot.bar()


# The cost of the Partial conditioned houses are more than the rest of the houses.

# # **Summary Of Observations**

# * The most expensive house costs 800K $
# * Many houses that were initially built were later remodeled, thus the cost of such houses have increased over time.
# * There is a range of houses that are affordable, and these can be filtered as per the requirement of the buyers.
# * A positive relation is noticed between ground living area, basement square foot, garage area and the sales price. These features can be very useful to estimate the cost of a house.
# * Apart from the above mentioned features, there also exists many positive correlation of other features with sales price.
# 
# 

# # **Data Modeling**

# For the purpose of Data Modeling we need to split our data into training and test set.
# Once the split is done we can put our data into various models and check each the precision of each model.
# We select the model with the highest precision score.

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


# We select the numeric values from the training and the testing set.

# In[ ]:


numeric_train = pd.DataFrame(house_train.select_dtypes(include=[np.number]))
numeric_test =  pd.DataFrame(house_test.select_dtypes(include=[np.number]))


# In[ ]:


new_data = [numeric_train, numeric_test]


# In both the dataset there is an ID for the houses. ID is used to individually identify a sale, therefore it may not be very useful for us during our analysis. We need to drop each of the ID from the 2 sets.  

# In[ ]:


for dataset in new_data:
    dataset.drop(['Id'],axis=1,inplace=True)


# In[ ]:


for dataset in new_data:
    for i in dataset.columns:
            dataset[i] = dataset[i].astype(int)


# We will modify yearbuilt and yearremodadd as the age of sale.

# In[ ]:


for dataset in new_data:
    dataset['YearBltAge'] = dataset['YrSold'] - dataset['YearBuilt']
    dataset['RemodAge'] = dataset['YrSold'] - dataset['YearRemodAdd']


# In[ ]:


for dataset in new_data:
    dataset.drop(['YrSold','YearBuilt'],axis=1,inplace=True)


# In[ ]:


numeric_train['SalePrice'] = numeric_train['SalePrice'].fillna(0)
numeric_train['SalePrice'] = numeric_train['SalePrice'].astype(int)


# In[ ]:


print("Shape of the training set")
print(numeric_train.shape)
print("\nShape of the testing set")
print(numeric_test.shape)


# In[ ]:


X = numeric_train.drop('SalePrice',axis=1)
y = np.log(numeric_train['SalePrice'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# # **1. Linear Regression**

# In[ ]:


#Import Packages 
from sklearn.linear_model import LinearRegression


# In[ ]:


#Object creation and fitting of training set
lrm = LinearRegression()
lrm.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionslrm = lrm.predict(X_test)


# In[ ]:


#Create a prediction score
scorelrm = round((lrm.score(X_test, y_test)*100),2)
print ("Model Score:",scorelrm,"%")


# # **2. Ridge Regression**

# In[ ]:


#Import Packages 
from sklearn.linear_model import Ridge


# In[ ]:


#Object creation and fitting of training set
rrm = Ridge(alpha=100)
rrm.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionrrm = rrm.predict(X_test)


# In[ ]:


#Create a prediction score
scorerrm = round((rrm.score(X_test, y_test)*100),2)
print ("Model Score:",scorerrm,"%")


# # **Conclusion**

# In[ ]:


data = [['Linear Regression',scorelrm],['Ridge Regression',scorerrm]]
final = pd.DataFrame(data,columns=['Algorithm','Precision'],index=[1,2])
final

