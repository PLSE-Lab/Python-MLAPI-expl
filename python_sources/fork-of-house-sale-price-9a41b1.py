#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data1 = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
data1.shape, test.shape,sub.shape


# In[ ]:


test['SalePrice']=sub['SalePrice']


# In[ ]:


data=pd.concat([data1,test])


# In[ ]:


test.shape


# In[ ]:


data.shape


# In[ ]:


#categorial data
df5=data.select_dtypes('object')


# In[ ]:


data.select_dtypes(np.number).columns


# In[ ]:


data=data.set_index('Id')


# ****DATA CLEANING****

# In[ ]:


#MISSING VALUES
def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percentage=(df.isnull().sum()/(df.shape[0]))*100
    temp = pd.concat([total,percentage], axis = 1,keys= ['Total','percentage'])
    return temp.loc[(temp['Total'] > 0)]


df1=missingValuesInfo(data)


# From above we can say that Alley,MiscFeature,Fence,PoolQC have more than 50% of missing values so we need to drop it 

# In[ ]:


df1


# In[ ]:


cloumnsdrop=df1[df1.percentage>20]
cloumnsdrop
data.drop(['Alley','MiscFeature','Fence','PoolQC','FireplaceQu'],axis=1,inplace=True)


# In[ ]:


data['BsmtQual']=data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
data['BsmtCond']=data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
data['BsmtExposure']=data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
data['BsmtFinType1']=data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0])
data['BsmtFinType2']=data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])
data['Electrical']=data['Electrical'].fillna(data['Electrical'].mode()[0])
data['GarageType']=data['GarageType'].fillna(data['GarageType'].mode()[0])
data['GarageFinish']=data['GarageFinish'].fillna(data['GarageFinish'].mode()[0])
data['GarageQual']=data['GarageQual'].fillna(data['GarageQual'].mode()[0])
data['GarageCond']=data['GarageCond'].fillna(data['GarageCond'].mode()[0])
data['MasVnrType']=data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
data['MSZoning']=data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['SaleType']=data['SaleType'].fillna(data['SaleType'].mode()[0])
data['Exterior2nd']=data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['Exterior1st']=data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Utilities']=data['Utilities'].fillna(data['Utilities'].mode()[0])
data['Functional']=data['Functional'].fillna(data['Functional'].mode()[0])
data['KitchenQual']=data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])


# In[ ]:


data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].median())
data['GarageYrBlt']=data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())
data['MasVnrArea']=data['MasVnrArea'].fillna(data['MasVnrArea'].median())

data['GarageArea']=data['GarageArea'].fillna(data['GarageArea'].median())


data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].median())
data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].median())
data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].median())
data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].median())
data['BsmtFullBath']=data['BsmtFullBath'].fillna(data['BsmtFullBath'].median())
data['BsmtHalfBath']=data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].median())

data['GarageCars']=data['GarageCars'].fillna(data['GarageCars'].median())


# Data Explore

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
g=sns.distplot(data1['SalePrice'])
g=g.legend(['Skewness :{:.2f}'.format(data['SalePrice'].skew())],loc='best')


# In[ ]:


numerical_features = [feature for feature in data.columns if data[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
data1[numerical_features].head()


# In[ ]:


year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature


# In[ ]:


for feature in year_feature:
    print(feature, data1[feature].unique())


# In[ ]:


data.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[ ]:



## Here we will compare the difference between All years feature with SalePrice

for feature in year_feature:
    if feature!='YrSold':
        data1=data1.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data1[feature]=data1['YrSold']-data1[feature]

        plt.scatter(data1[feature],data1['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[ ]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(data1[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[ ]:


data[discrete_feature].head()


# In[ ]:


## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data1=data1.copy()
    #data.groupby(feature)['SalePrice'].median().plot.bar()
    sns.catplot(x=feature, y="SalePrice", kind="bar", data=data1)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[ ]:


for feature in continuous_feature:
    data1=data1.copy()
    data1[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[ ]:


## We will be using logarithmic transformation


for feature in continuous_feature:
    data1=data1.copy()
    if 0 in data1[feature].unique():
        pass
    else:
        data1[feature]=np.log(data1[feature])
        data1['SalePrice']=np.log(data1['SalePrice'])
        plt.scatter(data1[feature],data1['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# In[ ]:


for feature in continuous_feature:
    data1=data1.copy()
    if 0 in data1[feature].unique():
        pass
    else:
        data1[feature]=np.log(data1[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# Feature Scaling

# In[ ]:


## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    data[feature]=data['YrSold']-data[feature]


# In[ ]:


data[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# In[ ]:


import numpy as np
num_features=['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']

for feature in num_features:
    data[feature]=np.log(data[feature])


# In[ ]:


data.sample(10)


# In[ ]:


data = pd.get_dummies(data)
data.head()


# In[ ]:


scaling_feature=[feature for feature in data.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature)


# In[ ]:


scaling_feature


# In[ ]:


feature_scale=[feature for feature in data.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(data[feature_scale])


# In[ ]:


scaler.transform(data[feature_scale])


# Model

# In[ ]:


y=data[['SalePrice']]


# In[ ]:


X=data.drop(['SalePrice'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4998, random_state=42)


# In[ ]:


X_test.shape


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


clf = Lasso(alpha=20.0) # remember to set the seed, the random state in this function
clf.fit(X_train,y_train)


# In[ ]:


prediction=clf.predict( X_test)


# In[ ]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[ ]:


r2_score(y_test, prediction)


# In[ ]:


testdf = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

#Visualize the first 5 rows
submission.head()
filename = 'HosingPrice.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:





# In[ ]:




