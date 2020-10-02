#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction (Regression)  

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pd.pandas.set_option('display.max_columns', None)

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data = data.drop(['Id','Alley','FireplaceQu', 'PoolQC', 'Fence','MiscFeature' ], axis=1)


# # Handling Missing Values

# ## Numeric Features

# In[ ]:


missNum = [f for f in data if data[f].isnull().sum()>0 and data[f].dtype!='O']


# In[ ]:


missNum_mean = data[missNum].mean()


# In[ ]:


missNum_mean


# In[ ]:


data[missNum] = data[missNum].fillna(missNum_mean)


# In[ ]:


data


# ## Categorical Features

# In[ ]:


missCat = [f for f in data if data[f].isnull().sum()>0 and data[f].dtype == 'O' ]


# In[ ]:


missCat


# In[ ]:


missCat_mode = data[missCat].mode().sum()


# In[ ]:


missCat_mode


# In[ ]:


data[missCat] = data[missCat].fillna(missCat_mode)


# In[ ]:


data[missCat].isnull().sum()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(data.isnull())


# In[ ]:


numerical_r = [f for f in data if data[f].dtype !='O']


# In[ ]:


for f in numerical_r:
    dataNr = data.copy()
    plt.scatter(dataNr[f], data['SalePrice'])
    plt.xlabel(f)
    plt.show()
    


# In[ ]:


data.describe()


# # Feature Engineering

# ## Categorical Features

# In[ ]:


categorical =[f for f in data if data[f].dtype == 'O']


# In[ ]:


data[categorical].shape


# In[ ]:


for f in categorical:
    dataCat = data.copy()
    dataCat.groupby(f)['SalePrice'].mean().plot.bar()
    plt.show()


# ## Numerical Features

# In[ ]:


sale_price = data['SalePrice']


# In[ ]:


data = data.drop(['SalePrice'], axis=1)


# In[ ]:


numerical = [f for f in data if data[f].dtype !='O']


# In[ ]:


numerical


# ## Separating Year Features

# In[ ]:


year = [f for f in numerical if 'Year' in f or 'Yr' in f]
data[year].head()


# In[ ]:


data = data.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'], axis=1)


# In[ ]:


numerical = [f for f in data if data[f].dtype !='O']


# In[ ]:


for f in numerical:
    dataC = data.copy()
    data[f].hist()
    plt.xlabel(f)
    plt.show()


# In[ ]:


dataT = np.log(data[numerical]+1)


# In[ ]:


dataT.describe()


# In[ ]:


for f in numerical:
    dataC = dataT.copy()
    dataC[f].hist()
    plt.xlabel(f)
    plt.show()


# ## Detecting and Removing Outliers

# In[ ]:


maxTh = dataT.quantile(0.95)


# In[ ]:


maxTh[1:32]


# In[ ]:


minTh = dataT.quantile(0.05)


# In[ ]:


minTh[1:32]


# In[ ]:


df2  = dataT[(dataT<maxTh) & (dataT>minTh)]


# In[ ]:


df2


# In[ ]:


df2.isnull().sum()


# In[ ]:


df3 = df2.drop(['BsmtFinSF2', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces',
               'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
               'MiscVal'], axis=1)


# In[ ]:


df3.isnull().sum()


# In[ ]:


df3 = df3.drop(['MasVnrArea', 'BedroomAbvGr'], axis=1)


# In[ ]:


numMissN = [f for f in df3 if df3[f].isnull().sum()>1]


# In[ ]:


numMissN = df3[numMissN].mean()


# In[ ]:


numMissN


# In[ ]:


df4 = df3.fillna(numMissN)


# In[ ]:


df4.isnull().sum()
df4


# # Distribution of Numerical Features After removing Outliers

# In[ ]:


for f in df4:
    dataL = df4.copy()
    dataL[f].hist()
    plt.show()


# # Concatinate Categorical and Numerical Features

# In[ ]:


df_cat = data[categorical]


# In[ ]:


df6 = pd.concat([df_cat, df4], axis=1)


# In[ ]:


df6.shape


# In[ ]:


df6


# In[ ]:


cat_features = [f for f in df6 if df6[f].dtype == 'O']
cat_features


# # Encoding Categorical Features

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


df7= df6[cat_features].apply(LabelEncoder().fit_transform)


# In[ ]:


df7


# In[ ]:


df8 = pd.concat([df7, df4], axis=1)


# In[ ]:


df8


# # Bulid Model
# 
# ### - Linear Regression
# ### - Random Forest Regressor 
# ### - Ridge Regressor

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


# In[ ]:


#independent features

X = df8


# In[ ]:


X


# In[ ]:


#dependent feature

y = sale_price


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# ## Splittind into train and test data

# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)


# In[ ]:


xtrain.shape


# In[ ]:


ytrain.shape


# # - Linear Regression

# In[ ]:


model1 = LinearRegression()
model1.fit(xtrain, ytrain)


# In[ ]:


print("Train Accuracy:",model1.score(xtrain, ytrain))
print("Test Accuracy:",model1.score(xtest, ytest))


# In[ ]:


model1.predict([[3, 1,3,3,0,4,0,5,2,2,0,5,1,1,12,13,1,2,4,2,2,3,3,2,5,1,0,1,4,2,6,1,1,4,4,2,8,4,4.110874,4.189655,9.042040,2.079442,1.791759,6.561031,5.017280,6.753438,6.753438,7.444833,2.197225,1.098612,6.308098,1.934685]])


# # - Random Forest Regressor

# In[ ]:


model2 = RandomForestRegressor(n_estimators=250)


# In[ ]:


model2.fit(xtrain, ytrain)


# In[ ]:


print("Train Accuracy:",model2.score(xtrain, ytrain))
print("Test Accuracy:",model2.score(xtest, ytest))


# In[ ]:


model2.predict([[3, 1,3,3,0,4,0,5,2,2,0,5,1,1,12,13,1,2,4,2,2,3,3,2,5,1,0,1,4,2,6,1,1,4,4,2,8,4,4.110874,4.189655,9.042040,2.079442,1.791759,6.561031,5.017280,6.753438,6.753438,7.444833,2.197225,1.098612,6.308098,1.934685]])


# # - Ridge Regression

# In[ ]:


model3 = Ridge(max_iter=100)
model3.fit(xtrain, ytrain)


# In[ ]:


print("Train Accuracy:",model3.score(xtrain, ytrain))
print("Test Accuracy:",model3.score(xtest, ytest))


# In[ ]:


model3.predict([[3, 1,3,3,0,4,0,5,2,2,0,5,1,1,12,13,1,2,4,2,2,3,3,2,5,1,0,1,4,2,6,1,1,4,4,2,8,4,4.110874,4.189655,9.042040,2.079442,1.791759,6.561031,5.017280,6.753438,6.753438,7.444833,2.197225,1.098612,6.308098,1.934685]])


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df8.corr())

