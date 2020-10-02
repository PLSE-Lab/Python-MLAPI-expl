#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


train.shape


# In[ ]:


test.head(5)


# In[ ]:


test.shape


# In[ ]:


train["SalePrice"].describe()


# In[ ]:


print("Skewness: %f" % train["SalePrice"].skew())
print("Kurtosis: %f" % train["SalePrice"].kurt())


# **Detect outliers**

# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x=train["GrLivArea"], y=train["SalePrice"])
plt.ylabel("SalePrice", fontsize=13)
plt.xlabel("GrLivArea", fontsize=13)
plt.show()


# In[ ]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)


# OverallQual, GrLivArea, YearBuilt, TotalBsmtSF, 1stFlrSF, FullBath, GarageArea, GarageCars

# In[ ]:


train.drop(['Id', 'Alley', 'PoolQC'], axis=1)


# In[ ]:


var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[ ]:


fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


# In[ ]:


location=[]
for x in train['Neighborhood']:
    if x not in location:
        location.append(x)
print(location)
train.sort_values(["Neighborhood", "SalePrice"])

nb_data = pd.concat([train['SalePrice'], train['Neighborhood']], axis=1)
fig = sns.boxplot(x="Neighborhood", y="SalePrice", data=nb_data)
fig.axis(ymin=0, ymax=800000)


# In[ ]:


from sklearn import preprocessing
f_names=["MSSubClass", "MSZoning", "LotShape", "CentralAir", 'Neighborhood']
for x in f_names:
    label=preprocessing.LabelEncoder()
    train[x] = label.fit_transform(train[x])
corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)


# In[ ]:


k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(train[cols], size=2.5)
plt.show()


# In[ ]:


train = train[['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']]
train.head()


# **Data Scalling**

# null values

# In[ ]:


train.isnull().sum()


# In[ ]:



from sklearn import preprocessing

train_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = train[train_cols].values
y = train['SalePrice'].values
print(x)
print(y)
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
print(x_scaled)
print(y_scaled)


# In[ ]:


from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_train, y_train) # coefficient of determination R2


# In[ ]:


reg.coef_


# In[ ]:


reg.intercept_


# In[ ]:


y_pred = reg.predict(X_test)


# In[ ]:


print("cost: " + str(np.sum(y_pred - y_test)/len(y_pred)))


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:


sum(abs(y_pred - y_test))/len(y_pred)


# In[ ]:


compare_df = pd.DataFrame({'predicted': y_pred.flatten(), 'real': y_test.flatten()})
compare_df.head(5)


# In[ ]:


compare_df.head(20).plot.bar()


# **Prediction**

# In[ ]:


test_data = pd.read_csv("../input/test.csv")
test_data.head()


# In[ ]:


test_data[train_cols].isnull().sum()


# In[ ]:


test_data['GarageCars'].describe()


# In[ ]:


test_data['TotalBsmtSF'].describe()


# In[ ]:


test_data.TotalBsmtSF = test_data.TotalBsmtSF.fillna(1046.117970)


# In[ ]:


test_data.GarageCars = test_data.GarageCars.fillna(1.766118)


# In[ ]:


ids = test_data['Id']


# In[ ]:


test_data = test_data[train_cols]
test_data.head()


# In[ ]:


x = test_data.values
y_test_pred = reg.predict(x)
print(y_test_pred)
print(y_test_pred.shape)
print(x.shape)


# **To save the result into csv**

# In[ ]:


prediction = pd.DataFrame(y_test_pred, columns=["SalePrice"])
result = pd.concat([ids, prediction], axis=1)
result.columns


# In[ ]:



result.to_csv('./Prediction.csv', index=False)


# In[ ]:




# sns.distplot(train['SalePrice'], fit=norm)
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu={:.2f} and sigma={:.2f}\n'.format(mu, sigma))

# plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()


# In[ ]:


# import numpy as np
# train["SalePrice"] = np.log1p(train["SalePrice"])
# sns.distplot(train["SalePrice"], fit=norm)
# (mu, sigma) = norm.fit(train["SalePrice"])
# print("\n mu={:.2f} and sigma={:.2f}\n".format(mu, sigma))

# plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

