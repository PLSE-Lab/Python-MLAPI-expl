#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt



# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# get house prices csv files as a DataFrame
houseprice_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the train data
houseprice_df.head()


# In[ ]:


houseprice_df.info()
print("----------------------------")
test_df.info()

#Alley is almost empty, remove
# FireplacesQu low, could be removed
#PoolQC low, can be removed
#Fence low, can be removed
#MiscFeature low, can be removed

#After reading col description, additional useless columns:


# In[ ]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
houseprice_df = houseprice_df.drop(['Alley','PoolQC','Fence'], axis=1)
test_df    = test_df.drop(['Alley','PoolQC','Fence'], axis=1)


# In[ ]:


#correlation matrix of all variables
corrmat = houseprice_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#saleprice correlation matrix for n variables most correlated to salesprice
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(houseprice_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#MSSubClass

print(pd.value_counts(houseprice_df['MSSubClass'].values, sort=False))
#houseprice_df['MSSubClass'].plot(kind='hist', figsize=(15,3),bins=50, xlim=(0,100))

#create box plot of the variables
sns.factorplot(x="MSSubClass", y="SalePrice", data=houseprice_df,kind="box", size = 10)
#sns.boxplot(x="MSSubClass", y="SalePrice", data=houseprice_df)

#create dummy variables
houseprice_df_MSSubClass_dummies  = pd.get_dummies(houseprice_df['MSSubClass'], prefix='MSSubClass')


# In[ ]:





# In[ ]:


#MSZoning
print(pd.value_counts(houseprice_df['MSZoning'].values, sort=False))
sns.factorplot(x="MSZoning", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_MSZoning_dummies  = pd.get_dummies(houseprice_df['MSZoning'], prefix='MSZoning')
#houseprice_df_MSZoning_dummies.drop('MSZoning_RL'axis=1, inplace=True)


# In[ ]:


#LotFrontage
# get average, std, and number of NaN values in houseprice_df
average_LotFrontage   = houseprice_df["LotFrontage"].mean()
std_LotFrontage      = houseprice_df["LotFrontage"].std()
count_nan_LotFrontage = houseprice_df["LotFrontage"].isnull().sum()

print(average_LotFrontage)
print(std_LotFrontage)
print(count_nan_LotFrontage)

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_LotFrontage - std_LotFrontage, average_LotFrontage + std_LotFrontage, size = count_nan_LotFrontage)

#replace NAN for random generated numbers
houseprice_df["LotFrontage"][np.isnan(houseprice_df["LotFrontage"])] = rand_1

sns.regplot(x="LotFrontage", y="SalePrice", data=houseprice_df)


# In[ ]:


#LotArea
# get average, std, and number of NaN values in houseprice_df
average_LotArea   = houseprice_df["LotArea"].mean()
std_LotArea     = houseprice_df["LotArea"].std()
count_nan_LotArea = houseprice_df["LotArea"].isnull().sum()

print(average_LotArea)
print(std_LotArea)
print(count_nan_LotArea)

sns.regplot(x="LotArea", y="SalePrice", data=houseprice_df)


# In[ ]:


#Street
print(pd.value_counts(houseprice_df['Street'].values, sort=False))

#Few entries Grvl, Street category can be omited, gives no extra info.


# In[ ]:


#Alley

Too few entries, category ommited.


# In[ ]:


#LotShape
print(pd.value_counts(houseprice_df['LotShape'].values, sort=False))
sns.factorplot(x="LotShape", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_LotShape_raw_dummies  = pd.get_dummies(houseprice_df['LotShape'])

houseprice_df_LotShape_dummies  = pd.get_dummies(houseprice_df['LotShape'].map({'Reg':'Reg','IR2':'IR','IR3':'IR','IR1':'IR'})
, prefix='LotShape')


# In[ ]:


#LandContour
print(pd.value_counts(houseprice_df['LandContour'].values, sort=False))
sns.factorplot(x="LandContour", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_LandContour_raw_dummies  = pd.get_dummies(houseprice_df['LandContour'])

houseprice_df_LandContour_dummies  = pd.get_dummies(houseprice_df['LandContour'].map({'Lvl':'Lvl','HLS':'NL','Low':'NL','Bnk':'NL'})
, prefix='LandContour')


# In[ ]:


#Utilities
print(pd.value_counts(houseprice_df['Utilities'].values, sort=False))

#create dummy variables
#houseprice_df_LandContour_raw_dummies  = pd.get_dummies(houseprice_df['Utilities'])

#discard useless variable


# In[ ]:


#LotConfig
print(pd.value_counts(houseprice_df['LotConfig'].values, sort=False))
sns.factorplot(x="LotConfig", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_LotConfig_dummies  = pd.get_dummies(houseprice_df['LotConfig'])


# In[ ]:


#OverallQual
print(pd.value_counts(houseprice_df['OverallQual'].values, sort=False))
sns.factorplot(x="OverallQual", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_OverallQual_dummies  = pd.get_dummies(houseprice_df['OverallQual'], prefix='OverallQual')


# In[ ]:


#OverallCond
print(pd.value_counts(houseprice_df['OverallCond'].values, sort=False))
sns.factorplot(x="OverallCond", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_OverallCond_dummies  = pd.get_dummies(houseprice_df['OverallCond'], prefix='OverallCond')


# In[ ]:


#YearBuilt
print(pd.value_counts(houseprice_df['YearBuilt'].values, sort=False))
sns.factorplot(x="YearBuilt", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_YearBuilt_dummies  = pd.get_dummies(houseprice_df['YearBuilt'], prefix='YearBuilt')


# In[ ]:


#GrLivArea
# get average, std, and number of NaN values in houseprice_df
average_GrLivArea   = houseprice_df["GrLivArea"].mean()
std_GrLivArea     = houseprice_df["GrLivArea"].std()
count_nan_GrLivArea = houseprice_df["GrLivArea"].isnull().sum()

print(average_GrLivArea)
print(std_GrLivArea)
print(count_nan_GrLivArea)

sns.regplot(x="GrLivArea", y="SalePrice", data=houseprice_df)


# In[ ]:


#GarageCars
print(pd.value_counts(houseprice_df['GarageCars'].values, sort=False))
sns.factorplot(x="GarageCars", y="SalePrice", data=houseprice_df,kind="box", size = 10)

#create dummy variables
houseprice_df_GarageCars_dummies  = pd.get_dummies(houseprice_df['GarageCars'], prefix='GarageCars')


# In[ ]:


# define training and testing sets

X_train = houseprice_df_MSSubClass_dummies
X_train = X_train.join(houseprice_df_MSZoning_dummies)
X_train = X_train.join(houseprice_df["LotFrontage"])
X_train = X_train.join(houseprice_df["LotArea"])
X_train = X_train.join(houseprice_df_LotShape_raw_dummies)
X_train = X_train.join(houseprice_df_LandContour_raw_dummies)
X_train = X_train.join(houseprice_df_LotConfig_dummies)
X_train = X_train.join(houseprice_df_OverallQual_dummies)
X_train = X_train.join(houseprice_df_OverallCond_dummies)
#X_train = X_train.join(houseprice_df["YearBuilt"])
X_train = X_train.join(houseprice_df_YearBuilt_dummies)
X_train = X_train.join(houseprice_df["GrLivArea"])
X_train = X_train.join(houseprice_df_GarageCars_dummies)



Y_train = houseprice_df["SalePrice"]


# In[ ]:


# Linear Regression

linReg = LinearRegression()

linReg.fit(X_train, Y_train)

#Y_pred = logreg.predict(X_test)

linReg.score(X_train, Y_train)


# In[ ]:




