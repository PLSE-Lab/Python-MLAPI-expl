#!/usr/bin/env python
# coding: utf-8

# **Introduction :**
#     

#  HI friends !! in this kernal i will be doing Exploratory data analysis and data visualisation and modelling and this is my first kernal i hope you will like it  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
from scipy.stats import norm
from scipy import stats

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
print(train_df.shape)
print(test_df.shape)


# As we can see there are about 80 features that affect the variable we are going to predict. 
# so we will see how data is...

# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# Now we will see what features have missing values and how many missing values each of these features contain

# In[ ]:


missing_data=pd.DataFrame(train_df.isnull().sum().reset_index())

missing_data.columns=["index","missingcount"]
missing_data=missing_data[missing_data["missingcount"]>0]
missing_data['missingper']=(missing_data['missingcount']/train_df.shape[0])*100
missing_data


# So there are about 19 features which has null values and from this there are some feauters whose missing percent is very large so instead of filling these null vaues with average values it is better to remove those particular features because they dont provide much information. so i will remove the features whose missing percentage is greater.

# In[ ]:


train_df.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley','LotFrontage'],axis=1,inplace=True)
print(train_df.shape)


# Even after doing this we still have so many feautures which are filled with null values and luckily their missing percentage is low . so instead of deleting the features, we can do two things one is removing those paricular rows and another one is filling these null values with average values. but i have another option you wanna know what it is? welll 

# In[ ]:


cor=train_df.corr()
cor['SalePrice'].sort_values(ascending=False)[0:20]


# In[ ]:


list_cor=list(cor['SalePrice'].sort_values(ascending=False)[0:20].index)
final_df=train_df[list_cor]
final_df.shape


# Here i have printed top 20 features which make huge contrubution to target variables and now i will check whether any of these variables have null values or not if there are any i will fill them with most common values

# In[ ]:


#list_cor=list(cor['SalePrice'].sort_values(ascending=False)[0:20].index)
final_df[list_cor].isnull().sum()


# We have two variables which are in good correlation with target variable and as well as has null values

# In[ ]:


print(final_df['GarageYrBlt'].dtype)
print(final_df['MasVnrArea'].dtype)


# In[ ]:


final_df['GarageYrBlt']=final_df['GarageYrBlt'].fillna(final_df['GarageYrBlt'].mode()[0])
final_df['MasVnrArea']=final_df['MasVnrArea'].fillna(final_df['MasVnrArea'].mode()[0])
final_df[list_cor].isnull().sum()


# Now we have handled missing values for the variables which are necessary so lets see how target variable values are and how these variable relate to the target variable which is SalePrice through visualizations

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(final_df['SalePrice'], color='r')
plt.title('Distribution of Sales Price', fontsize=18)

plt.show()


# In[ ]:


print(final_df['SalePrice'].skew())
print(final_df['SalePrice'].kurt())


# See there is skewness in the SalePrice variable since it has a positive value it has right Skewness (ofcourse we can tell it from the graph). Here we can remove the skewness by log transforming the values of the SalePrice

# In[ ]:


final_df['SalePrice']=np.log(final_df.loc[:,'SalePrice'])
print(final_df['SalePrice'].skew())


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(final_df['SalePrice'], color='r')
plt.title('Distribution of Sales Price', fontsize=18)
plt.show()

fig = plt.figure(figsize=(12,8))
res = stats.probplot(final_df['SalePrice'], plot=plt)
plt.show()


# In[ ]:


corrmat = final_df.corr()
f, ax = plt.subplots(figsize=(22, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True,cmap='YlOrRd',linewidths=0.2,annot_kws={'size':10})
plt.title("Heat map",fontsize=20)


# if observing heat map thoroughly we can see that there is a colinearity between the varibles.collinearity means having some relationship between the independent variables which may leads to so many problems when we interpret the results so it is better to remove the variables which are highly correlated between themselves

# In[ ]:


final_df=final_df.drop(["GarageArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF","GarageYrBlt"],axis=1)
final_df.shape


# In[ ]:


corrmat = final_df.corr()
f, ax = plt.subplots(figsize=(22, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True,cmap='YlOrRd',linewidths=0.2,annot_kws={'size':10})
plt.title("Heat map",fontsize=20)


# **1.OveralQual vs SalePrice**

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(x='OverallQual',y='SalePrice',data =final_df)
plt.ylabel("SalesPrice")
plt.xlabel("OverallQual")
plt.title("OverallQual vs SalePice")


# **2.GrLivArea vs SalePrice ** 

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(x='GrLivArea',y='SalePrice',data =final_df)
plt.ylabel("SalesPrice")
plt.xlabel("GrLivArea")
plt.title("GrLivArea vs SalePice")


# **3.GarageCars vs SalePrice **

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(x='GarageCars',y='SalePrice',data =final_df)
plt.ylabel("SalesPrice")
plt.xlabel("GarageCars")
plt.title("GarageCars vs SalePice")


# **4.TotalBsmtSF vs SalePrice**

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(x='TotalBsmtSF',y='SalePrice',data =final_df)
plt.ylabel("SalesPrice")
plt.xlabel("TotalBsmtSF")
plt.title("TotalBsmtSF vs SalePice")


# **Modelling**

# In[ ]:


finaltest_df=final_df["SalePrice"]
finaltrain_df=final_df.drop("SalePrice",axis=1)
print(finaltrain_df.shape)
print(finaltest_df.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(finaltrain_df,finaltest_df,test_size=0.3)

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[ ]:


score = r2_score(y_test,y_pred)
score


# In[ ]:


from sklearn.preprocessing import Imputer

my_imputer = Imputer()
train_X = my_imputer.fit_transform(x_train)
test_X = my_imputer.transform(x_test)


# In[ ]:


from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(x_train, y_train, verbose=False)


# In[ ]:


predictions = my_model.predict(x_test)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))


# In[ ]:


my_model = XGBRegressor(n_estimators=1000)
my_model.fit(x_train, y_train, early_stopping_rounds=5, 
             eval_set=[(x_test, y_test)], verbose=False)


# In[ ]:


score = r2_score(y_test,predictions)
score


# :**Conlclusion**

# In this kernal i try do some new things and i hope this will be useful to the beginers like me and there is more yet to come and i will modify this kernal again and agian till i make best out of it stay tune!!
