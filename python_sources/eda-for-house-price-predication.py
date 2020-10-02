#!/usr/bin/env python
# coding: utf-8

# ## House Prices Prediction
# 

# ### The main of the project which predicted House Prices based on Advanced Regression Techniques.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## import the dataset 
import os
print(os.listdir("../input"))


# In[ ]:


df=pd.read_csv("../input/train.csv")
df.head(10)


# ### to which how many colunms and rows avaiable in dataset

# In[ ]:


df.shape


# ### to  check columns names

# In[ ]:


df.columns


# In[ ]:


### givies you the summary
df.describe()


# ### to Identify numerical and categorical variables

# In[ ]:


df.info()


# ### to know which fields or colunms has  missing value is avaiable in the dataset

# In[ ]:


df.columns[df.isnull().any()].tolist()


# ### to see my rows in colunms contains the missing data

# In[ ]:


df.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data["Percent"],color="green")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[ ]:


df.drop(['PoolQC','MiscFeature','Alley','Fence'] ,axis=1, inplace=True)
df.head()


# In[ ]:


df.isnull().sum().sort_values(ascending=False).head(20)


# ### To which columns are has numerical and catergorical variable

# In[ ]:


# syntax to know which colunm has the categorical features
categorical_features = df.select_dtypes(include = ["object"]).columns
categorical_features


# In[ ]:


# syntax to know which colunm has the numerical features
numerical_features = df.select_dtypes(exclude = ["object"]).columns
numerical_features


# In[ ]:


# Differentiate numerical features (minus the target) and categorical features
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))


# ## Visualization 

# ### to visulaization dependent variable which is target variable which is SalePrice

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.set_style('whitegrid')
sns.distplot(df.SalePrice,color='red')
plt.xlabel('SalePrice', fontsize=15)
#plt.title('Given Data', fontsize=15)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
#plt.figure()
#plt.subplot(212)
sns.set_style('whitegrid')
stats.probplot(df['SalePrice'], plot=plt)
plt.xlabel('SalePrice', fontsize=15)
plt.title('Given Data', fontsize=15)
plt.show()


# ### The target variable is right skewed. As (linear) models love normally distributed data , 
# ### we need to transform this variable and make it more normally distributed.
# ### we can take log or square transform 

# In[ ]:


f, ax = plt.subplots(figsize=(12, 10))
plt.figure(1)
plt.subplot(211)
sns.set_style('whitegrid')
sns.distplot(df['SalePrice'].apply(np.sqrt),color='red')
plt.xlabel('Square of SalePrice', fontsize=15)
plt.title('Square Transform', fontsize=15)



f, ax = plt.subplots(figsize=(12, 10))
plt.figure(2)
plt.subplot(212)
sns.set_style('whitegrid')
sns.distplot(df['SalePrice'].apply(np.log),color='red')
plt.xlabel('Log of SalePrice', fontsize=15)
plt.title('Log Transform', fontsize=15)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12, 12))
plt.figure(1)
plt.subplot(211)
sns.set_style('whitegrid')
stats.probplot(df['SalePrice'].apply(np.sqrt), plot=plt)
plt.xlabel('Square of SalePrice', fontsize=15)
plt.title('Square Transform', fontsize=15)



f, ax = plt.subplots(figsize=(12, 12))
plt.figure(2)
plt.subplot(212)
sns.set_style('whitegrid')
stats.probplot(df['SalePrice'].apply(np.log), plot=plt)
plt.xlabel('Log of SalePrice', fontsize=15)
plt.title('Log Transform', fontsize=15)
plt.show()


#  ### Find out the Relationship with categorical feature

# In[ ]:



data = pd.concat([df['SalePrice'], df['BedroomAbvGr']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x='BedroomAbvGr', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=500000);


# In[ ]:


## plot to check the total number of rooms
data = pd.concat([df['SalePrice'], df['TotRmsAbvGrd']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x='TotRmsAbvGrd', y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=500000);


# In[ ]:


# to check total SaleCondition
data = pd.concat([df['SalePrice'], df['SaleCondition']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.violinplot(x='SaleCondition', y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=500000);


# In[ ]:


## to check over all conditions of house
data = pd.concat([df['SalePrice'], df['OverallCond']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.violinplot(x='OverallCond', y="SalePrice", data=data)


# In[ ]:


### Draw a set of vertical bars with nested grouping by a two variables


# In[ ]:



## to check over all conditions of house with bed room
data = pd.concat([df['SalePrice'], df['OverallCond'],df['BedroomAbvGr']],axis=1)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(x="OverallCond", y="SalePrice", hue="BedroomAbvGr",data=data)


# In[ ]:


## to check over all conditions of house with sale condintion
data = pd.concat([df['SalePrice'], df['OverallCond'],df['SaleCondition']],axis=1)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(x="OverallCond", y="SalePrice", hue="SaleCondition",data=data)


# In[ ]:


## to check BedroomAbvGr house with sale condintion
data = pd.concat([df['SalePrice'], df['BedroomAbvGr'],df['SaleCondition']],axis=1)
f, ax = plt.subplots(figsize=(15, 12))
sns.boxplot(x="BedroomAbvGr", y="SalePrice", hue="SaleCondition",data=data)
fig.axis(ymin=0, ymax=500000)


# ### To Know The Relationship with numerical variable

# In[ ]:


#scatter plot grlivarea/saleprice
#var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000))


# #### It seems that 'SalePrice' and 'GrLivArea' are really old friends, with a linear relationship.

# In[ ]:


#scatter plot TotalBsmtSF /saleprice
#var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice',ylim=(0,800000))


# ### if see the above plot its not linear relationship so we are going to make change

# In[ ]:


#Deleting outliers
df = df.drop(df[(df['TotalBsmtSF']>2000) & (df['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df['TotalBsmtSF'], df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()


# ### It seems that 'SalePrice' and 'TotalBsmtsf' are really old friends, with a linear relationship

# ### Correlation matrix  in Visulization Form

# In[ ]:


#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(corrmat, vmax=.9, square=True);


# # most correlated features
# 

# In[ ]:


# most correlated features
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="Oranges")


# ### Scatter plots between 'SalePrice' and correlated variables

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size = 2.5)
plt.show();


# ### Although we already know some of the main figures, 
# ### this mega scatter plot gives us a reasonable idea about variables relationships

# In[ ]:




