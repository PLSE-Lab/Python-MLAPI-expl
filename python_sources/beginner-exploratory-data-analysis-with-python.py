#!/usr/bin/env python
# coding: utf-8

# # BEGINNER EXPLORATORY DATA ANALYSIS WITH PYTHON

# **Index**
# 1. Initiation
# 2. Data cleaning and preprocessing
# 3. Study of variables (features) in isolation
# 4. Study of variables in groups and combinations
# 5. To be continued...

# # 1. Initiation

# In[ ]:


#importing the required packages
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


#Load the train data in dataframe
df_train = pd.read_csv('../input/train.csv')


# In[ ]:


#Display the columns in training set
df_train.columns


# # 2. Data cleaning and preprocessing
# First step in any project - to clean and preprocess the data for further analysis and model building. Here the focus is on missing values (NA) and the corresponding patterns. 

# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.info()


# In[ ]:


#Analysing Missing Values (NA)
total_na = df_train.isnull().sum().sort_values(ascending=False)
percent_na = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
na_df = pd.concat([total_na, percent_na], axis=1, keys=['No of Missing Values', '% of Missing Values']).sort_values(by= '% of Missing Values', ascending = False)
na_df.head(20)


# Below set of variables 'PoolQC', 'MiscFeature' and 'FireplaceQu' looks like probable outliers. Similarly, 'Electrical'and 'PavedDrive' also looks like probable outliers with fewest occurences. Let us drop them from our dataframe.
# 
# Remaining missing values will be handled by replacing them with mean values.

# In[ ]:


na_df[na_df['% of Missing Values'] > 0.4]


# In[ ]:


na_df.loc[['MasVnrType', 'MasVnrArea','Electrical']]


# In[ ]:


#dealing with missing data
df_train = df_train.drop((na_df[na_df['% of Missing Values'] > 0.4]).index,1)
df_train = df_train.drop((na_df.loc[['MasVnrType', 'MasVnrArea','Electrical']]).index,1)


# In[ ]:


na_df.isnull().sum().max() #Checking for any missed out NAs
# na_df.head(20)


# In[ ]:


#Filling NA for other Missing Values with Mean values
df_train['LotFrontage'].fillna(value = df_train['LotFrontage'].mean, inplace = True)
df_train['GarageCond'].fillna(value = df_train['GarageCond'].mean, inplace = True)
df_train['GarageType'].fillna(value = df_train['GarageType'].mean, inplace = True)
df_train['GarageFinish'].fillna(value = df_train['GarageFinish'].mean, inplace = True)
df_train['GarageQual'].fillna(value = df_train['GarageQual'].mean, inplace = True)
df_train['GarageYrBlt'].fillna(value = df_train['GarageYrBlt'].mean, inplace = True)
df_train['BsmtExposure'].fillna(value = df_train['BsmtExposure'].mean, inplace = True)
df_train['BsmtFinType2'].fillna(value = df_train['BsmtFinType2'].mean, inplace = True)
df_train['BsmtFinType1'].fillna(value = df_train['BsmtFinType1'].mean, inplace = True)
df_train['BsmtCond'].fillna(value = df_train['BsmtCond'].mean, inplace = True)
df_train['BsmtQual'].fillna(value = df_train['BsmtQual'].mean, inplace = True)


# ## Detecting Outliers
# Outliers can be detected with the help of Univariate and Bivariate analysis

# ### Univariate analysis

# In[ ]:


sns.boxplot(x = df_train['SalePrice'])


# In[ ]:


#standardizing data
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


#bivariate analysis saleprice/grlivarea
sns.jointplot(x = 'GrLivArea', y = 'SalePrice', data = df_train, kind = 'reg');


# ### Bivariate analysis

# Deleting the outliers identified from boxplot, StandardScaler and Jointplot...

# In[ ]:


#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[ ]:


#bivariate analysis saleprice/grlivarea
sns.jointplot(x = 'TotalBsmtSF', y = 'SalePrice', data = df_train, kind = 'reg');


# Not much outliers in this plot. Moving on to study of SalePrice variable...

# # 3. Study of variables (features) in isolation: 'SalePrice' variable

# In[ ]:


#descriptive statistics summary
df_train['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(df_train['SalePrice'], fit = norm);


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# We can observe that the distribution is showing peakedness with a kurtosis of 6.5362 and positively skewed with Skewness at 1.8828.

# In[ ]:


#Normal probability plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# 'SalePrice' has positive skew and is not following the shape of normal distribution. In the last part, let us convert it into logarathamic scale and recheck.

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In[ ]:


#Creating a new column for category variable
#if area>0 then 1, else if area==0 then 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


# ### Relationship with numerical variables

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
sns.lmplot(x=var, y='SalePrice', markers = 'x', data = df_train)


# SalePrice and GrLivArea are positive + linearly correlated. 

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
sns.lmplot(x=var, y='SalePrice', markers = 'x', fit_reg = True, data = df_train)


# SalePrice and TotalBsmtSF are even more positively correlated with much stronger linear correlation. 

# ### Relationship with categorical features

# In[ ]:



#box plot overallqual/saleprice
var = 'OverallQual'
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=df_train)
fig.axis(ymin=0, ymax=800000);


# We can see that as OverallQual improves, SalePrice also improves with significant linear increase in terms of IQR.

# In[ ]:


var = 'YearBuilt'
f, ax = plt.subplots(figsize=(18, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=df_train)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# Not a clear relationship established here. However, we can still say that the recently built houses has an higher sale price. 
# Also, year 2009 is an anomaly here due to the sub-prime crisis - all the asset classes went south. 

# # 4. Study of variables in groups and combinations

# Now let us delve deep into the study of relationships among variables in combinations. We will be using the regression scatter plots, correlation matrix, heatmap etc for this section.

# #### Correlation matrix (heatmap style)

# In[ ]:


#Heatmap from Correlation Matrix for all the variables in dataset
corr_mat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=.8, square=True);


# In[ ]:


#STRONG POSITIVELY CORRELATED
corr_mat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat[corr_mat > 0.7], vmax=.8, annot = True, square=True);


# In[ ]:


#STRONG NEGATIVELY CORRELATED
corr_mat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat[corr_mat < -0.3], vmax=.8, annot = True, square=True);
# sns.heatmap(corr_mat, mask = corr_mat < -0.4, vmax=.8, annot = True, square=True);


# Following fields are observed to be vs having a strong correlation:
# 1. 'TotalBsmtSF'vs '1stFlrSF'. 
# 2. 'YearBlt'vs 'GarageYrBlt'. 
# (This is obvious isn't it! Not an useful observation but it shows that these variables are multi-collinear in nature. 

# #### 'SalePrice' correlation matrix (zoomed heatmap style)

# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
mask = np.zeros_like(cm)
mask[np.triu_indices_from(mask)] = True
hm = sns.heatmap(cm, cbar=True, mask = mask, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# #### Scatter plots : 'SalePrice' vs Correlated Variables

# In[ ]:


#scatterplot
sns.set(palette = 'deep')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# ## Now let us try to apply StandardScaler and log transformations and recheck some of the variable plots

# In[ ]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);


# In[ ]:


#applying log transformation
df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])


# In[ ]:


sns.distplot(df_train['SalePrice_Log'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice_Log'], plot=plt)


# In[ ]:


#standardizing data
totalBsmtSF_scaled = StandardScaler().fit_transform(df_train['TotalBsmtSF'][:,np.newaxis]);


# In[ ]:


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[ ]:


sns.distplot(df_train['TotalBsmtSF'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# TO BE CONTINUED....

# In[ ]:




