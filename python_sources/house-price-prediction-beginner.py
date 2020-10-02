#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.preprocessing import scale
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# DATASET LOAD AND INFO

# In[ ]:


df=pd.read_csv('../input/home-data-for-ml-course/train.csv')


# In[ ]:


df.head(6)


# In[ ]:


df.tail(6)


# In[ ]:


df.info()


# Total numerical values are 38
# And categorical values are 43.

# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.describe().style.background_gradient(cmap='Blues')


# In[ ]:


df.skew()


# In[ ]:


df.kurt()


# from above two lines we are sneeking the skewness and kurtosis of the data.

# missing values

# In[ ]:


df.isna().sum().sort_values()


# In[ ]:


#total no of missing value
df.isna().sum().sum()


# In[ ]:


mis = df.isnull().sum().to_frame()

mis.columns = ['nMissings']


mis['perMissing'] = mis['nMissings']/1460
mis = mis[mis.nMissings >= 1]

misor = mis.sort_values(by = ['nMissings'], ascending=False)
plt.figure(figsize=(30,10))          
sns.barplot(x = misor.index, y = misor['perMissing']);
plt.xticks(rotation=90);


# #now working with saleprice as we are working on housing price so our main point of interest is sales price and how it is change with different parameters 

# In[ ]:


print(df['SalePrice'].describe())


# from the above lines we can see that median is 163000.000000, 1st quantile is 129975.000000, 3rd quantile is 214000.000000, minimum sale is 34900, maximum is 755000 

# In[ ]:


#distribution plot of sales price
fig, ax=plt.subplots(figsize=(30,10))
sns.distplot(a=df['SalePrice'], ax=ax);


# from the above plot we can see that sold price is skewd to left side and peake at in between 100000 to 200000

# In[ ]:


sns.boxplot(data=df['SalePrice']);


# In[ ]:


sns.scatterplot(df['SalePrice'], df['YrSold']);


# highst soldprice is in the year of 2007 the price is more than 700000

# numeric values related to SalePrice 

# In[ ]:


var = ['SalePrice', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']


# In[ ]:


plt.figure(figsize=(30,10))
corr = df[var].corr()
sns.heatmap(data=corr, annot=True);


# SalePrice strong correlations with GrLivArea (0.71), GarageArea (0.62), 1stFlSF (0.61), TotalBsmtSF (0.61). MasVnrArea (0.48) , medium correlation with Saleprice . The remaining weak or very weak (PoolArea, ScreenPorch, 3SsnPorch, EnclosedPorch, BsmtFinSF2) correlations with SalePrice.

# In[ ]:


print('correlation in betwn saleprice and yearsold:',pearsonr(df.SalePrice, df.YrSold))
print(sm.OLS(df.YrSold, df.SalePrice).fit().summary())
chart =sns.lmplot(y= 'SalePrice', x='YrSold', data=df)


# the correlation in between year sold and sold price is not that much variate. it is a steady one.

# In[ ]:


print('correlation in betwn saleprice and GarageArea:',pearsonr(df.SalePrice, df.GarageArea))
print(sm.OLS(df.GarageArea, df.SalePrice).fit().summary())
chart =sns.lmplot(y= 'SalePrice', x='GarageArea', data=df)


# In[ ]:


print('correlation in betwn saleprice and ScreenProch:',pearsonr(df.SalePrice, df.ScreenPorch))
print(sm.OLS(df.ScreenPorch, df.SalePrice).fit().summary())
chart =sns.lmplot(y= 'SalePrice', x='ScreenPorch', data=df)


# In[ ]:


#work with categorical features
# Columns containing text values (dtypes == 'object') are categorical features.


# In[ ]:


catdf = (df.dtypes == 'object')


# In[ ]:


cat = list(catdf[catdf].index)
mancat = ['MSSubClass', 'OverallQual', 'OverallCond', ]
c= cat + mancat


# In[ ]:


data = {}
for i in c:
    v = i
    uniq = len(df[i].unique().tolist())
    data[i] = (v, uniq)


# In[ ]:


dfcat= pd.DataFrame.from_dict(data, orient='index', columns=['v','uniq'])


ordf = dfcat.sort_values(by = ['uniq'], ascending=True)

plt.figure(figsize=(30,10))
sns.barplot(ordf.v, ordf.uniq)
plt.xticks(rotation=90)
plt.show()


# categories varies from two to 25 different categories. neighbourhood is the unique one.

# lets take different categorical values from the above plot and do some more plotting for sale price.

# In[ ]:


#1
df.Street.unique()


# In[ ]:


#break it now and plotting# break data into different parts
Paver = df[df.Street == 'Pave']
Gravel = df[df.Street == 'Grvl']
fig, ax=plt.subplots(figsize=(30,10))
sns.distplot(a = np.log(Paver['SalePrice']), label="Paver block", kde=False);
sns.distplot(a = np.log(Gravel['SalePrice']), label="gravel one", kde=False);
plt.legend();


# In[ ]:


df.BldgType.unique()


# In[ ]:



singleframe = df[df.BldgType == '1Fam']
doubleframe = df[df.BldgType == '2fmCon']
duplex = df[df.BldgType == 'Duplex']
townhouseeast = df[df.BldgType == 'TwnhsE']
townhousesouth = df[df.BldgType == 'Twnhs']

plt.figure(figsize=(30,10))


sns.distplot(a = np.log(singleframe['SalePrice']), label="single frame", kde=False);
sns.distplot(a = np.log(doubleframe['SalePrice']), label="double frame", kde=False);
sns.distplot(a = np.log(duplex['SalePrice']), label="duplex", kde=False);
sns.distplot(a = np.log(townhouseeast['SalePrice']), label="town house east", kde=False);
sns.distplot(a = np.log(townhousesouth['SalePrice']), label="town house south", kde=False);

plt.legend();


# i am using log values for normalise any outliers, and bring down the scale for sale price

# now randomly i am try to find out the relation with saleprice for different categorical data

# In[ ]:


#for house style
plt.figure(figsize=(30,10))
temp = df.groupby(['HouseStyle'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)
plot1 = sns.boxplot(data=df,x='HouseStyle',y="SalePrice",order=temp['HouseStyle'].to_list());
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);


# In[ ]:


#with exterior
plt.figure(figsize=(30,10))
table = df.groupby(['Exterior2nd'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)
plot1  = sns.stripplot(data=df,x='Exterior2nd',y="SalePrice",order=table['Exterior2nd'].to_list());
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);


# In[ ]:


# with neighborhood
plt.figure(figsize=(30,10))
temp = df.groupby(['Neighborhood'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)
plot1  = sns.violinplot(data=df,x='Neighborhood',y="SalePrice",order=temp['Neighborhood'].to_list());
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);


# In[ ]:


#with foundation
plt.figure(figsize=(30,10))
temp = df.groupby(['Foundation'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)
plot1  = sns.barplot(data=df,x='Foundation',y="SalePrice",order=temp['Foundation'].to_list());
plot1 .set_xticklabels(plot1.get_xticklabels(), rotation=90);


# In[ ]:


# with lot shape
plt.figure(figsize=(30,10))
temp = df.groupby(['LotShape'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)
plot1 = sns.boxenplot(data=df,x='LotShape',y="SalePrice",order=temp['LotShape'].to_list());
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);


# In[ ]:


temp = df.groupby(['Neighborhood'],as_index=False)['SalePrice'].median()
temp = temp.sort_values(by='SalePrice',ascending=False)
temp.style.background_gradient(cmap='Blues')


# from the color code we can see the costliest and cheapest neighborhood
