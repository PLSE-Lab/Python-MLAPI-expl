#!/usr/bin/env python
# coding: utf-8

# In[276]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer


# In[277]:


data1 = pd.read_csv("../input/train.csv")
data2 = pd.read_csv("../input/test.csv")


# In[278]:


data1.head()
# Top 5 rows


# In[279]:


data1.shape
# (rows,columns)


# In[280]:


data1.info()
# getting through data 


# ** data1.get_dtype_counts() --> Data types counts **

# In[281]:


data1.describe()
# statistical analysis (Numerical columns)


# Making correlation heatmap will not be a good idea !!! <br>
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(data1.corr(),annot=True)

# In[282]:


corr=data1.corr()["SalePrice"]
corr[np.argsort(corr,axis=0)[::-1]]


# Correlations are the best way to know the important columns.

# In[283]:


# For lots of columns dataset, and when finding correlations/anything visually

num_feat=data1.columns[data1.dtypes!=object]
num_feat=num_feat[1:-1] 
# first and last column excluded
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(data1[col].values, data1.SalePrice.values)[0,1])
    # np.corrcoef(dataframe[column].values,dataframe.Column.values)[range])
    
ind = np.arange(len(labels))
# Array from 0 to length of list 
width = 0.45
fig, ax = plt.subplots(figsize=(12,40))
# Plot size
ax.barh(ind, np.array(values), color='green')
# Horizontal bar(no.of bars,length of each,color)
ax.set_yticks(ind+((width)))
# green bars from title 
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Sale Price");


# In[284]:


# above 0.5 +ve correlation
corr=data1[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='magma',linecolor="black")
# (columns,max range,width of lines,square shape,color mapping,color of line)
plt.title('Correlation between features');


# In[285]:


# Categorical Variable compared with Target Variable 

#data1[['OverallQual','SalePrice']]
data1[['OverallQual','SalePrice']].groupby(['OverallQual'],
as_index=False).mean().sort_values(by='OverallQual', ascending=False)


# In[286]:


data1[['GarageCars','SalePrice']].groupby(['GarageCars'],
as_index=False).mean().sort_values(by='GarageCars', ascending=False)


# In[287]:


data1[['Fireplaces','SalePrice']].groupby(['Fireplaces'],
as_index=False).mean().sort_values(by='Fireplaces', ascending=False)


# <font size="3">**Univariate Analysis**</font>

# In[288]:


# Checking which numbers are frequently occuring in a column

sns.distplot(data1['SalePrice'], color="r", kde=False)
# sns.distplot(column, color, curve)
plt.title("Distribution of Sale Price")
plt.ylabel("Number of Occurences")
plt.xlabel("Sale Price");


# In[289]:


data1['SalePrice'].skew()


# Positive Skewness means when the tail on the right side of the distribution is longer or fatter. The mean and median will be greater than the mode

# In[290]:


data1['SalePrice'].kurt()


# Positive kurtosis. A distribution with a positive kurtosis value indicates that the distribution has heavier tails than the normal distribution. For example, data that follow a t distribution have a positive kurtosis value

# In[291]:


#upperlimit = np.percentile(houses.SalePrice.values, 99.5)
#houses['SalePrice'].ix[houses['SalePrice']>upperlimit] = upperlimit
#upperlimit
plt.scatter(range(data1.shape[0]), data1["SalePrice"].values,color='orange')
plt.title("Distribution of Sale Price")
plt.xlabel("Number of Occurences")
plt.ylabel("Sale Price");


# In[292]:


# One Way of dealing with outliers

upperlimit = np.percentile(data1.SalePrice.values, 99.5)
data1['SalePrice'].loc[data1['SalePrice']>upperlimit] = upperlimit
#upperlimit
plt.scatter(range(data1.shape[0]), data1["SalePrice"].values,color='orange')
plt.title("Distribution of Sale Price")
plt.xlabel("Number of Occurences")
plt.ylabel("Sale Price");


# In[293]:


null_columns=data1.columns[data1.isnull().any()]
labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(data1[col].isnull().sum())
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,50))
ax.barh(ind, np.array(values), color='r')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");


# In[294]:


# Counting missing values column-wise

missing_column = (data1.isnull().sum())
print(missing_column[missing_column > 0])

#null_columns=houses.columns[houses.isnull().any()]
#houses[null_columns].isnull().sum()


# In[295]:


# We have to fill na values, thus we can fill it with column values that is highly correlated with this column.
data1['LotFrontage'].corr(data1['LotArea'])


# In[296]:


# Take the square root and this increase the correlation
data1['SqrtLotArea']=np.sqrt(data1['LotArea'])
data1['LotFrontage'].corr(data1['SqrtLotArea'])


# In[297]:


# Draw a plot of two variables with bivariate and univariate graphs
sns.jointplot(data1['LotFrontage'],
              data1['SqrtLotArea'],
              color='red');


# In[298]:


filter = data1['LotFrontage'].isnull()
data1.LotFrontage[filter]=data1.SqrtLotArea[filter]


# In[299]:


missing_column = (data1.isnull().sum())
print(missing_column[missing_column > 0])


# Now checking what can be filled in place of missing values of "MasVnrArea" and "MasVnrTyp"

# In[300]:


plt.scatter(data1["MasVnrArea"],data1["SalePrice"])
plt.title("MasVnrArea Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Mas Vnr Area in sq feet");


# In[301]:


# For categorical variable missing values
sns.boxplot("MasVnrType","SalePrice",data=data1)


# In[302]:


data1["MasVnrType"] = data1["MasVnrType"].fillna('None')
data1["MasVnrArea"] = data1["MasVnrArea"].fillna(0.0)


# In[303]:


missing_column = (data1.isnull().sum())
print(missing_column[missing_column > 0])


# In[304]:


sns.boxplot("Electrical","SalePrice",data=data1)
plt.title("Electrical Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Electrical")


# In[305]:


data1["Electrical"] = data1["Electrical"].fillna('SBrkr')


# In[306]:


missing_column = (data1.isnull().sum())
print(missing_column[missing_column > 0])


# In[307]:


#sns.boxplot("Alley","SalePrice",data=data1)
sns.stripplot(x=data1["Alley"], y=data1["SalePrice"],jitter=True);


# In[308]:


data1["Alley"] = data1["Alley"].fillna('None')


# In[309]:


missing_column = (data1.isnull().sum())
print(missing_column[missing_column > 0])


# In[310]:


X_train = data1.drop('SalePrice',axis=1)
X_train = X_train.select_dtypes(exclude=['object'])
y_train = data1.SalePrice
X_train = X_train.drop('SqrtLotArea',axis=1)


# In[311]:


X_test = data2.select_dtypes(exclude=['object'])


# In[312]:



imputed_X_train = X_train.copy()
imputed_X_test = X_test.copy()
# Copying the orginal data,original data should not change(avoid it)
col_missing_val = (col for col in X_train.columns if X_train[col].isnull().any())
# Any column having missing values, it will be put into above variable
for col in col_missing_val:
    imputed_X_train[col +'_was_missing'] = imputed_X_train[col].isnull()
    imputed_X_test[col +'_was_missing'] = imputed_X_test[col].isnull()
#Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(imputed_X_train)
imputed_X_test = my_imputer.transform(imputed_X_test)

# Predicting prices
model = RandomForestRegressor()
model.fit(imputed_X_train,y_train)
preds = model.predict(imputed_X_test)
print(preds)


# In[313]:


submission = pd.DataFrame({'Id': data2.Id, 'SalePrice': preds})
# you could use any filename. We choose submission here
submission.to_csv('FirstCompetition.csv', index=False)

