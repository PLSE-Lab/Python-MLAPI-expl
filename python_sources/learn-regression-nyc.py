#!/usr/bin/env python
# coding: utf-8

# # 1.Loading libraries and Dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from scipy import stats


# In[ ]:


#Reading Dataset
df = pd.read_csv('../input/nyc-rolling-sales.csv')


# In[ ]:


# Little peek into the dataset
df.head()


# In[ ]:


#Dropping column as it is empty
del df['EASE-MENT']
#Dropping as it looks like an iterator
del df['Unnamed: 0']

del df['SALE DATE']


# In[ ]:


#Checking for duplicated entries
sum(df.duplicated(df.columns))


# In[ ]:


#Delete the duplicates and check that it worked
df = df.drop_duplicates(df.columns, keep='last')
sum(df.duplicated(df.columns))


# # 2.Data Inspection & Visualization

# In[ ]:


#shape of dataset
df.shape


# In[ ]:


#Description of every column
df.info()


# In[ ]:


#Let's convert some of the columns to appropriate datatype

df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')
df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
#df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['BOROUGH'] = df['BOROUGH'].astype('category')


# In[ ]:


#checking missing values

df.columns[df.isnull().any()]


# In[ ]:


miss=df.isnull().sum()/len(df)
miss=miss[miss>0]
miss.sort_values(inplace=True)
miss


# In[ ]:


miss=miss.to_frame()
miss.columns=['count']
miss.index.names=['Name']
miss['Name']=miss.index
miss


# In[ ]:


#plot the missing values
sns.set(style='whitegrid',color_codes=True)
sns.barplot(x='Name', y='count',data=miss)
plt.xticks(rotation=90)
sns


# There are many missing values in the columns : 
# * LAND SQUARE FEET
# * GROSS SQUARE FEET
# * SALE PRICE
# 
# We can drop the rows with missing values or we can fill them up with their mean, median or any other relation.
# 
# For time being, let's fill these up with mean values.<br>
# Further, We will try to predict the value of SALE PRICE as test data.

# In[ ]:


# For time being, let's fill these up with mean values.
df['LAND SQUARE FEET']=df['LAND SQUARE FEET'].fillna(df['LAND SQUARE FEET'].mean())
df['GROSS SQUARE FEET']=df['GROSS SQUARE FEET'].fillna(df['GROSS SQUARE FEET'].mean())


# In[ ]:


# Splitting dataset 
test=df[df['SALE PRICE'].isna()]
data=df[~df['SALE PRICE'].isna()]


# In[ ]:


test = test.drop(columns='SALE PRICE')


# In[ ]:


# Print first 5 rows of test
print(test.shape)
test.head()


# In[ ]:


#Printing first rows of our data
print(data.shape)
data.head(10)


# In[ ]:


#correlation between the features
corr = data.corr()
sns.heatmap(corr)


# Last row represents the correlation of different features with SALE PRICE

# In[ ]:


#numeric correlation
corr['SALE PRICE'].sort_values(ascending=False)


# In[ ]:


numeric_data=data.select_dtypes(include=[np.number])
numeric_data.describe()


# 
# **SALE PRICE**

# In[ ]:


plt.figure(figsize=(15,6))

sns.boxplot(x='SALE PRICE', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()


# In[ ]:


sns.distplot(data['SALE PRICE'])


# In[ ]:


# Remove observations that fall outside those caps
data = data[(data['SALE PRICE'] > 100000) & (data['SALE PRICE'] < 5000000)]


# Let's Check Again

# In[ ]:


sns.distplot(data['SALE PRICE'])


# In[ ]:


#skewness of SalePrice
data['SALE PRICE'].skew()


# SALE PRICE is highly right skewed. So, we will log transform it so that it give better results.

# In[ ]:


sales=np.log(data['SALE PRICE'])
print(sales.skew())
sns.distplot(sales)


# Well now we can see the symmetry and thus it is normalised.

# **Let's Visualize Numerical data**

# **SQUARE FEET**

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='GROSS SQUARE FEET', data=data,showfliers=False)


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='LAND SQUARE FEET', data=data,showfliers=False)


# In[ ]:


data = data[data['GROSS SQUARE FEET'] < 10000]
data = data[data['LAND SQUARE FEET'] < 10000]


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=data, fit_reg=False, scatter_kws={'alpha':0.3})


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x='LAND SQUARE FEET', y='SALE PRICE', data=data, fit_reg=False, scatter_kws={'alpha':0.3})


# **Total Units, Commercial Units, Residential Units**

# In[ ]:


data[["TOTAL UNITS", "SALE PRICE"]].groupby(['TOTAL UNITS'], as_index=False).count().sort_values(by='SALE PRICE', ascending=False)


# Removing rows with TOTAL UNITS == 0 and one outlier with 2261 units

# In[ ]:


data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] != 2261)] 


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='TOTAL UNITS', y='SALE PRICE', data=data)
plt.title('Total Units vs Sale Price')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='COMMERCIAL UNITS', y='SALE PRICE', data=data)
plt.title('Commercial Units vs Sale Price')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='RESIDENTIAL UNITS', y='SALE PRICE', data=data)
plt.title('Residential Units vs Sale Price')
plt.show()


# **Let's Visualize categorical data**

# In[ ]:


cat_data=data.select_dtypes(exclude=[np.number])
cat_data.describe()


# **TAX CLASS AT PRESENT**

# In[ ]:


# Starting with TAX CLASS AT PRESENT
data['TAX CLASS AT PRESENT'].unique()


# In[ ]:


pivot=data.pivot_table(index='TAX CLASS AT PRESENT', values='SALE PRICE', aggfunc=np.median)
pivot


# In[ ]:


pivot.plot(kind='bar', color='black')


# **TAX CLASS AT TIME OF SALE**

# In[ ]:


#  TAX CLASS AT TIME OF SALE
data['TAX CLASS AT TIME OF SALE'].unique()


# In[ ]:


pivot=data.pivot_table(index='TAX CLASS AT TIME OF SALE', values='SALE PRICE', aggfunc=np.median)
pivot


# In[ ]:


pivot.plot(kind='bar', color='red')


# **BOROUGH**

# In[ ]:


# BOROUGH
data['BOROUGH'].unique()


# In[ ]:


pivot=data.pivot_table(index='BOROUGH', values='SALE PRICE', aggfunc=np.median)
pivot


# In[ ]:


pivot.plot(kind='bar', color='blue')


# ***It means max sale price is of BOROUGH==1 that is Manhattan.***

# **BUILDING CLASS CATEGORY**

# In[ ]:


# BUILDING CLASS CATEGORY
print(data['BUILDING CLASS CATEGORY'].nunique())

pivot=data.pivot_table(index='BUILDING CLASS CATEGORY', values='SALE PRICE', aggfunc=np.median)
pivot


# In[ ]:


pivot.plot(kind='bar', color='Green')


# # 3. Data Pre Processing

# **Let's see our dataset again**

# In[ ]:


del data['ADDRESS']
del data['APARTMENT NUMBER']


# In[ ]:


data.info()


# **Normalising and Transforming Numerical columns**

# In[ ]:


numeric_data.columns


# In[ ]:


#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = data[numeric_data.columns].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
data[skewed] = np.log1p(data[skewed])


# In[ ]:



scaler = StandardScaler()
scaler.fit(data[numeric_data.columns])
scaled = scaler.transform(data[numeric_data.columns])

for i, col in enumerate(numeric_data.columns):
       data[col] = scaled[:,i]


# In[ ]:


data.head()


# In[ ]:


#Dropping few columns
del data['BUILDING CLASS AT PRESENT']
del data['BUILDING CLASS AT TIME OF SALE']
del data['NEIGHBORHOOD']


# **One hot encoding categorical columns**

# In[ ]:


#Select the variables to be one-hot encoded
one_hot_features = ['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']


# In[ ]:


# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(data[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)


# In[ ]:


# Replacing categorical columns with dummies
fdf = data.drop(one_hot_features,axis=1)
fdf = pd.concat([fdf, one_hot_encoded] ,axis=1)


# In[ ]:


fdf.info()


# ## Train/Test Split

# In[ ]:


Y_fdf = fdf['SALE PRICE']
X_fdf = fdf.drop('SALE PRICE', axis=1)

X_fdf.shape , Y_fdf.shape


# In[ ]:


X_train ,X_test, Y_train , Y_test = train_test_split(X_fdf , Y_fdf , test_size = 0.3 , random_state =34)


# In[ ]:


# Training set
X_train.shape , Y_train.shape


# In[ ]:


#Testing set
X_test.shape , Y_test.shape


# # 4. Modelling

# In[ ]:


# RMSE
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))


# ### 4.1 Linear Regression

# In[ ]:


linreg = LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred_lin = linreg.predict(X_test)
rmse(Y_test,Y_pred_lin)


# ### 4.2. Lasso Regression

# In[ ]:


alpha=0.00099
lasso_regr=Lasso(alpha=alpha,max_iter=50000)
lasso_regr.fit(X_train, Y_train)
Y_pred_lasso=lasso_regr.predict(X_test)
rmse(Y_test,Y_pred_lasso)


# ### 4.3. Ridge Regression

# In[ ]:


ridge = Ridge(alpha=0.01, normalize=True)
ridge.fit(X_train, Y_train)
Y_pred_ridge = ridge.predict(X_test)
rmse(Y_test,Y_pred_ridge)


# ### 4.4. RandomForest Regressor

# In[ ]:


rf_regr = RandomForestRegressor()
rf_regr.fit(X_train, Y_train)
Y_pred_rf = rf_regr.predict(X_test)
rmse(Y_test,Y_pred_rf)


# # 5. Conclusion

# **We can see that Random Forest Regressor works best for this dataset with RSME score of 0.588**

# **Please UPVOTE if found useful !**

# In[ ]:




