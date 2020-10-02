#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd                 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import datetime
import pandas_profiling
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, ElasticNet, Ridge, Lasso


# # for visualization

# In[ ]:


import matplotlib.pyplot as plt     
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Using Pandas library to read in csv files. The pd.read_csv() method creates a DataFrame from a csv file

# In[ ]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# checking out the size of the data

# In[ ]:


print("Train data shape:", train.shape)
print("Test data shape:", test.shape)


# Looking at  few rows using the DataFrame.head() method

# In[ ]:


train.head(2)


# Getting more information like count, mean, std, min, max 

# In[ ]:


train['price']=train['data-price']


# In[ ]:


#train.SalePrice.describe()
print (train.price.describe())


# plot a  distribution of SalePrice using seaborn

# In[ ]:


print ("Skew is:", train.price.skew())
sns.distplot(train.price, color='magenta')
plt.show()


# In[ ]:


target = np.log(train.price)


# In[ ]:


print ("Skew is:", target.skew())
sns.distplot(target, color='magenta')
plt.show()


# return a subset of columns matching the specified data types

# In[ ]:


cor=train.corr()
cor


# In[ ]:


#numeric features.dtypes
numeric_data = train.select_dtypes(include=[np.number])
print(numeric_data.dtypes)


# displays the correlation between the columns and examine the correlations between the features and the target

# In[ ]:


corr = numeric_data.corr()


# The first six features are the most positively correlated with SalePrice, while the next five are the most negatively correlated.

# In[ ]:


train.corr()['price'].sort_values(ascending=False)


# In[ ]:


sns.pairplot(numeric_data)


# In[ ]:


sns.heatmap(corr,annot=True)


# We  generate some scatter plots and visualize the relationship between the bathroom   and price

# In[ ]:


#We set index='Arae' and values='SalePrice'. We chose to look at the median here.
quality_pivot = train.pivot_table(index='area', values='price', aggfunc=np.median)
print(quality_pivot)


# In[ ]:


#visualize this pivot table more easily, we can create a bar plot
plt.rcParams['figure.figsize'] = (10, 8)
quality_pivot.plot(kind='bar', color='b')
plt.xlabel('Area')
plt.ylabel('Median Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


plt.scatter(x=train['bathroom'], y=train['price'],color='b')
plt.ylabel('Price')
plt.xlabel('bathroom')
plt.show()


# In[ ]:


plt.scatter(x=train['buildingSize'], y=train['price'],color='b')
plt.ylabel('Price')
plt.xlabel('buildingSize')
plt.show()


# In[ ]:


pandas_profiling.ProfileReport(train)


# In[ ]:


tes=test.merge(train,on='house-id')


# ## Handling Null Values

# In[ ]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls.head()


# We check columns with the missing values using the heatmap

# In[ ]:


plt.rcParams['figure.figsize'] = (10, 6)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# columns where NaN values have meaning e.g. no pool etc

# In[ ]:


train.columns


# In[ ]:


cols=[ 'bathroom','garage']


# In[ ]:


for i in cols:
    train[i].fillna(0,inplace=True)
    test[i].fillna(train[i].mean(),inplace=True)


# In[ ]:


col=[ 'buildingSize','erfSize','bedroom']


# In[ ]:


for i in col:
    train[i].fillna(train[i].mean(),inplace=True)
    test[i].fillna(train[i].mean(),inplace=True)


# A list of the unique values

# In[ ]:


print ("Unique values are:", train.type.unique())


# In[ ]:


print(train.area.unique())


# In[ ]:


print(train['data-isonshow'].unique())


# In[ ]:


cat_data = train.select_dtypes(exclude=[np.number])
#categoricals.describe()
cat_data.dtypes


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Feature Engineering

# In[ ]:


train['data-date']= pd.to_datetime(train['data-date'])
train['year']= train['data-date'].dt.year
test['data-date']= pd.to_datetime(test['data-date'])
test['year']= test['data-date'].dt.year


# In[ ]:


train.head()


# In[ ]:


# Dummy variables added

train = pd.get_dummies(train, columns = ['data-isonshow'], prefix ='data', drop_first = True)
train = pd.get_dummies(train, columns = ['area'], prefix ='area', drop_first = True)
train = pd.get_dummies(train, columns = ['data-location'], prefix ='data-location', drop_first = True)
train= pd.get_dummies(train, columns = ['type'], prefix ='type', drop_first = True)


# In[ ]:


test = pd.get_dummies(test, columns = ['data-isonshow'], prefix ='data', drop_first = True)
test = pd.get_dummies(test, columns = ['area'], prefix ='area', drop_first = True)
test = pd.get_dummies(test, columns = ['data-location'], prefix ='data-location', drop_first = True)
test= pd.get_dummies(test, columns = ['type'], prefix ='type', drop_first = True)


# In[ ]:


test.head(2)


# In[ ]:


train.drop(columns=['data-date','data-price','data-url',],inplace=True)
test.drop(columns=['data-date','data-url'],inplace=True)


# In[ ]:





# In[ ]:


test.head(2)


# ### Let's Build a linear model   

# In[ ]:


y = target
X = train.drop(['price','house-id'], axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# In[ ]:


lm = linear_model.LinearRegression()


# In[ ]:


model = lm.fit(X_train, y_train)


# In[ ]:


predictions = model.predict(X_train)


# In[ ]:


print('RMSE is: \n', mean_squared_error(y_train, predictions))


# In[ ]:


plt.scatter(predictions, y_train,color='blue') 
plt.xlabel('Predicted Price')
plt.ylabel('y_train(Actual Price)')
plt.title('Linear Regression Model')
plt.show()


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


print('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[ ]:


plt.scatter(predictions, y_test,color='b') 
plt.xlabel('Predicted Price')
plt.ylabel('y_test(Actual Price)')
plt.title('Linear Regression Model')
plt.show()


# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


ridge = Ridge(alpha=1)


# In[ ]:


ridge.fit(X_train, y_train)


# In[ ]:


coeff = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficient'])


# In[ ]:


coeff.head()


# In[ ]:


predictions= ridge.predict(X_train)


# In[ ]:


# calculates the rmse
print('RMSE is: \n', mean_squared_error(y_train, predictions))


# In[ ]:


predictions= ridge.predict(X_test)
len(predictions)


# In[ ]:


# calculates the rmse
print('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[ ]:


model_lasso = Lasso(alpha=0.00055)
model_lasso.fit(X_train, y_train)

predictions = model_lasso.predict(X_train)


# In[ ]:


print('RMSE is: \n', mean_squared_error(y_train, predictions))


# In[ ]:


predictions = model_lasso.predict(X_test)


# In[ ]:


print('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfor = RandomForestRegressor(random_state=42)
rfor.fit(X_train,y_train)


# In[ ]:


predictions=rfor.predict(X_train)


# In[ ]:


print('RMSE is: \n', mean_squared_error(y_train, predictions))


# In[ ]:


predictions=rfor.predict(X_test)
len(predictions)


# In[ ]:


print('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




