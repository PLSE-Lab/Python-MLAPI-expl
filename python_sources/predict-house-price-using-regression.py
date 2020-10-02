#!/usr/bin/env python
# coding: utf-8

# # Predict house price using regression

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


ds = pd.read_csv("../input/kc_house_data.csv")
ds.head()


# In[4]:


ds.describe()


# In[5]:


ds.dtypes


# In[9]:


ds = ds.drop(['id', 'date'], axis = 1)


# # Price distribution

# In[10]:


f, ax = plt.subplots(figsize=(12,5))
sns.distplot(ds.price, ax = ax, fit = stats.gausshyper)
plt.show()


# As we can see the price distribution is skewed to the right (Positive Skew). 

# In[11]:


f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x = 'price', data = ds, ax=ax, showmeans=True, fliersize=3, orient="h", color = "silver")
plt.show()

print('Min: ' + str(ds['price'].min()))
print('1 Q: ' + str(np.percentile(ds['price'], 25)))
print('Median:' + str(ds.price.median()))
print('3 Q: ' + str(np.percentile(ds['price'], 75)))
print('Max: ' + str(ds['price'].max()))


# The boxplot above shows us that there are many outliers. There are few prices above 500.000.  

# In[12]:


# add a new variable to analyse if the house is renovated
ds['is_renovated'] = ds['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)


# In[13]:


sns.countplot(x = ds.is_renovated, data = ds)
print(ds['is_renovated'].value_counts())


# As we can see there are many houses that were sold without a renewed.
# Lets see the correlation among variables to have an idea regarding the impact of each variable on house price.

# # Correlation coefficient

# In[14]:


# Continous and Categorical variables
# To biserial variables (i.e. is_renovated and waterfront) we could use stats.biserial (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html)
ds.drop(['view','grade','floors','bedrooms','bathrooms','condition'], axis = 1).corr(method = 'pearson')


# We can see that the variables 'sqft_living', 'sqft_above' and 'sqft_living15' have a significant positive relationship with price.

# In[15]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')


# Below we have plots to show us the relationship among continuous variables that have at least correlation coefficient higher than 0.3 with price.

# In[16]:


sns.jointplot(x = 'sqft_living', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'sqft_above', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'sqft_living15', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'sqft_basement', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'lat', y = 'price', data = ds, kind = 'reg')
plt.show()


# We can see that there are a lot of zeros in the plot that show the relationship between price and sqlt_basement. Maybe we can create another biserial variable based on sqlt_basement.
# 

# # Biserial variables

# In[17]:


#Price by waterfront
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="waterfront", y="price" , hue="waterfront", ax=ax, data=ds, dodge = False)
plt.show()


# Looking at boxplot above, the price varies more when the house has waterfront.

# In[18]:


#Price by is_renovated
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="is_renovated", y="price" , hue="is_renovated", ax=ax, data=ds, dodge = False)
plt.show()


# In[19]:


#Ordinal variables
ds[['price','view','grade','floors','bedrooms','bathrooms','condition']].corr(method = 'spearman')


# Let's analyse the price by grade. As we can see above, grade has a significant relationship with price.

# In[20]:


#Price by grade
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="grade", y="price" , hue="grade", ax=ax, data=ds, dodge = False);
plt.show()


# In[66]:


f, ax = plt.subplots(figsize=(12,5))
sns.countplot(x = ds.grade, data = ds)
plt.show()
ds.groupby(["grade"])["grade"].count()


# Let's see the value accumulated by grade.

# In[63]:


ds[['grade', 'price']].groupby('grade')['price'].sum().map('{:,.2f}'.format)


# In[67]:


#Price by view
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="view", y="price" , hue="view", ax=ax, data=ds, dodge = False);
plt.show()


# In[68]:


#Price by bedrooms
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="bedrooms", y="price" , hue="bedrooms", ax=ax, data=ds, dodge = False);
plt.show()


# In[49]:


#Price by bathrooms
f, ax = plt.subplots(figsize=(12,10))
sns.boxplot(x="bathrooms", y="price" , hue="bathrooms", ax=ax, data=ds, dodge = False)
plt.show()


# # Linear Regression - Model Version 1

# In[50]:


#Define X and Y
x = ds.drop(['price'], axis = 1)
y = ds['price'].values

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#create linear model
model_1 = linear_model.LinearRegression()

#train model
model_1_fit = model_1.fit(x_train, y_train)

#evaluating error
mean_squared_error(model_1_fit.predict(x_test), y_test)


# # Linear Regression - Model Version 2
# ### Apply MinMaxScaler

# In[51]:


#Define X and Y
x = ds.drop(['price'], axis = 1)
scaler = MinMaxScaler(feature_range=(0,1))
x = scaler.fit_transform(x)
y = ds['price'].values

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#create Linear Model
model_2 = linear_model.LinearRegression()

#train model
model_2_fit = model_2.fit(x_train, y_train)

#evaluating error
mean_squared_error(model_2_fit.predict(x_test), y_test)


# # Random Forest Regressor - Model Version 1

# In[69]:


#Define x and y
x = ds.drop(['price'], axis = 1)
y = ds['price'].values

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#train
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

#evaluating error
mean_squared_error(rf.predict(x_test), y_test)


# In[70]:


#Feature importance 
sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), ds.drop(['price'], axis = 1).columns), reverse=True)


# # Next steps:
# * Try to add dummy variables to deal with  view, grade, condition and floors and increase Linear Model performance
# * GridSearch to RandomForest
# * Find a way (learn how) to deal with zipcode
# 

# Feel free to send me your thoughts or questions about this Kernel. It Would be great.
