#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


data_path = "../input/housing.csv"
df = pd.read_csv(data_path)


# This kernel is divided into 7 parts.  The most important part of  "house price prediction" is knowing the data we are working with. Almost 75% of the prediction effort goes into getting familiar with data, data cleaning and making the data ready for maching learning algorithms. So, broadly you can divide the seven substep into 2 major categories.
# 
# **Working with data**
# 
# * Get to know your data
# * Data Cleaning(if required)
# * Scaling the data into machine learning readbable format
# * Dividing the data into train/test
# 
# **Working with Machine Learning Algorithms**
# 
# * Applying machine learning algorithm
# * Testing the effectiveness of the machine learning Algorithm
# * Trying different Algorithms

#  **Get to know your data**

# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


# In the given datasets we have 9 continuous variables and one categorical variable. ML algorithms do not work well with categorical data. 
# So, we will convert the categorical data. 
df.columns


# **Working with categorical data**

# In[ ]:


df.ocean_proximity.value_counts()


# In[ ]:


sns.countplot(df.ocean_proximity)


# In[ ]:


new_val = pd.get_dummies(df.ocean_proximity)


# In[ ]:


new_val.head(5)


# In[ ]:


df[new_val.columns] = new_val


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income'
       , '<1H OCEAN', 'INLAND',
       'ISLAND', 'NEAR BAY', 'NEAR OCEAN','median_house_value']]


# In[ ]:


df.describe() 


# In[ ]:


# Now, let's understand the correlation between variable by plotting correlation plot


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


df.corr().sort_values(ascending=False, by = 'median_house_value').median_house_value


# In[ ]:


df.hist(figsize=(15,12))


# In[ ]:


df.median_house_value.hist()


# In[ ]:


sns.distplot(df.median_house_value)


# In[ ]:


# We can see that the median house value is mostly falls between 10,0000 to 30,0000 with few exceptions. 


# **Data Cleaning**

# In[ ]:


# We will need to replace all the null values.
df.isna().sum() 


# In[ ]:


# So, we have 207 null values. We can drop the rows with null values or we can replace the null values.
# 207 is too big a number to drop rows
df = df.fillna(df.mean())


# In[ ]:


df.isna().sum() 


# **Data Scaling**

# In[ ]:


from sklearn import preprocessing
convert = preprocessing.StandardScaler() 


# In[ ]:


df.columns 


# In[ ]:


feature = df.drop(['median_house_value'], axis=1)
label = df.median_house_value


# In[ ]:


featureT = convert.fit_transform(feature.values)
labelT = convert.fit_transform(df.median_house_value.values.reshape(-1,1)).flatten() 


# In[ ]:


featureT


# In[ ]:


labelT


# **Split the data into train and test**

# In[ ]:


from sklearn.model_selection import train_test_split
feature_train, feature_test,label_train, label_test = train_test_split(featureT,labelT, test_size=0.2, random_state=19)                                   


# **ML Model - Linear Regression**

# In[ ]:


from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
linear_reg = linear_model.LinearRegression()
linear_reg.fit(feature_train,label_train)
r2_score(linear_reg.predict(feature_train),label_train)


# **Cross Validation Score**

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score(linear_reg, feature_train,label_train, cv=10) 


# In[ ]:


reg_score = r2_score(linear_reg.predict(feature_test),label_test) 


# In[ ]:


reg_score


# In[ ]:


linear_reg.coef_


# In[ ]:


pd.DataFrame(linear_reg.coef_, index=feature.columns, columns=['Coefficient']).sort_values(ascending=False, by = 'Coefficient')


# In[ ]:


df.corr().median_house_value.sort_values(ascending=False) 


# ****RANSAC Regression****

# In[ ]:


ransac_reg = linear_model.RANSACRegressor()


# In[ ]:


ransac_reg.fit(feature_train,label_train)
r2_score(ransac_reg.predict(feature_train),label_train)


# In[ ]:


ransac_score = r2_score(ransac_reg.predict(feature_test),label_test)


# In[ ]:


ransac_score


# In[ ]:


# Ransac regrssor is performing way poorly than Linear Regresson


# **Ridge Regressor**

# In[ ]:


ridge_reg = linear_model.Ridge(random_state=19) 
ridge_reg.fit(feature_train,label_train) 


# In[ ]:


r2_score(ridge_reg.predict(feature_train),label_train)


# In[ ]:


ridge_score = r2_score(ridge_reg.predict(feature_test),label_test) 


# In[ ]:


ridge_score


# In[ ]:


ridge_reg.coef_


# In[ ]:


pd.DataFrame(ridge_reg.coef_, index=feature.columns, columns=['Coefficient']).sort_values(ascending=False, by = 'Coefficient') 


# **Decision Tree Regressor**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(feature_train,label_train)


# In[ ]:


r2_score(tree_reg.predict(feature_train),label_train)


# In[ ]:


# 99% seems like overfitting. Let's cross validate it.

cross_val_score(tree_reg, feature_train, label_train, cv=10)


# In[ ]:


tree_score = r2_score(tree_reg.predict(feature_test),label_test) 
tree_score


# **Random Forest Regressor**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()


# In[ ]:


forest_reg.fit(feature_train,label_train)


# In[ ]:


r2_score(forest_reg.predict(feature_train),label_train)


# In[ ]:


cross_val_score(forest_reg, feature_train, label_train, cv=10)


# In[ ]:


# let's see how well the random forest regressor fits well with the test data
forest_score = r2_score(forest_reg.predict(feature_test),label_test) 


# In[ ]:


forest_score


# In[ ]:


# 76% is not a bad score. We can also use GridSearchCV to find the best paramters for random forest regressor


# In[ ]:


data = [reg_score, ransac_score, ridge_score, tree_score, forest_score]
index = ['Linear Regression', 'Ransac Regression', 'Ridge Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
pd.DataFrame(data, index=index, columns=['Scores']).sort_values(ascending = False, by=['Scores'])


# In[ ]:


# So, the random forest regressor is winner here out of all the ML Algorithm


# In[ ]:




