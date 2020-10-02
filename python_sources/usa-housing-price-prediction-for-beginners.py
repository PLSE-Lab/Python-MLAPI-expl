#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df  = pd.read_csv('../input/USA_Housing.csv')


# Before jumping into code directly its import to follow some basics. I have divided this kernel into 7 parts.
# 
# * Understanding your data
# * Data Cleaning(if required)
# * Outlier treatment
# * Scaling the data into machine learning readbable format
# * Dividing the data into train/test
# * Applying machine learning algorithm
# * Testing the effectiveness of the machine learning Algorithm
# * Trying different Algorithms

# **Understanding the data**

# In[ ]:


#let's see the number of rows and columns
df.shape


# In[ ]:


df.info()


# * Six columns have Float64 datatype.
# * One column(Address)  have object datatype.

# In[ ]:


# So, we have 6 countinuos variable and one categorical variable. 
# Now, lets look at some of the data.
df.head()


# In[ ]:


df.describe()


# * Describe method will show the statitical information of continues variable column.

# In[ ]:


# lets see if there is any null/missing values in the datasets or not. It's important to remove or
# replace all the missing values before moving further. 
df.isna().sum()


# In[ ]:


# We don't have any missing values. So, we are good to go. 
# Now, let's understand the correlation between variable by plotting correlation plot.
df.corr()


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


# As we can see that price is more correlated to Avg. Income Area, House Age and Area Population than Number of Bedrooms and Rooms. Lets see these metrics in tabular format.
df.corr().Price.sort_values(ascending=False)


# In[ ]:


sns.pairplot(df)


# In[ ]:


# As we can see here in the last line of graph that all the features seems to be in a linear relationship with price except Avg. Area Number of Bedroom.
# We can also see this by plotting a separate graph

plt.scatter(df.Price, df[['Avg. Area Income']])


# In[ ]:


sns.distplot(df.Price)


# In[ ]:


# We can see the price plot seems like a bell shaped curve and all the price is normally distributed.


# **Detection and remove of Outliers**

# In[ ]:


plt.figure(figsize=(12,10))
plt.subplot(2,3,1)
plt.title("price")
plt.boxplot(df.Price)

plt.subplot(2,3,2)
plt.title("Average income")
plt.boxplot(df["Avg. Area Income"])
plt.subplot(2,3,3)
plt.title("Avg. Area House Age")
plt.boxplot(df["Avg. Area House Age"])
plt.subplot(2,3,4)
plt.title("Avg. Area Number of Rooms")
plt.boxplot(df["Avg. Area Number of Rooms"])
plt.subplot(2,3,5)
plt.title("Avg. Area Number of Bedrooms")
plt.boxplot(df["Avg. Area Number of Bedrooms"])
plt.subplot(2,3,6)
plt.title("Area Population")
plt.boxplot(df["Area Population"])


# Till now, we had a detailed look at the given data and fortunately we don't have any missing values. So, the data cleaning is not required for this data.
# 
# We will also be deleting the address data, as it does not seem that useful.'

# In[ ]:


df = df.drop(['Address'], axis=1)
df.head()


# ![](http://)**Scale the data - Prepare the data to feed into the machine learning algorithm**

# In[ ]:


from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()


# In[ ]:


feature = df.drop(['Price'], axis = 1)
label = df.Price

# Now, we have feature and label for machine learning algorithms. Now, we can scale the data by using standard scaler.

feature = pre_process.fit_transform(feature)


# In[ ]:


feature 
#this is how the scaled data looks like.


# ![](http://)**Divding the data into train/test**

# In[ ]:


from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, label.values, test_size = 0.2, random_state = 19)


# **Applying machine learning algorithm**

# In[ ]:


from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(feature_train, label_train)


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error

score = r2_score(linear_regression.predict(feature_train), label_train)
error = mean_squared_error(linear_regression.predict(feature_train), label_train)


# In[ ]:


score, error


# In[ ]:


linear_regression.coef_


# In[ ]:


linear_regression.intercept_


# In[ ]:


pd.DataFrame(linear_regression.coef_, index=df.columns[:-1], columns=['Values'])


# In[ ]:


# Applying this on test data.
score_test = r2_score(linear_regression.predict(feature_test), label_test)
score_test


# Some Other Algorithms
# 
# **RANSAC Regressor**

# In[ ]:


ransac = linear_model.RANSACRegressor()
ransac.fit(feature_train, label_train)

# Scoring the Ransac model

ransac_r2_score = r2_score(ransac.predict(feature_test), label_test)
ransac_r2_score


# **Ridge Regression**

# In[ ]:


ridge_model = linear_model.Ridge()
ridge_model.fit(feature_train, label_train)

# Scoring the Ridge Regression

ridge_r2_score = r2_score(ridge_model.predict(feature_test), label_test)
ridge_r2_score


# ![](http://)**Testing Decision Tree Model**

# In[ ]:


from sklearn import tree
tree_model = tree.DecisionTreeRegressor()
tree_model.fit(feature_train, label_train)

# Scoring the Ridge Regression

tree_r2_score = r2_score(tree_model.predict(feature_test), label_test)
tree_r2_score


# **Random Forest Regressor**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
random_model = RandomForestRegressor()
random_model.fit(feature_train, label_train)

# Scoring the Ridge Regression

random_r2_score = r2_score(tree_model.predict(feature_test), label_test)
random_r2_score


# In[ ]:


data = [score_test, ransac_r2_score, ridge_r2_score, tree_r2_score,random_r2_score]
index = ['Linear Regression', 'Ransac Regression', 'Ridge Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
pd.DataFrame(data, index=index, columns=['Scores']).sort_values(ascending = False, by=['Scores'])


# In[ ]:




