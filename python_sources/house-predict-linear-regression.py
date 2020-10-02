#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This model is using Linear Regression.


# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the dataset and extracting the Independent and Dependent Variables

house = pd.read_csv('../input/house.csv')
X = house.iloc[:,:-1].values
y = house.iloc[:,20].values

house.head()


# In[ ]:


y


# In[ ]:


#Data Visualization
# Corelation of various parameters

fig, ax = plt.subplots(figsize=(20,15))

sns.heatmap(house.corr())


# In[ ]:


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

X[:,1] = labelencoder.fit_transform(X[:,1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()


# In[ ]:


# Avoiding dummy variables

X = X[:,1:]


# In[ ]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[ ]:


# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)


# In[ ]:


# Calculating the Co-efficient
print(regressor.coef_)


# In[ ]:


# Calculating the Intercept
print(regressor.intercept_)


# In[ ]:


# Calculating the R Squared Value
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# **It is worth to note here that the dataset has many outliers and hence the efficiency is less.
# lets dig further to find where is the outlier**

# In[ ]:


fig=sns.jointplot(y='price',x='bedrooms',data=house)
plt.show()


# ***As you can see the price of 4-6 bedrooms is ranging from 1,000,000 to 8,000,000. Which is an obvious outlier.
# lets dig in further to see where this outlier is coming from****

# In[ ]:


ax = sns.boxplot(x=house['price'])


# In[ ]:


ay = sns.boxplot(x='bedrooms',y='price', data = house, width= 0.6)


# 

# In[ ]:


house.groupby('bedrooms')['price'].describe()


# **Conclusion : We can further improve this model by taking the mean for all the bedrooms, and removing the outliers from each bedroom types.
# 
# 
# This could also mean that those outliers, may have some additional features, such as beach facing home or a house with a costlier locality. All these parameters will be considered before removing any outlier.
# 
# Do you specifically have any suggessions to improve the efficiency?**

# In[ ]:




