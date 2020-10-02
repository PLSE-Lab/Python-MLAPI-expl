#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input")) 


# In[ ]:


data_path = "../input/abalone.csv"
df = pd.read_csv(data_path) 


# In[ ]:


df.head() 


# In[ ]:


df.info() 


# In[ ]:


df.describe() 


# In[ ]:


# It seems height is 0 for some rows.
df = df[df.Height > 0]


# In[ ]:


df.info() 


# In[ ]:


df.describe()  


# In[ ]:


#  Our job is to predict the age of the Ring on the given feature. So, let look at the Ring in detail.

plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.countplot(df.Rings)

plt.subplot(2,2,2)
sns.distplot(df.Rings)

plt.subplot(2,2,3)
stats.probplot(df.Rings, plot=plt)

plt.subplot(2,2,4)
sns.boxplot(df.Rings) 

plt.tight_layout()

#It seems that the label value is skewed after 15 years of age. We will deal with that in a latter.df.describe()  


# In[ ]:


plt.figure(figsize=(12,10))
sns.pairplot(df) 


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True)


# It seems the most of the matrices are corelated with the length, if the length is more, weight will be more, height will be more and dimeter will be more, but
# our job is to predict Rings age. So, let's focus on that.

# In[ ]:


df.corr().Rings.sort_values(ascending=False) 


# In[ ]:


plt.figure(figsize=(15, 15))

plt.subplot(3,3,1)
plt.title('Shell weight vs Rings')
plt.scatter(df['Shell weight'],df['Rings'])

plt.subplot(3,3,2)
plt.title('Diameter vs Rings')
plt.scatter(df['Diameter'],df['Rings'])

plt.subplot(3,3,3)
plt.title('Height vs Rings')
plt.scatter(df['Height'],df['Rings'])

plt.subplot(3,3,4)
plt.title('Length vs Rings')
plt.scatter(df['Length'],df['Rings'])

plt.subplot(3,3,5)
plt.title('Whole weight vs Rings')
plt.scatter(df['Whole weight'],df['Rings'])

plt.subplot(3,3,6)
plt.title('Viscera weight vs Rings')
plt.scatter(df['Viscera weight'],df['Rings'])

plt.tight_layout()


# In[ ]:


# As we can see that the data we have at disposal is great for predicting the Rings between 3 to 15 years.

new_df = df[df.Rings < 16]
new_df = new_df[new_df.Rings > 2]


# In[ ]:


plt.figure(figsize=(12,6))
sns.violinplot(data=new_df, x='Rings', y='Length') 


# In[ ]:


new_df.head()


# In[ ]:


new_df.info()


# In[ ]:


plt.figure(figsize=(12, 10))

plt.subplot(3,3,1)
plt.title('Shell weight vs Rings')
plt.scatter(new_df['Shell weight'],new_df['Rings'])

plt.subplot(3,3,2)
plt.title('Diameter vs Rings')
plt.scatter(new_df['Diameter'],new_df['Rings'])

plt.subplot(3,3,3)
plt.title('Height vs Rings')
plt.scatter(new_df['Height'],new_df['Rings'])

plt.subplot(3,3,4)
plt.title('Length vs Rings')
plt.scatter(new_df['Length'],new_df['Rings'])

plt.subplot(3,3,5)
plt.title('Whole weight vs Rings')
plt.scatter(new_df['Whole weight'],new_df['Rings'])

plt.subplot(3,3,6)
plt.title('Viscera weight vs Rings')
plt.scatter(new_df['Viscera weight'],new_df['Rings'])

plt.tight_layout()


# In[ ]:


# there seems to be few outliers, we can remove them outliers
new_df = new_df[new_df.Height < 0.4]


# In[ ]:


plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
sns.boxplot(data= new_df, x = 'Rings', y = 'Diameter')

plt.subplot(3,2,2)
sns.boxplot(data= new_df, x = 'Rings', y = 'Length')

plt.subplot(3,2,3)
sns.boxplot(data= new_df, x = 'Rings', y = 'Height')

plt.subplot(3,2,4)
sns.boxplot(data= new_df, x = 'Rings', y = 'Shell weight')

plt.subplot(3,2,5)
sns.boxplot(data= new_df, x = 'Rings', y = 'Whole weight')

plt.subplot(3,2,6)
sns.boxplot(data= new_df, x = 'Rings', y = 'Viscera weight')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.countplot(new_df.Rings)

plt.subplot(2,2,2)
sns.distplot(new_df.Rings)

plt.subplot(2,2,3)
stats.probplot(new_df.Rings, plot=plt)

plt.subplot(2,2,4)
sns.boxplot(new_df.Rings)

plt.tight_layout()


# In[ ]:


# Everything looks normal, but we can see from the the changes in all the variable looks constant after 11 years. 
# We can segregate the data more from 3 to 15 to 3 to 10 for better resulls.


# > ****Regression ****

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


new_df.columns


# In[ ]:


new_col = pd.get_dummies(new_df.Sex)
new_df[new_col.columns] = new_col


# In[ ]:


new_df.columns


# In[ ]:


feature = new_df.drop(['Sex', 'Rings'], axis = 1)
label = new_df.Rings


# In[ ]:


from sklearn.preprocessing import StandardScaler
convert = StandardScaler()
feature = convert.fit_transform(feature)


# In[ ]:


feature.shape, label.shape


# In[ ]:


from sklearn.model_selection import train_test_split
f_train, f_test, l_train, l_test = train_test_split(feature, label, random_state = 23, test_size = 0.2)


# In[ ]:


model = linear_model.LinearRegression()
model.fit(f_train, l_train)
r2_score(l_train, model.predict(f_train))


# **Polynomial Features**

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4) 

feature_train = poly.fit_transform(f_train)


# In[ ]:


poly_model = linear_model.LinearRegression()
poly_model.fit(feature_train, l_train)
r2_score(l_train, poly_model.predict(feature_train)) 


# > **Ridge Regression**

# In[ ]:


ridge_model = linear_model.Ridge()
ridge_model.fit(f_train, l_train)
r2_score(l_train, ridge_model.predict(f_train)) 


# **Descision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor()
tree_model.fit(f_train, l_train)
r2_score(l_train, tree_model.predict(f_train))  # Overfit


# In[ ]:


r2_score(l_test, tree_model.predict(f_test)) 

# This model works poorly on new data sets


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
random_model = RandomForestRegressor()
random_model.fit(f_train, l_train)
r2_score(l_train, random_model.predict(f_train))


# In[ ]:


r2_score(l_test, random_model.predict(f_test)) 


# In[ ]:


random_model.get_params()


# In[ ]:


from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [5,10, 15, 20, 25, None],    
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [25, 50, 100, 200]}

grid_search = GridSearchCV(random_model, params, cv = 3)


# In[ ]:


grid_search.fit(f_train, l_train) 


# In[ ]:


grid_search.best_estimator_


# In[ ]:


new_random = RandomForestRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators= 100 )


# In[ ]:


new_random.fit(f_train, l_train)
r2_score(l_train, new_random.predict(f_train)) 


# In[ ]:


r2_score(l_test, new_random.predict(f_test)) 


# In[ ]:


# So, It seems Random forest performs better among all the model we have.
# We have used Only Regression model for this kernel. We will use classification models in the next kernel

