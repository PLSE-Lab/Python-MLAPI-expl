#!/usr/bin/env python
# coding: utf-8

# # Import the necessary libraries
# 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LassoLars, Ridge, LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# #### Load the data into a dataframe

# In[ ]:


df = pd.read_csv('../input/diamonds.csv',delimiter=',')
df.head()


# In[ ]:


df.info()


# There are no missing values in the dataset.

# Drop the first column and the *x*, *y*,*z* columns as the total depth and the table width will be used for further analysis.

# In[ ]:


df.drop(['Unnamed: 0','x','y','z'],inplace=True, axis=1)
df.head()


# # Data Visualization

# Various count plots.

# In[ ]:


f, ax = plt.subplots(3, figsize=(12,18))
sns.countplot('cut', data=df, ax=ax[0])
sns.countplot('color', data=df, ax=ax[1])
sns.countplot('clarity', data=df, ax=ax[2])
ax[0].set_title('Diamond cut')
ax[1].set_title('Colour of the diamond')
ax[2].set_title('Clarity of the diamond')


# From the above plots, the following observations could be made:
# * Quite a lot of diamonds have an *Ideal* cut following by *Premium* and *Very Good.*
# * The dataset contains good distribution of various colours, with *I* and *J* having the least number.
# * Only a small fraction of the diamonds have the best quality while even smaller fracition of the dianonds have poor quality.

# #### Describe the characteristics of the columns containing numeric values.

# In[ ]:


df.describe()


# In[ ]:


f, ax = plt.subplots(4, figsize=(12,24))
sns.distplot(df.carat,color='c',ax=ax[0])
sns.distplot(df.depth,color='c',ax=ax[1])
sns.distplot(df.table,color='c',ax=ax[2])
sns.distplot(df.price,color='c',ax=ax[3])
ax[0].set_title('Diamond carat distribution')
ax[1].set_title('Total depth distribution')
ax[2].set_title('Table width distribution')
ax[3].set_title('Price distribution')


# * Most of the diamonds in the dataset are below 2 carats.
# * The depth peaks at around 62.
# * Table width peaks around 58.
# * Most of the diamonds in the dataset sold are less than $5000.

# #### More plots to understand the variation of price

# In[ ]:


f, ax = plt.subplots(3,figsize=(12,16))
sns.violinplot(x='clarity',y='price',data=df,ax=ax[2])
sns.violinplot(x='color',y='price',data=df,ax=ax[1])
sns.violinplot(x='cut',y='price',data=df,ax=ax[0])
ax[0].set_title('Cut vs Price')
ax[1].set_title('Color vs Price')
ax[2].set_title('Clarity vs Price')


# In[ ]:


sns.jointplot(x='carat',y='price',data=df,color='c')


# # Regression

# Convert the non-numeric data into numbers.

# In[ ]:


le = LabelEncoder()
df.cut = le.fit_transform(df.cut)
df.color = le.fit_transform(df.color)
df.clarity = le.fit_transform(df.clarity)
df.info()


# In[ ]:


x = df.drop('price',axis=1)
y = df.price


# In[ ]:


x_train,x_, y_train,y_ = train_test_split(x,y,test_size=0.15,random_state=25)
x_dev,x_test,y_dev,y_test = train_test_split(x_,y_,test_size=0.5,random_state=25)


# In[ ]:


sc = StandardScaler()
sc.fit(x_train)
sc.transform(x_train)
sc.transform(x_dev)


# #### Linear Model

# In[ ]:


clf = LinearRegression()
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# In[ ]:


clf = ElasticNet(alpha=1)
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# In[ ]:


clf = Ridge(alpha=10)
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# In[ ]:


clf = LassoLars(alpha=1)
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# In[ ]:


clf = Lasso(alpha=40)
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# ### Polynomial model
# 
# LassoLars and ElasticNet were not chosen for further analysis due to poor performance.

# In[ ]:


poly = PolynomialFeatures(2)
x_train = poly.fit_transform(x_train)
x_dev = poly.fit_transform(x_dev)


# In[ ]:


clf = LinearRegression()
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# In[ ]:


clf = Ridge(alpha=10)
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# In[ ]:


clf = Lasso(alpha=30,max_iter=5000)
clf.fit(x_train,y_train)
print('Training score: {:0.3f}'.format(r2_score(y_train,clf.predict(x_train))))
print('Training MSE: {:0.3f}'.format(mean_squared_error(y_train,clf.predict(x_train))))
print('Dev set score: {:0.3f}'.format(r2_score(y_dev,clf.predict(x_dev))))
print('Dev set MSE: {:0.3f}'.format(mean_squared_error(y_dev,clf.predict(x_dev))))
print('Coefficients: {}\n Intercept: {}'.format(clf.coef_,clf.intercept_))


# All the three models showed comparable performance but Lasso was chosen as it had a simpler model due to sparsity.

# In[ ]:


sc.transform(x_test)
x_test = poly.transform(x_test)
print('Test set score: {:0.3f}'.format(r2_score(y_test,clf.predict(x_test))))
print('Test set MSE: {:0.3f}'.format(mean_squared_error(y_test,clf.predict(x_test))))


# In[ ]:




