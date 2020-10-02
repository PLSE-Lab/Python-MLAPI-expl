#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/BlackFriday.csv")


# In[ ]:


df.head()


# In[ ]:


# first we check number of unique users and products
print("Number of Unique Users ",df['User_ID'].nunique())
print("Number of Unique Products ",df['Product_ID'].nunique())
print("Number of Transactions ",len(df))


# # From above information we can see that above data is many to many mapping between Users and Products

# ## Purchase is an important variable which we want to predict which is the price of the product that the user is buying
# ## So lets conduct analysis on the products variable 

# In[ ]:


fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(15,15))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4,hspace=0.4)

df['User_ID'].groupby(df['Gender']).nunique().plot(kind='bar',ax=axes[0,0])
df['User_ID'].groupby(df['Age']).nunique().plot(kind='bar',ax=axes[0,1])
df['User_ID'].groupby(df['Occupation']).nunique().plot(kind='bar',ax=axes[1,0])
df['User_ID'].groupby(df['City_Category']).nunique().plot(kind='bar',ax=axes[1,1])
df['User_ID'].groupby(df['Stay_In_Current_City_Years']).nunique().plot(kind='bar',ax=axes[2,0])
df['User_ID'].groupby(df['Marital_Status']).nunique().plot(subplots=True,kind='bar',ax=axes[2,1])
df['User_ID'].groupby(df['Product_Category_1']).nunique().plot(subplots=True,kind='bar',ax=axes[3,0])
df['User_ID'].groupby(df['Product_Category_2']).nunique().plot(subplots=True,kind='bar',ax=axes[3,1])
df['User_ID'].groupby(df['Product_Category_3']).nunique().plot(kind='bar',ax=axes[4,0])


# In[ ]:


fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(15,15))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4,hspace=0.4)

df['Product_ID'].groupby(df['Gender']).nunique().plot(kind='bar',ax=axes[0,0])
df['Product_ID'].groupby(df['Age']).nunique().plot(kind='bar',ax=axes[0,1])
df['Product_ID'].groupby(df['Occupation']).nunique().plot(kind='bar',ax=axes[1,0])
df['Product_ID'].groupby(df['City_Category']).nunique().plot(kind='bar',ax=axes[1,1])
df['Product_ID'].groupby(df['Stay_In_Current_City_Years']).nunique().plot(kind='bar',ax=axes[2,0])
df['Product_ID'].groupby(df['Marital_Status']).nunique().plot(subplots=True,kind='bar',ax=axes[2,1])
df['Product_ID'].groupby(df['Product_Category_1']).nunique().plot(subplots=True,kind='bar',ax=axes[3,0])
df['Product_ID'].groupby(df['Product_Category_2']).nunique().plot(subplots=True,kind='bar',ax=axes[3,1])
df['Product_ID'].groupby(df['Product_Category_3']).nunique().plot(kind='bar',ax=axes[4,0])


# In[ ]:


lb = LabelEncoder()
df['Gender'] = lb.fit_transform(df['Gender'])
df['Age'] = lb.fit_transform(df['Age'])
df['City_Category'] = lb.fit_transform(df['City_Category'])
df['Stay_In_Current_City_Years'] = lb.fit_transform(df['Stay_In_Current_City_Years'])


# In[ ]:


fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(df.drop(['User_ID','Product_ID'],axis=1).corr())


# ## From the above heatmap we can see that the correlation between product category 1 and product category 2 is high compared to product category 1 and product category 3
# 
# ## we can also observe that there is a negative correlation between purchase and product category 1 and product category 2 which might indicate the prices of product category 1 and product category 2 are high and few people purchase them
# 
# ## we can also check that age and marital status have high correlation which makes lot of sense
# 
# ## we can also observe that no variable is highly correlated with the purchase variable which is our target .It shows that purchase variable depends on ensemble of all the variables

# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop(['User_ID','Product_ID','Purchase'],axis=1)
# df.reset_index()
y = df['Purchase']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ## Running linear regression Model on the data

# In[ ]:


lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# ## Even cross validation havent improved our accuracy metrics much

# In[ ]:


scores = cross_val_score(model, X, y, cv=6)
print("Cross-validated scores:", scores)


# ## Running Random Forest Regressor on data

# In[ ]:


regr = RandomForestRegressor(max_depth=5,random_state=42,n_estimators=100)


# In[ ]:


regr.fit(X_train,y_train)


# In[ ]:


regr.score(X_test,y_test)


# In[ ]:


regr.feature_importances_


# In[ ]:




