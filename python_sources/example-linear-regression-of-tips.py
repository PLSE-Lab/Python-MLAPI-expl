#!/usr/bin/env python
# coding: utf-8

# ## Example(Linear Regression of Tips)

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression


# In[ ]:


#read data
df = pd.read_csv("../input/tips-datasets/tips.csv")
print(df.shape)
df.head(3)


# ## What is the maximum percent of the total bill tipped? Describe the details of this particular tip.

# In[ ]:


#convert tips to precent tips
df['precent_tip'] = 100*df.tip/df.total_bill
df.head(3)


# In[ ]:


df.precent_tip.max()


# In[ ]:


ix=df.precent_tip.idxmax()
df.iloc[ix,:]


# ## Use Linear regression to predict the precent tip paid. What is the training R2?

# In[ ]:


df.head(3)


# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


features = df.drop('precent_tip',axis=1)
targets = df.precent_tip


# In[ ]:


#standarize features
features = (features - features.mean())/features.std()
features.describe().T[['mean','std']]


# In[ ]:


#fit linear model
lr = LinearRegression()
lr.fit(features,targets)
R2 = lr.score(features,targets)
print('R-sqr',R2.round(2))


# In[ ]:


coef = lr.coef_
coef = pd.Series(coef,index=features.columns)
coef.sort_values(ascending=False)


# ## (d) Drop the features tip and percent_tip and use total_bill as the target and repear part (c).

# In[ ]:


features = df.drop(['tip','precent_tip','total_bill'],axis=1)
target = df.total_bill


# In[ ]:


features = pd.get_dummies(features)


# In[ ]:


# standardize
features = (features -features.mean())/features.std()
features.describe().T[['mean','std']]


# In[ ]:


lr = LinearRegression()
lr.fit(features,target)


# In[ ]:


coef = lr.coef_
R2 = lr.score(features,target)
print('R-squared',R2.round(2))
coef = pd.Series(coef,index=features.columns)
coef.sort_values(ascending=False)


# In[ ]:




