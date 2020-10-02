#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


D1 = pd.read_csv("../input/heart.csv")
D1.head()


# In[34]:


D1.dtypes


# In[35]:


D1.info()


# In[36]:


sns.pairplot(D1)


# In[37]:


# Change Datatype Of Columns:- 
D1.sex = D1.sex .astype('object')
D1.fbs = D1.fbs .astype('object')
D1.restecg = D1.restecg .astype('object')
D1.exang = D1.exang .astype('object')
D1.slope = D1.slope .astype('object')
D1.ca = D1.ca .astype('object')
D1.thalthal = D1.thal .astype('object')
D1.target = D1.target .astype('object')
D1.info()


# In[38]:


# divide data in Numeric and Cat variable
cat_var = [key for key in dict(D1.dtypes)
             if dict(D1.dtypes)[key] in ['object'] ] # Categorical Varible

numeric_var = [key for key in dict(D1.dtypes)
                   if dict(D1.dtypes)[key]
                       in ['float64','float32','int32','int64']] # Numeric Variable


# In[39]:


# check any Extreme value is present in numeric variable
D1.describe()


# In[40]:


D1.boxplot(column= numeric_var)


# In[41]:


median = D1.loc[D1['trestbps']<140, 'trestbps'].median()
D1.loc[D1.trestbps > 140, 'trestbps'] = np.nan
D1.fillna(median,inplace=True)

median = D1.loc[D1['chol']<246.264026, 'chol'].median()
D1.loc[D1.chol > 246.264026, 'chol'] = np.nan
D1.fillna(median,inplace=True)


# In[42]:


import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi


# In[43]:


# Select imprortant varaible in Catagorical
#ANOVA F Test COVERAGE
model = smf.ols(formula='age ~ target', data=D1)
results = model.fit()
print (results.summary())
# Here The F-statistic is  high
# p-value is to low.


# In[44]:


model = smf.ols(formula='cp ~ target', data=D1)
results = model.fit()
print (results.summary())


# In[45]:


model = smf.ols(formula='trestbps ~ target', data=D1)
results = model.fit()
print (results.summary())


# In[46]:


model = smf.ols(formula='chol ~ target', data=D1)
results = model.fit()
print (results.summary())


# In[48]:


model = smf.ols(formula='thalach ~ target', data=D1)
results = model.fit()
print (results.summary())


# In[49]:


model = smf.ols(formula='oldpeak ~ target', data=D1)
results = model.fit()
print (results.summary())


# In[52]:


model = smf.ols(formula='thal ~ target', data=D1)
results = model.fit()
print (results.summary())


# In[53]:


# Start Some Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
D1.head(n = 3)


# In[64]:


X = D1.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]  #independent columns
y = D1.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
test = SelectKBest(score_func=chi2, k=2)
y = y.astype('int')


# In[65]:


fit = test.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[66]:


#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
features = fit.transform(X)
print(features[0:4,:4])


# In[67]:


# 1. Linear Reggression
from sklearn.model_selection import train_test_split
X = D1.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]  #independent columns
y = D1.iloc[:,-1] #dependent columns.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[69]:


# fit a model
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)


# In[70]:


predict_train = lm.predict(X_train)
predict_test  = lm.predict(X_test)


# In[71]:


# R- Square on Train
lm.score(X_train, y_train)


# In[72]:


# R- Square on Test
lm.score(X_test,y_test)


# In[73]:


# Mean Square Error
mse = np.mean((predict_train - y_train)**2)
mse


# In[74]:


mse = np.mean((predict_test - y_test)**2)
mse


# In[ ]:




