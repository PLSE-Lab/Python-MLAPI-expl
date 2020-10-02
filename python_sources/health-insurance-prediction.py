#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv("../input/insurance.csv")


# In[3]:


print (df.head())
print (df.info())


# In[4]:


df.shape


# In[5]:


sns.kdeplot(df[df['sex']=='female']['charges'], shade=True, label = 'Female charge')
sns.kdeplot(df[df['sex']=='male']['charges'], shade=True, label = 'Male charge')


# In[6]:


sns.swarmplot(x='sex', y='charges', data=df)


# In[7]:


#The impact of smoke on charges

df.groupby("smoker")['charges'].agg('mean').plot.bar()


# In[8]:


sns.scatterplot(x='bmi', y='charges',hue='smoker',data=df)


# In[9]:


sns.regplot(x='bmi',y='charges',data=df)


# In[10]:


sns.lmplot(x='bmi',y='charges',hue='sex',data=df)


# In[11]:


sns.scatterplot(x='age', y='charges', hue='sex',data=df)


# In[12]:


sns.lineplot(x='children', y='charges',  estimator=np.median, data=df);


# In[13]:


#sns.lineplot(x='children', y='charges', data=df);
#sns.scatterplot(x='children', y='charges', data=df)
df.groupby('children')['charges'].agg('median')


# # Missing values
# 
# Good news: 0 missing data in this dataset

# In[14]:


df.info()


# In[15]:


df_dummies = pd.get_dummies(df)
df_dummies.head()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X = df_dummies.drop('charges', axis= 1)
y = df_dummies.charges

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[20]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, lm.predict(X_test))


# # Baseline Model
# 
# To compare the models, we need to have a baseline model.
# 
# In our baseline model, the predicted values (^y) are the average of response values (y).

# In[21]:


avg_charges = pd.Series([y_test.mean()]* y_test.shape[0])
avg_charges
mean_absolute_error(y_test, avg_charges)


# # Ridge, Lasso and Elastic Net Regression
# 
# We are going to implement Ridge and Lasso regressions to test the improvement of these models. Especially we wish to see a huge improvement compared to our baseline model (Mean_absolute_error : 9596.63)

# In[22]:


from sklearn.linear_model import Lasso,Ridge, ElasticNet


# In[52]:


ridge = Ridge()
ridge.fit(X_train, y_train)
mean_absolute_error(y_test, ridge.predict(X_test))


# In[24]:


lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
mean_absolute_error(y_test, lasso.predict(X_test))


# In[25]:


elasticnet = ElasticNet()
elasticnet.fit(X_train,y_train)
mean_absolute_error(y_test, elasticnet.predict(X_test))


# # Hyperparameter Tuning
# 
# The improvements can be observed from the Ridge and Lasso regressions. Next, we are going to take this to the next level by tuning the hyperparameter. **GridSearchCV** is the best function in sklearn library to test these tweaks.

# In[26]:


from sklearn.model_selection import GridSearchCV


# In[44]:


params = {'alpha': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}

params_elastic ={'alpha': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                'l1_ratio': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}


# In[49]:


# Ridge
ridge_cv = GridSearchCV(ridge, params, scoring = 'neg_mean_absolute_error')
ridge_cv.fit(X_train,y_train)

# Lasso
lasso_cv = GridSearchCV(lasso, params, scoring = 'neg_mean_absolute_error')
lasso_cv.fit(X_train,y_train)

# Elastic Net
elasticnet_cv = GridSearchCV(elasticnet, params_elastic, scoring = 'neg_mean_absolute_error')
elasticnet_cv.fit(X_train,y_train)


# In[51]:


print (ridge_cv.best_params_)
print (lasso_cv.best_params_)
print (elasticnet_cv.best_params_)
print (mean_absolute_error(y_test, ridge_cv.predict(X_test)))
print (mean_absolute_error(y_test, lasso_cv.predict(X_test)))
print (mean_absolute_error(y_test, elasticnet_cv.predict(X_test)))


# ## Conclusion
# 
# **Lasso regression** is a superior model as it gives the lowest mean absolute error.

# # Future Consideration
# 
# 1. Feature Selection
# 2. Feature Extraction
# 3. Cross validation
# 4. Other Models: DecisionTreeRegressor, RandomForestRegressor and many more ensemble models
