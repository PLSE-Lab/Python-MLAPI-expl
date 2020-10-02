#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Project
# #### Benjamin Jones

# This is a portfolio project for the Udemy online course https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/.
# 
# We will be using Linear Regression to advise a (fake) Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. 
# 
# We'll first explore this data, before using Linear Regression to study the correlations between the different factors and how they affect the how much the customer spends.

# ### Imports:

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Exploration

# Let's get the data and explore it:

# In[7]:


df = pd.read_csv('../input/Ecommerce Customers')


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# Let's use Seaborn to plot some graphs to compare the relationships between the columns:

# In[ ]:


sns.pairplot(df)


# We can also construct a heatmap of these correlations:

# In[ ]:


sns.heatmap(df.corr(),cmap = 'Blues', annot=True)


# We can see that there is a strong correlation between Length of Membership and Yearly Amount Spent

# ## Splitting the Data

# Let's split the data into training and testing data. The feature we are interested in predicting is the Yearly Amount Spent.

# In[24]:


X = df[['Avg. Session Length', 'Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model

# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lm = LinearRegression()


# In[29]:


lm.fit(X_train,y_train)


# In[30]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# In[31]:


lm.coef_


# ## Predicting the Model

# Let's see how well our model performs on the test data (for which we already have the labels)

# In[32]:


pred = lm.predict(X_test)


# In[33]:


plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# We can see that our model is pretty good!

# ## Evaluating the Model

# Let's calculate some errors:

# In[34]:


from sklearn import metrics


# In[36]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# ## Conclusions

# In[37]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# These numbers mean that holding all other features fixed, a 1 unit increase in Avg. Session Length will lead to an increase in $25.981550 in Yearly Amount Spent, and similarly for the other features

# So as Time on App is a much more significant factor than Time on Website, the company has a choice: they could either focus all the attention into the App as that is what is bringing the most money in, or they could focus on the Website as it is performing so poorly!

# ### Thanks for reading!

# Benjamin Jones

# In[ ]:




