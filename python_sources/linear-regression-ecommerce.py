#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


customers = pd.read_csv('../input/linear-regression/Ecommerce Customers.csv')


# In[ ]:


customers.head()


# In[ ]:


customers.describe()


# In[ ]:


customers.info()


# ## Exploratory Data Analysis
# 
#     

# In[ ]:


sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)


# ** Do the same but with the Time on App column instead. **

# In[ ]:


sns.jointplot(x='Time on App', y ='Yearly Amount Spent', data = customers)


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[ ]:


sns.jointplot(x='Time on App', y='Length of Membership', data = customers, kind = 'hex')


# In[ ]:


sns.pairplot(customers)


# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[ ]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)


# ## Training and Testing Data
# 
# 

# In[ ]:


customers.columns


# In[ ]:


X= customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y= customers ['Yearly Amount Spent']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
# 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[ ]:


lm.fit(X_train, y_train)


# **Print out the coefficients of the model**

# In[ ]:


print (lm.coef_)


# ## Predicting Test Data
# 

# In[ ]:


predictions = lm.predict(X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[ ]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ## Evaluating the Model
# 
#     

# In[ ]:


from sklearn import metrics
print (metrics.mean_absolute_error(y_test, predictions))
print (metrics.mean_squared_error(y_test, predictions))
print (np.sqrt (metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 

# In[ ]:


sns.distplot(y_test-predictions)


# In[ ]:


pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficient'])


# In[ ]:


# 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.


# In[ ]:




