#!/usr/bin/env python
# coding: utf-8

# An Ecommerce company based in New York City sells clothing online, but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. This project uses fake customer data.

# In[ ]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#saving data from csv file into dataframe
customers = pd.read_csv('../input/Ecommerce Customers')


# In[ ]:


customers.head()


# In[ ]:


customers.info()


# In[ ]:


customers.describe()


# In[ ]:


#Performing exploratory analysis
#analyzing yearly amount spent vs time on website
sns.set_style('darkgrid')
sns.jointplot(x=customers['Time on Website'], y=customers['Yearly Amount Spent'], data=customers)


# In[ ]:


#analyzing yearly amount spent vs time on app
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)


# In[ ]:


#analyzing yearly amount spent vs the length of membership
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')


# In[ ]:


#analyzing these types of relationships all across the data set
sns.pairplot(customers)


# Based on this, the Length of Membership appears to be most corelated with the Yearly Amount Spent.

# In[ ]:


#Creating a linear model using seaborn to plot the relationship between Yearly Amount Spent vs Length of Membership
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)


# In[ ]:


#Training and testing data
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


#Training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[ ]:


#printing coefficients
lm.coef_


# In[ ]:


predictions = lm.predict(X_test)
sns.scatterplot(y = predictions, x = y_test)
plt.ylabel('Predicted Y')
plt.xlabel('y test')


# In[ ]:


#Evaluating the performance of the model by calculating the residual sum of squares and the variance score R^2
from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, predictions), ', Mean Squared Error: ',
      metrics.mean_squared_error(y_test, predictions) , ', Root Mean Squared Error: ', 
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


print('R^2: Variance Score is ', metrics.explained_variance_score(y_test, predictions))
#This means the model explains nearly 99% of the variance.


# In[ ]:


#Plotting a histogram of residuals
sns.distplot(y_test-predictions)


# In[ ]:


#Analyzing coefficients
pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])


# **Conclusion: **

# Length of Membership appears to be most critical to the Yearly Amount Spent.
# The app helps make 38.59 dollars and the website only 19 cents per unit time spent on it.
# Depending on the cost associated with improving either, a decision may be made on where it is best to focus in order to improve Yearly Amount Spent of consumers.

# In[ ]:




