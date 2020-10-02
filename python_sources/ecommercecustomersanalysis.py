#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Linear Regression Project
# ### An Ecommerce company based in New York City that sells clothing online also has in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# ### My goal is to advise the company on wheter to focus their efforts on their mobile app experience or their website. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ## Get the Data
# ### We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# - Avg. Session Length: Average session of in-store style advice sessions.
# - Time on App: Average time spent on App in minutes
# - Time on Website: Average time spent on Website in minutes
# - Length of Membership: How many years the customer has been a member.
# - Read in the Ecommerce Customers csv file as a DataFrame called customers.

# In[ ]:


ECommCust = pd.read_csv('../input/Ecommerce Customers')


# ### Check the head of customers, and check out its info() and describe() methods.

# In[ ]:


ECommCust.head()


# In[ ]:


ECommCust.info()


# In[ ]:


ECommCust.describe()


# # Exploratory Data Analysis
# ### Let's explore the data!
# 
# ### For the rest of the exercise we'll only be using the numerical data of the csv file.
# 
# ### Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?

# In[ ]:


sns.set(style="whitegrid", palette='GnBu_d')


# In[ ]:


j = sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=ECommCust, kind='reg', scatter_kws={"s": 10})
j.annotate(stats.pearsonr)
plt.show()


# ### Do the same but with the Time on App column instead.

# In[ ]:


j = sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=ECommCust, kind='reg', scatter_kws={"s": 10})
j.annotate(stats.pearsonr)
plt.show()


# ### Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.

# In[ ]:


j = sns.jointplot(x='Time on App', y='Length of Membership', data=ECommCust, kind='hex')
j.annotate(stats.pearsonr)
plt.show()


# ### Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.

# In[ ]:


sns.pairplot(ECommCust)


# # Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
# 
# ## Length of Membership has the highest correlation with Time on App coming in second.

# ### Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.

# In[ ]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=ECommCust, scatter_kws={"s": 10})


# # Training and Testing Data
# ## Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.

# In[ ]:


X = ECommCust[['Avg. Session Length', 
               'Time on App',
               'Time on Website', 
               'Length of Membership']]
y = ECommCust['Yearly Amount Spent']


# ### Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Training the Model
# ## Now its time to train our model on our training data!
# 
# ### Create an instance of a LinearRegression() model named lm.

# In[ ]:


lm = LinearRegression()


# ### Train/fit lm on the training data.

# In[ ]:


lm.fit(X_train, y_train)


# ### Print out the coefficients of the model

# In[ ]:


print('Coefficients: \n', lm.coef_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df


# # Predicting Test Data
# ## Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ### Use lm.predict() to predict off the X_test set of the data.

# In[ ]:


predictions = lm.predict(X_test)


# ### Create a scatterplot of the real test values versus the predicted values.

# In[ ]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# # Evaluating the Model
# ## Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ### Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # Residuals
# ## We got a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.
# 
# ### Plot a histogram of the residuals and make sure it looks normally distributed.

# In[ ]:


sns.distplot((y_test-predictions), bins=50)


# ### The histogram of the residuals is normally distributed, so a linear regression model is a good fit.
# 
# # Conclusion
# ## We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.

# In[ ]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df


# ### Interpreting the coefficients (notes placed in descending order of coeeficients):
# 
# - Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.
# - Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# - Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# - Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# 
# # Recommendation for company:
# 
# ## Length of membership is by far the strongest indicator of dollars spent. So perhaps digging into why members stay vs churn would yield more insight. The app seems to yield more dollars spent than the website, but is that because of a difference between the platforms or because of a difference between the customers who use the respective platforms? More research should be done regarding why customers churn and to examine the differences and similarities between the app and website customers before a conclusion is made.

# In[ ]:




