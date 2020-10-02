#!/usr/bin/env python
# coding: utf-8

# 
# ___
# # Linear Regression - Project Exercise
# 

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[ ]:


df=pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')


# **Check the head of customers, and check out its info() and describe() methods.**

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[ ]:


df["Time on Website"].corr(df['Yearly Amount Spent'])


# In[ ]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(df["Time on Website"],df['Yearly Amount Spent'])


# ** Do the same but with the Time on App column instead. **

# In[ ]:


sns.jointplot(df["Time on App"],df['Yearly Amount Spent'])


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[ ]:


sns.jointplot(df['Time on App'],df['Length of Membership'],kind='hex')


# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

# In[ ]:


sns.pairplot(df)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# In[ ]:


df["Length of Membership"].corr(df['Yearly Amount Spent'])


# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[ ]:


sns.lmplot("Length of Membership",'Yearly Amount Spent',data=df,fit_reg=True)


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[ ]:


X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']


# In[ ]:





# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[ ]:


from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

# In[ ]:


lm=LinearRegression()


# ** Train/fit lm on the training data.**

# In[ ]:


lm.fit(x_test,y_test)


# **Print out the coefficients of the model**

# In[ ]:


lm.coef_


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[ ]:


pred=lm.predict(x_test)
pred


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[ ]:


plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[ ]:


from sklearn import metrics
print('R2:',metrics.r2_score(y_test,pred))
print('MAE:',metrics.mean_absolute_error(y_test,pred))
print('MSE:',metrics.mean_squared_error(y_test,pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[ ]:


sns.distplot((y_test-pred),bins=50)


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# ** Recreate the dataframe below. **

# In[ ]:


cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Yearly Amount Spent'])
cdf


# ** How can you interpret these coefficients? **

# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# 
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# 
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# 
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

# In[ ]:


#### ** Do you think the company should focus more on their mobile app or on their website? **


# there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!

# In[ ]:




