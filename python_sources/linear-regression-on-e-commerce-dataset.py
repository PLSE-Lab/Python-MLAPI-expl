#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First we are going to start by importing all the libraries that we would need to analyse this data. 
# I have downloaded pandas, numpy, matplotlib and seaborn. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


customer = pd.read_csv("../input/Ecommerce Customers.csv")


# Now we are going to try to get fimiliar with the dataset. For this we take a look at the first 5 rows of the dataset.  

# In[ ]:


customer.head()


# Now we are going to check the object type, number of entries, columns etc.

# In[ ]:


customer.info()


# Now we start with the initial data exploration. We firrst build a heatmap with all the parameters. 

# In[ ]:


sns.heatmap(customer.corr(), linewidth=0.5, annot=True)


# Now we try to see in there is any correlation between the time spent on the website to the yearly amount spent.

# In[ ]:


sns.jointplot(data = customer, x = "Time on Website", y = "Yearly Amount Spent")


# From the above plot it is difficult to see if there is any correlation between the two variables taken. Now we would try to see if there is any correlation between Time on App with yearly amount spent.

# In[ ]:


sns.jointplot(data = customer, x = "Time on App", y = "Yearly Amount Spent")


# The above plot seems correlated. Increasing time on App seems to be related to higher Yearly Amount spent. 
# Now we will develop a 2D hex bin plot comparing the Time on App and Length of membership. 

# In[ ]:


sns.jointplot(data = customer, x = "Time on App", y = "Length of Membership", kind = "hex")


# Now we develop a pairplot to see all the parameters and try to visually find any parameter that would be correlated. 

# In[ ]:


sns.pairplot(customer)


# From the above plot we can see that Yearly Amount Spent is correlated to Length of Membership. Now we would try to plot this relation in a little more detail. To do this we build a linear model plot. 

# In[ ]:


sns.lmplot(data = customer, x= "Length of Membership", y ="Yearly Amount Spent")


# The above plot shows that the longer a customer remains a member, the higher the yearly amount spent is going to be. 
# Now we start the model building provcess. We devide the data into 2 parts (x and y). X would be the parameters that we would use to predict the y value and the y value in our case is Yearly Amount Spent. 

# In[ ]:


x=customer[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=customer['Yearly Amount Spent']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# Now we will train the model on the training dataset. 

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(x_train, y_train)


# In[ ]:


lm.coef_


# Now we will preddict using the test data. 

# In[ ]:


result = lm.predict(x_test)


# Now to better visualise the model's performance we would build a scatterplot of the result vs the test values.

# In[ ]:


plt.scatter(y_test, result)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")


# The above plot shows that the model is performing well although there are some divations from the line. Now we will evaluated the model using statistics. We would be calculating Mean Absolute Error, Mean Squared  Error and Root Mean Squared Error. 

# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE ', metrics.mean_absolute_error(y_test,result))
print('MSE ', metrics.mean_squared_error(y_test,result))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,result)))


# Now we will check how well the model explains varience. 

# In[ ]:


metrics.explained_variance_score(y_test,result)


# The above figure confirms that the model is well fit to the test dataset.Now we will check the residual. IF the histogram comes out to be normal then it means that everything is ok with the data.

# In[ ]:


plt.hist((y_test-result))


# The plot looks normally distributed. This means that our modelling was a success!!
