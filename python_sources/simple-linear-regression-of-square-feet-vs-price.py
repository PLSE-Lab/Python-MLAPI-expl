#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
from subprocess import check_output


# **We will take a look at how the square feet of a house affects it's price**

# In[ ]:


data = pd.read_csv('../input/kc_house_data.csv', usecols = ['sqft_living', 'price'], sep = ',')


# *We'll create two variables for sqft and price*

# In[ ]:


print(data.shape)
sft = data.iloc[:, 1:2].values
price = data.iloc[:, 0:1].values


# *Split the data into training and test groups. Random state = 0 will ensure that we get identical sets every time we split the data*

# In[ ]:


sft_train, sft_test, price_train, price_test = train_test_split(sft, price, test_size = 0.2776, random_state = 0)


# *We'll take a look at the relation between square feet and price for the training and test data*

# In[ ]:


fig, ax = plt.subplots(1)
sns.regplot(x= sft_train, y= price_train, color = 'c', fit_reg = False, scatter = True)
plt.title('Square feet vs Price (Training set)')
plt.xlabel('Square feet')
plt.ylabel('Price')
ax.set_yticklabels([])
plt.show()


# In[ ]:


fig, ax = plt.subplots(1)
sns.regplot(x= sft_test, y= price_test, color = 'b', fit_reg = False, scatter = True)
plt.title('Square feet vs Price (Test set)')
plt.xlabel('Square feet')
plt.ylabel('Price')
ax.set_yticklabels([])
plt.show()


# *The plots show that the relation between square feet of a house and it's selling price is quite linear! So, let's try and fit a simple linear regression model on the training data first and then calculate errors (RSS) on both training and test data*

# In[ ]:


linreg = LinearRegression()
linreg.fit(sft_train, price_train)
print(linreg.coef_, linreg.intercept_)


# *So, what do we have here? 
# According to the linear regression equation f(x) = w0 + w1(x), the coefficients are -
# w0 = -41443.58
# w1 = 280.25.
# The w1 parameter tells us that the price of any house increases by 280.25 units in price, per unit increase in square footage*

# *Now let's predict some prices in the test set and look at the errors associated. Error in regression would generally mean Regression Sum of Squares RSS, where, 
# RSS = SUM[(predictions - RealValue)^2]*

# In[ ]:


predictions = linreg.predict(sft_test)
Errors = predictions - price_test
RSS = np.sum(np.square(Errors))
print(RSS)


# *Plotting predicted values*

# In[ ]:


fig, ax = plt.subplots(1)
sns.regplot(x= sft_test, y= price_test, color = 'b', fit_reg = False, scatter = True)
plt.scatter(x = sft_test, y = linreg.predict(sft_test), color = 'r')
plt.title('Square feet vs Price')
plt.xlabel('Square feet')
plt.ylabel('Price')
ax.set_yticklabels([])
plt.show()


# *One of the assumptions of linear regression is of scedasticity. Scedasticity (scatter) plots residuals vs. predictions. If the residuals w.r.t. the predicted values are somewhat uniform throughout the predictions, then the data is said to be linearly dependent on the variate. Let us examine how our prediction errors match with price.*

# In[ ]:


sns.regplot(x= linreg.predict(sft_test), y= Errors, color= 'c', fit_reg= False, scatter= True)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Checking scedasticity of linear model')
plt.show()


# *The plot above shows that the residuals increase in magnitude with increasing predicted values for the test set. This means that the houses with higher price have way more pull on the residuals than those with lower prices. Also, it would seem that the dependence of a house's price is not very linear on square feet. Let us try to confirm this behavior by looking at the distribution of residuals.*

# In[ ]:


sns.distplot(Errors)
plt.ylabel('Residual magnitude')
plt.title('Residual distribution')


# *The above plot shows that the residuals are normally distributed. Even though the residuals are not uniformly distributed w.r.t. predictions, the distribution of the residuals once again hints at the linearity of the dependence of price on square feet*

# *After this, we shall take a look at the way training and test errors behave with increasing model complexity. I invite critical comments and suggestions from you guys, please do feel free to correct me wherever you feel like. I'll be more than happy to learn and share :)*
