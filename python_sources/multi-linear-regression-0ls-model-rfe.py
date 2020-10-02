#!/usr/bin/env python
# coding: utf-8

# ## Multiple Linear Regression
# 
# I will explain regression in the simplest possible way. Lets assume that you are using a cab to go from place A to B. You notice that there's a fixed fare, let's say, $3 or 50 rupees (for example).  You owe this fixed amount to the driver the moment you step into the cab, irrespective of the distance you travel with him.
# As the cab starts moving, for every 100 meters or 1 kilometer, the fare increases by a certain amount. If you're not moving, and you're stuck in traffic, for every additional minute, you have to pay more. As the distance increases, your fare increases. As the waiting period increases, your fare increases. Therefore, your fare is calculated based on the distance and wait  period. There is a relation with respect to distance, wait time and fare. This is what regression is.<br/> 
# Let's  form an equation based on the above scenario, <font size='4'>fare = fixed amount + (distance)*(X) + (wait time)*(Y)</font>.<br/>
# From the above equation, if you know the  fare, distance and wait time, using regression you can find the constant and the relationship among the variables. So, regression is used to find a formula that fits the relationship between the variables(like fare, distance..etc) and we can use that formula to predict fare value.

# Lets now solve a basic Linear Regression problem.<br/>
# **Problem Statement** <br/>
# Let's say there is a Company want to increase it's sales. They are investing lot of money on marketing in Newspaper, TV and Radio.
# They want to know which marketing(Newspaper or TV or Radio) is impacting more on Sales. 

# ### Step_1 : Importing and Understanding Data

# In[ ]:


import pandas as pd


# In[ ]:


# Importing advertising.csv
advertising_multi = pd.read_csv('../input/advertising-mul/advertising.csv')


# In[ ]:


# Looking at the first five rows
advertising_multi.head()


# In[ ]:


# Looking at the last five rows
advertising_multi.tail()


# In[ ]:


# What type of values are stored in the columns?
advertising_multi.info()


# In[ ]:


# Let's look at some statistical information about our dataframe.
advertising_multi.describe()


# ### Step_2: Visualising Data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Let's plot a pair plot of all variables in our dataframe
sns.pairplot(advertising_multi)


# In[ ]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(advertising_multi, x_vars=['TV','Radio','Newspaper'], y_vars='Sales',size=7, aspect=0.7, kind='scatter')


# ### Step_3: Splitting the Data for Training and Testing

# In[ ]:


# Putting feature variable to X
X = advertising_multi[['TV','Radio','Newspaper']]

# Putting response variable to y
y = advertising_multi['Sales']


# In[ ]:


#random_state is the seed used by the random number generator. It can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)


# ### Step_4 : Performing Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()


# In[ ]:


# fit the model to the training data
lm.fit(X_train,y_train)


# ### Step_5 : Model Evaluation

# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# From the above result we may infern that if TV price increses by 1 unit it will affect sales by 0.045 units.

# ### Step_6 : Predictions

# In[ ]:


# Making predictions using the model
y_pred = lm.predict(X_test)


# ### Step_7: Calculating Error Terms

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# ### Checking for P-value Using STATSMODELS

# In[ ]:


import statsmodels.api as sm
X_train_sm = X_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)
# create a fitted model in one line
lm_1 = sm.OLS(y_train,X_train_sm).fit()

# print the coefficients
lm_1.params


# In[ ]:


print(lm_1.summary())


# From the above we can see that Newspaper(p value is very high) is insignificant.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize = (5,5))
sns.heatmap(advertising_multi.corr(),annot = True)


# ### Step_8 : Implementing the results and running the model again

# From the data above, you can conclude that Newspaper is insignificant.

# In[ ]:


# Removing Newspaper from our dataset
X_train_new = X_train[['TV','Radio']]
X_test_new = X_test[['TV','Radio']]


# In[ ]:


# Model building
lm.fit(X_train_new,y_train)


# In[ ]:


# Making predictions
y_pred_new = lm.predict(X_test_new)


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label


# In[ ]:


# Error terms
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred_new)
r_squared = r2_score(y_test, y_pred_new)


# In[ ]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


X_train_final = X_train_new
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_final = sm.add_constant(X_train_final)
# create a fitted model in one line
lm_final = sm.OLS(y_train,X_train_final).fit()

print(lm_final.summary())


# ### Model Refinement Using RFE

# The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the less important features are pruned from the the current set of features. This procedure is recursively repeated on the pruned dataset until the desired number of features to select is reached.

# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


rfe = RFE(lm, 2)


# In[ ]:


rfe = rfe.fit(X_train, y_train)


# In[ ]:


print(rfe.support_)
print(rfe.ranking_)


# ### Simple Linear Regression: Newspaper(X) and Sales(y)

# In[ ]:


import pandas as pd
import numpy as np
# Importing dataset
advertising_multi = pd.read_csv('../input/advertising-mul/advertising.csv')

x_news = advertising_multi['Newspaper']

y_news = advertising_multi['Sales']

# Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_news, y_news, 
                                                    train_size=0.7 , 
                                                    random_state=110)

# Required only in the case of simple linear regression
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]

# Linear regression from sklearn
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Fitting the model
lm.fit(X_train,y_train)

# Making predictions
y_pred = lm.predict(X_test)

# Importing mean square error and r square from sklearn library.
from sklearn.metrics import mean_squared_error, r2_score

# Computing mean square error and R square value
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Printing mean square error and R square value
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# **Observe that R_square_value is very low, which indicates that it is insignificant to the model**
