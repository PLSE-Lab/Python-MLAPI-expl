#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Creating the data set for X1
# Values in X1 are random numbers between 500-2000

# In[ ]:


np.random.seed(1)
X1 = np.random.randint(500, 2000, 50)
X1


# ### Creating the data set for X2
# Values in X1 are random numbers between 100-500

# In[ ]:


np.random.seed(1)
X2 = np.random.randint(100, 500, 50)
X2


# ### Creating the data set for X3
# Values in X3 is 3 times the values in X1 + a random vector; where The random vector is a random value between 0 and 500 for each data. 

# In[ ]:


X3 = [i * 3 + np.random.randint(500) for i in X1]
X3


# ### Creating the data set for Y
# **Formula:** *Y = (6 xX1) + X2*

# In[ ]:


Y = [6 * X1[i] + X2[i] for i in range(0,50)]
Y


# ### Creation of the Dataframe using X1, X2, X3, and Y.

# In[ ]:


data = {'X1':X1, 'X2':X2, 'X3': X3, 'Y': Y} 
df = pd.DataFrame(data) 
df


# ### Find the Correlation.

# #### The correlation between X1 and Y
# 

# In[ ]:


#using pandas
corr = df['X1'].corr(df['Y'])
corr


# In[ ]:


#Spearmans's 's correlation coefficient
corr, _ = spearmanr(X1, Y)
corr


# In[ ]:


# Pearson's correlation coefficient
corr, _ = pearsonr(X1, Y)
corr


# #### The correlation between X2 and Y

# In[ ]:


#using pandas
corr = df['X2'].corr(df['Y'])
corr


# In[ ]:


#Spearmans's correlation coefficient
corr, _ = spearmanr(X2, Y)
corr


# In[ ]:


#Spearmans's 's correlation coefficient
corr, _ = spearmanr(X2, Y)
corr


# #### The correlation between X3 and Y

# In[ ]:


#using pandas
corr = df['X3'].corr(df['Y'])
corr


# In[ ]:


# Pearson's correlation coefficient
corr, _ = pearsonr(X3, Y)
corr


# In[ ]:


#Spearmans's 's correlation coefficient
corr, _ = spearmanr(X3, Y)
corr


# ### Illustrating the relationship between X1 and Y

# In[ ]:


plt.scatter(X1, Y, alpha=0.5)
plt.title('Scatter plot illustrating the relationship between X1 and Y')
plt.xlabel('X1')
plt.ylabel('Y')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.show()


# ### Illustrating the relationship between X2 and Y

# In[ ]:


plt.scatter(X2, Y, alpha=0.5)
plt.title('Scatter plot illustrating the relationship between X2 and Y')
plt.xlabel('X2')
plt.ylabel('Y')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.show()


# ### Regression Analysis

# In[ ]:


# Separate our data into independent(X) variables.
X_data = df[['X1','X2']]
X_data


# In[ ]:


# Separate our data into dependent(Y) variables.
Y_data = df['Y']
Y_data


# In[ ]:


# 70/30 Train Test Split.
# We will split the data using a 70/30 split. i.e. 70% of the data will be randomly 
# chosen to train the model and 30% will be used to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)


# In[ ]:


# Create an instance of linear regression
reg = linear_model.LinearRegression()


# In[ ]:


# Fitting the X_train onto y_train.
reg.fit(X_train,y_train)


# In[ ]:


print("Regression Coefficients")
pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])


# In[ ]:


# Intercept
reg.intercept_


# In[ ]:


# Make predictions using the testing set
test_predicted = reg.predict(X_test)
test_predicted


# In[ ]:


# Explained variance score: 1 is perfect prediction
# R squared
print('Variance score: %.2f' % r2_score(y_test, test_predicted))


# In[ ]:


reg.score(X_test,y_test)


# In[ ]:


scores = cross_val_score(reg,X_data, Y_data, cv=5)
scores     


# In[ ]:


#Residual Plot
plt.scatter(reg.predict(X_train), reg.predict(X_train)-y_train,c='b',s=40,alpha=0.5)
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,c='g',s=40)
plt.hlines(y=0,xmin=np.min(reg.predict(X_test)),xmax=np.max(reg.predict(X_test)),color='red',linewidth=3)
plt.title('Residual Plot using Training (blue) and test (green) data ')
plt.ylabel('Residuals')


# In[ ]:


model = Ridge()
visualizer = ResidualsPlot(model)

visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)

visualizer.show()


# Predicting the associated outcome

# In[ ]:


x=342
y=21
data = {'x':[x],'y':[y]}
df=pd.DataFrame(data)
reg.predict(df)


# In[ ]:


#MAE
mean_squared_error(y_test, test_predicted)


# In[ ]:


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))
print("Mean Absolute error: %.2f" % mean_absolute_error(y_test, test_predicted))
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, test_predicted)))

