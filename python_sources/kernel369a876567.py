#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing datasets
from sklearn.datasets import load_boston
boston = load_boston()


# In[ ]:


#Initialising the dataframe
data = pd.DataFrame(boston.data)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


#adding features 
data.columns = boston.feature_names
data.head()


# In[ ]:


data["price"] = boston.target


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


#Forming the correlation matrix
corr = data.corr()
corr.shape


# In[ ]:


#Plotting the heatmaps
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt ='.1f', annot=True)


# In[ ]:


X = data.iloc[:,0:13]
Y = data.iloc[:,13]


# In[ ]:


#Splitting dataset into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size = 0.2 , random_state=4)


# **Linear Regression**

# In[ ]:


#Fitting the dataset to regressor
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()


# In[ ]:


lin_reg.fit(X_train, Y_train)


# In[ ]:


#Predicting the test result
Y_pred = lin_reg.predict(X_train)


# In[ ]:


#Visualising the trainning set result
plt.scatter(Y_train, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[ ]:


# Checking Normality of errors
sns.distplot(Y_train-Y_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


# Predicting Test data with the model
Y_test_pred = lin_reg.predict(X_test)


# In[ ]:


# Model Evaluation
acc_linreg = metrics.r2_score(Y_test, Y_test_pred)
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_test, Y_test_pred))*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_test, Y_test_pred))
print('MSE:',metrics.mean_squared_error(Y_test, Y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train , Y_train)


# In[ ]:


#Predicting on train set
Y_pred= regressor.predict(X_train)


# In[ ]:


# Model Evaluation
print('R^2:',metrics.r2_score(Y_train, Y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_train, Y_pred))*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_train, Y_pred))
print('MSE:',metrics.mean_squared_error(Y_train, Y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_train, Y_pred)))


# In[ ]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(Y_train, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[ ]:


#Predicting for test set
Y_pred_test = regressor.predict(X_test)


# In[ ]:


# Model Evaluation
acc_rf = metrics.r2_score(Y_test, Y_test_pred)
print('R^2:', acc_rf)
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_test, Y_test_pred))*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_test, Y_test_pred))
print('MSE:',metrics.mean_squared_error(Y_test, Y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))


# **SVM**

# In[ ]:


#Fitting SVR to the dataset
from sklearn.svm import SVR
s_reg = SVR(kernel = 'rbf')
s_reg.fit(X_train , Y_train)


# In[ ]:


#Predicting on train set
Y_pred= s_reg.predict(X_train)


# In[ ]:


# Model Evaluation
print('R^2:',metrics.r2_score(Y_train, Y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_train, Y_pred))*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_train, Y_pred))
print('MSE:',metrics.mean_squared_error(Y_train, Y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_train, Y_pred)))


# In[ ]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(Y_train, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[ ]:


#Predicting for test set
Y_pred_test = s_reg.predict(X_test)


# In[ ]:


# Model Evaluation
acc_svm = metrics.r2_score(Y_test, Y_test_pred)
print('R^2:', acc_svm)
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_test, Y_test_pred))*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_test, Y_test_pred))
print('MSE:',metrics.mean_squared_error(Y_test, Y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))


# In[ ]:


#Visualizing the differences between actual prices and predicted values
plt.scatter(Y_train, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[ ]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Support Vector Machines'],
    'R-squared Score': [acc_linreg*100, acc_rf*100, acc_svm*100]})
models.sort_values(by='R-squared Score', ascending=False)


# In[ ]:




