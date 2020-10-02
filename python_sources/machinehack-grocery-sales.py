#!/usr/bin/env python
# coding: utf-8

# #### Machine Hack - Sales Forcasting

# ###### About Data 
# 
# Sales forecasting has always been one of the most predominant applications of machine learning. Big companies like Walmart have been employing this technique to achieve steady and enormous growth over decades now. In this challenge, you as a data scientist must use machine learning to help a small grocery store in predicting its future sales and making better business decisions.
# 
# Given the daily sales of a grocery shop recorded over a period of almost 2 years, your objective as a data scientist is to build a machine learning model that can forecast the sales for the upcoming 3 months.

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kurtosistest

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter("ignore")


# In[ ]:


#reading the data
train = pd.read_csv("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Train.csv")
test = pd.read_csv("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Test.csv")


# In[ ]:


#checking on the basic
print(train.head(2))
print("*"*50)
print(train.isnull().sum())
print("*"*50)
print("The shape of Train Dataset is:",train.shape)


# In[ ]:


train.describe()


# ### Pre-processing and Cleansing the Data

# In[ ]:


#checking normality of data
k2, p = stats.normaltest(train['GrocerySales'])
alpha = 1e-3
print("p = {:g}".format(p))
p = 3.27207e-11
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("Data Not Normalize")
else:
    print("Data Noramlize")


# In[ ]:


#correlation comparison 
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


#scatter matrix
scatter_matrix(train)
plt.show()


# In[ ]:


#checking on outliers
plt.figure(figsize=(5,5))
sns.boxplot(x= 'variable', y = 'value', data = pd.melt(train[['GrocerySales']]))


# In[ ]:


#treemap
fig = px.treemap(train, path=['GrocerySales'],color='GrocerySales', hover_data=['Day', 'GrocerySales'])
fig.show()


# In[ ]:


#scatter plot
fig = px.scatter(train, x="Day", y="GrocerySales", trendline="ols")
fig.show()


# In[ ]:


#Grocery Sales density plot 
sns.set_style("darkgrid")
sns.kdeplot(data=train['GrocerySales'],label="GrocerySales" ,shade=True)


# ### Model Implementation 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


X = train['Day']
y = train['GrocerySales']
X_pred = test['Day']


# In[ ]:


X = np.array(X)
X = X.reshape(-1, 1)


# In[ ]:


X_pred = np.array(X_pred)
X_pred = X_pred.reshape(-1, 1)


# In[ ]:


#scaling the data
rs = RobustScaler()


# In[ ]:


X_scaled = rs.fit_transform(X)
X_pred_scaled = rs.fit_transform(X_pred)


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=0.3,random_state = 42)


# #### Using Linear Regression (Non-Scaled Data)

# In[ ]:


lr = LinearRegression()
lr.fit(train_X,train_y)


# In[ ]:


predict = lr.predict(test_X)
print('Mean Squared Error: %.2f'% mean_squared_error(test_y,predict))
print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,predict))


# In[ ]:


y_pred_lr = lr.predict(X_pred)


# In[ ]:


y_pred_lr


# #### Using KNN Regressor (Non-Scaled Data)

# In[ ]:


knn = KNeighborsRegressor()
knn.fit(train_X,train_y)


# In[ ]:


predict_knn = knn.predict(test_X)
print('Mean Squared Error: %.2f'% mean_squared_error(test_y,predict_knn))
print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,predict_knn))


# In[ ]:


y_pred_knn = knn.predict(X_pred)


# In[ ]:


y_pred_knn


# #### Using Linear Regression (Scaled Data)

# In[ ]:


scaled_train_X, scaled_test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3,random_state = 42)


# In[ ]:


lr = LinearRegression()
lr.fit(scaled_train_X,train_y)


# In[ ]:


scaled_predict = lr.predict(scaled_test_X)
print('Mean Squared Error: %.2f'% mean_squared_error(test_y,scaled_predict))
print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,scaled_predict))


# In[ ]:


y_pred_scaled_lr = lr.predict(X_pred_scaled)


# In[ ]:


y_pred_scaled_lr


# #### Using KNN Regressor (Scaled Data)

# In[ ]:


knn = KNeighborsRegressor()
knn.fit(scaled_train_X,train_y)


# In[ ]:


scaled_predict_knn = knn.predict(scaled_test_X)
print('Mean Squared Error: %.2f'% mean_squared_error(test_y,scaled_predict_knn))
print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,scaled_predict_knn))


# In[ ]:


y_pred_scaled_knn = knn.predict(X_pred_scaled)


# In[ ]:


y_pred_scaled_knn


# ### Plottting the Prediction

# In[ ]:


sns.distplot(test_y-predict)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


sns.distplot(test_y-predict_knn)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


sns.distplot(test_y-scaled_predict)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


sns.distplot(test_y-scaled_predict_knn)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


df1 = pd.DataFrame(y_pred_lr,columns=['GrocerySales_lr'])
df2 = pd.DataFrame(y_pred_knn,columns=['GrocerySales_knn'])
df3 = pd.DataFrame(y_pred_scaled_lr,columns=['GrocerySales_lrs'])
df4 = pd.DataFrame(y_pred_scaled_knn,columns=['GrocerySales_knns'])


# In[ ]:


temp = [test, df1, df2, df3, df4]


# In[ ]:


test_result = pd.concat(temp, axis=1, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)


# In[ ]:


test_result.head(2)


# In[ ]:


fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_knn)
fig.show()


# In[ ]:


fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_knns)
fig.show()


# In[ ]:


fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_lr)
fig.show()


# In[ ]:


fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_lrs)
fig.show()


# ### The End!!!
