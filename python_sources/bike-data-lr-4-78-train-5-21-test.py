#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


bike_df=pd.read_csv("../input/bike_share.csv")


# In[ ]:


bike_df.shape


# In[ ]:


bike_df.isna().sum()


# In[ ]:


bike_df[bike_df.duplicated()]


# In[ ]:


bike_df_unique=bike_df.drop_duplicates()
bike_df_unique.shape


# In[ ]:


bike_df_unique.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(20, 10))
sns.boxplot(data=bike_df_unique,orient="h")


# In[ ]:


bike_df_unique.describe().transpose()


# **From above boxplot and statistic info, we see count and registered variable have data spread closely matches with range from 50 to 1000 with lot of outlier above 500 for registered variable and 630 for count variable. 
# 
# **Temp and atemp have data spread matches with range from 0.50 to 50 with no outlier. probaly scale would be in degrees**
# 
# **Humidity and windspeed also have smaller distribution with so many outlier presence in windspeed than humidity **
# 
# **Removing outlier was given by following kernel**
# 
# https://www.kaggle.com/anishkanth/insurance-premium-rmse-4574

# In[ ]:


lower_bnd = lambda x: x.quantile(0.25) - 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )
upper_bnd = lambda x: x.quantile(0.75) + 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )


# In[ ]:


bike_df_unique.shape


# In[ ]:


for i in bike_df_unique.columns:
    bike_df_clean = bike_df_unique[(bike_df_unique[i] >= lower_bnd(bike_df_unique[i])) & (bike_df_unique[i] <= upper_bnd(bike_df_unique[i])) ] 
bike_df_clean.shape


# In[ ]:


plt.figure(figsize=(20, 10))
sns.boxplot(data=bike_df_clean,orient="h")


# In[ ]:


bike_df_clean.corr()


# **Strong correlation value exist between count and registered (0.96) variable **
# **Next correlation for count is casual(0.717)**
# **Registered Variable has correlation with  casual (0.51) followed by temp (0.30) and atemp(0.30)**
# **Casual Variable has correlation with temp(0.46) and atemp(0.46)**
# **Humidity Variable has correlation with weather(0.40)**
# **Temp variable has correlation with atemp(0.98)**
# **Season variable has correlation with temp(0.263287) and atemp(0.269848)**

# In[ ]:


#sns.heatmap(bike_df_clean,annot=True)


# In[ ]:


sns.pairplot(data=bike_df_clean)


# **Above plot is not properly visible. Also some of numerical variable is discrete like season,weather, holiday and working day which we can avoid in pair plot**
# 
# **get only continous variable and try plot pairplot**

# In[ ]:


num_disc_list=['season', 'holiday','workingday','weather']
num_cont_list=bike_df_clean.columns.drop(num_disc_list)
sns.pairplot(data=bike_df_clean,vars = num_cont_list,hue=num_disc_list[0])


# In[ ]:


#for i in num_disc_list:
 #   sns.pairplot(data=bike_df_clean,vars = num_cont_list,hue=i)


# In[ ]:


# Importing necessary package for creating model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


# In[ ]:


#using one hot encoding
X=bike_df_clean.drop(columns='count')
y=bike_df_clean[['count']]


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=1234)


# In[ ]:


model = LinearRegression()

model.fit(train_X,train_y)


# In[ ]:


# Print Model intercept and co-efficent
print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)


# In[ ]:


cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])
cdf


# In[ ]:


# Print various metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print("Predicting the train data")
train_predict = model.predict(train_X)
print("Predicting the test data")
test_predict = model.predict(test_X)
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")
print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")
import numpy as np
print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")
print("R^2")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))
print("MAPE")
print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)
print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)


# In[ ]:


#Plot actual vs predicted value
plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted count",fontsize=25)
plt.xlabel("Actual count",fontsize=18)
plt.ylabel("Predicted count", fontsize=18)
plt.scatter(x=test_y,y=test_predict)


#  **with above graph, we see there exist linear relationship between Predicted output and actual count**

# **Tried running same model without removing outlier.It also gives linear relationship but see discrete line with small split at end **

# In[ ]:


#using one hot encoding
X=bike_df_unique.drop(columns='count')
y=bike_df_unique[['count']]


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=1234)


# In[ ]:


model = LinearRegression()

model.fit(train_X,train_y)


# In[ ]:


# Print Model intercept and co-efficent
print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)


# In[ ]:


cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])
cdf


# In[ ]:


# Print various metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print("Predicting the train data")
train_predict = model.predict(train_X)
print("Predicting the test data")
test_predict = model.predict(test_X)
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")
print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")
import numpy as np
print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")
print("R^2")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))
print("MAPE")
print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)
print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)


# In[ ]:


#Plot actual vs predicted value
plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted count",fontsize=25)
plt.xlabel("Actual count",fontsize=18)
plt.ylabel("Predicted count", fontsize=18)
plt.scatter(x=test_y,y=test_predict)


# **We will try to implement knn for this Linear regression to see how accuracy calculated**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.3, random_state = 100)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)


# In[ ]:


k = 5
#Train Model and Predict  
neigh = KNeighborsRegressor(n_neighbors = k).fit(X_train,y_train)
neigh


# In[ ]:


### Predicting
#we can use the model to predict the test set:

yhat = neigh.predict(X_test)
yhat[0:5]


# In[ ]:


mean_squared_error(y_test,yhat)


# In[ ]:


yhat_train = neigh.predict(X_train)
yhat_train[0:5]


# In[ ]:


mean_squared_error(y_train,yhat_train)


# **I Tried increase K value from 3 to 13 and found that k-value lower for K=5 . will try to validate same in elbow curve**

# In[ ]:




