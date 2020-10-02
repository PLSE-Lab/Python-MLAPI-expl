#!/usr/bin/env python
# coding: utf-8

# ### Lets download the dataset

# ### Import all the necessary packages

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import classification_report,accuracy_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import seaborn as sbn


# In[ ]:


os.chdir('../input')


# In[ ]:


os.listdir()


# ### Load the dataset

# In[ ]:


data = pd.read_csv('energydata_complete.csv')


# In[ ]:





# In[ ]:


data.shape


# ### Lets see the columns which are here

# In[ ]:


data.columns


# ### Lets see the data type of each column

# In[ ]:


data.dtypes


# ### As we can see the data type is float or int everywhere. So, we don't need to do any special preprocessing for each column

# ### We need to predict the Application Energy

# ### Lets check for null values

# In[ ]:


data.isnull().sum()


# In[ ]:


data['date']


# ### Lets convert the format of the dat to seconds

# In[ ]:


data['date'] = pd.to_timedelta(data['date'].astype('datetime64')).dt.seconds


# In[ ]:


data['date'].dtype


# In[ ]:


y = data['Appliances']


# In[ ]:


X = data
del X['Appliances']


# In[ ]:


X.columns


# In[ ]:


X.shape


# ### Now lets look at the distribution of each feature to understand what extra preprocessing we need to do

# In[ ]:


pd.plotting.scatter_matrix(data.iloc[:,:7],figsize = (10,10))


# ### Some insights from the above plot
# #### - T1,RH_1, T2,RH_2,T3 are normally distributed. No extra preprocessing needed.
# #### - lights seems to be skewed. Need preprocessing
# #### - date seems to be same for all the examples
# #### - Each pair of RH and T (RH_1&RH_2 or T1&T2) are positively correlated. This make sense as well as each T is the temperature in different regions of the house.So, they will vary together. Similar for the RH as each of them is the Humidity in different regions of the house.
# 

# #### As all the T and RH bring the same information. So, its is better to keep just one of them. We will keep T1 and RH_1

# In[ ]:


X = X.drop(['T2','RH_2','T3'],axis=1)


# In[ ]:


X.shape


# #### Lets look at date

# In[ ]:


X['date'].describe()


# In[ ]:


plt.plot(X['date'])


# In[ ]:


plt.scatter(np.arange(len(X['date'])),X['date'])


# 

# In[ ]:


X = X.drop(['date'],axis=1)


# In[ ]:


X.shape


# #### Lets look at the lights

# In[ ]:


plt.hist(X['lights'])


# In[ ]:


X['lights'].describe()


# ### It is highly skewed, So we need to see waht information is important here.

# #### Lets look at the other features

# In[ ]:


pd.plotting.scatter_matrix(data.iloc[:,14:23],figsize = (10,10))


# #### What we can infer from above.
# #### - Except T_out and Rh_out all other T and RH are correlated
# #### - RH_out is a bit skewed
# #### - Pressure_mm is normally distributed

# #### So we keep T_out, RH_out, RH_9 and T9. All else RH and T we remove

# In[ ]:


X = X.drop(['T7','T8','RH_7','RH_8'],axis=1)


# In[ ]:


X.shape


# In[ ]:


pd.plotting.scatter_matrix(data.iloc[:,23:28],figsize=(10,10))


# ### What we can infer:
# - ### rv1 and rv2 are highly correlated
# - ### Both rv seems to have very less variance in them.
# - ### Except rv's all the other feature have a normal distribution
# - ### We will analyse Windspeed, Tdewpoint, visibility independently

# In[ ]:


del X['rv1']


# In[ ]:


plt.scatter(np.arange(len(X['rv2'])),X['rv2'])


# ### As we can see that rv2 does not have any significant variation in it. So, we can remove it.

# In[ ]:


del X['rv2']


# In[ ]:


X.shape


# ### Lets divide the dataset into train and test

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[ ]:


X_train.shape,X_test.shape


# ### Lets scale the features

# In[ ]:


X.dtypes


# In[ ]:


scaler = StandardScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_train.shape


# In[ ]:


def adjusted_r_square(n,k,y,yhat):
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (((1-r_squared)*(k-1))/(n-k-1))
    return r_squared,adjusted_r_squared


# ### This function will return first the R-Squared and second the Adjusted R Squared

# ### Now we can start making the model

# - #### Logistic Regression

# In[ ]:


linReg = LinearRegression()
linReg.fit(X_train,y_train)


# In[ ]:


linReg.score(X_train,y_train)


# In[ ]:


y_predLinReg = linReg.predict(scaler.transform(X_test))


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predLinReg)


# In[ ]:


mean_squared_error(y_predLinReg,y_test)


# - #### Decision Tree Regression

# In[ ]:


DecisonReg = DecisionTreeRegressor()
DecisonReg


# In[ ]:


DecisonReg.fit(X_train,y_train)
DecisonReg.score(X_train,y_train)


# In[ ]:


y_predDecisonReg = DecisonReg.predict(scaler.transform(X_test))

mean_squared_error(y_predDecisonReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predDecisonReg)


# In[ ]:


DecisonReg.feature_importances_


# In[ ]:


X.columns


# #### Here decison tree seems to have over fitted

# - #### RandomForest

# In[ ]:


RandomForestReg = RandomForestRegressor(n_estimators=600)
RandomForestReg


# In[ ]:


RandomForestReg.fit(X_train,y_train)
RandomForestReg.score(X_train,y_train)


# In[ ]:


y_predRandomForestReg = RandomForestReg.predict(scaler.transform(X_test))

mean_squared_error(y_predRandomForestReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predRandomForestReg)


# - ### Support vector machine 

# In[ ]:


svmReg = SVR(C = 0.75)
svmReg


# In[ ]:


svmReg.fit(X_train,y_train)
svmReg.score(X_train,y_train)


# In[ ]:


y_predsvmReg = svmReg.predict(scaler.transform(X_test))

mean_squared_error(y_predsvmReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predsvmReg)


# - #### Here we can see that the RBF kernel is performing badly. Lets try Linear kernel

# In[ ]:


svmReg = SVR(kernel='linear')
svmReg


# In[ ]:


svmReg.fit(X_train,y_train)
svmReg.score(X_train,y_train)


# In[ ]:


y_predsvmReg = svmReg.predict(scaler.transform(X_test))

mean_squared_error(y_predsvmReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predsvmReg)


# - #### The linear kernel is performing badly. Lets try polynomial kernel

# In[ ]:


svmReg = SVR(kernel='poly')
svmReg


# In[ ]:


svmReg.fit(X_train,y_train)
svmReg.score(X_train,y_train)


# In[ ]:


y_predsvmReg = svmReg.predict(scaler.transform(X_test))

mean_squared_error(y_predsvmReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predsvmReg)


# ### As we ca see that currently the ensemble learning models are performing the best. So, lets try another ensemble learning 
# ### classifier Graient boosting 

# - #### Gradient Boosting 

# In[ ]:


gradBoostReg = GradientBoostingRegressor(loss='ls',n_estimators=500)
gradBoostReg


# In[ ]:


gradBoostReg.fit(X_train,y_train)
gradBoostReg.score(X_train,y_train)


# In[ ]:


y_predgradBoostReg = gradBoostReg.predict(scaler.transform(X_test))

mean_squared_error(y_predgradBoostReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predgradBoostReg)


# #### Here we worked with all type of loss functions and the best was least square(ls). At higher number of estimator it is overfitting.

# ### Lastly we try XGBoost

# In[ ]:


XGBoostReg = xgb.XGBRegressor(objective="reg:linear",n_estimators=500)
XGBoostReg


# In[ ]:


XGBoostReg.fit(X_train,y_train)
XGBoostReg.score(X_train,y_train)


# In[ ]:


y_predXGBoostReg = XGBoostReg.predict(scaler.transform(X_test))

mean_squared_error(y_predXGBoostReg,y_test)


# In[ ]:


adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predXGBoostReg)


# ## Here we conclude
# - ### Our best result was from Random forest with 600 estimators. A training R-Square of 93.9, testing R_square of 53.65, mse of 4950.
# - ### Other ensemble learning models are overfitting currently.

# ### For the next stage we can work with dimensionality reduction algorithms like PCA

# ### Thank you
