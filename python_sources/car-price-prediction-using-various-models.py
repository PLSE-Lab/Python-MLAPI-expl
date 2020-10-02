#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os

from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


# In[ ]:


training_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


training_data.head()


# In[ ]:


test_data.head()


# In[ ]:


#removing first column from training and test data both
training_data.drop('Unnamed: 0',axis=1,inplace=True)
test_data.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


"Training Null",training_data.isnull().sum(),"TEST Null",test_data.isnull().sum()


# In[ ]:


#checking data types of features and labels
training_data.info()


# In[ ]:


training_data.describe()


# In[ ]:



training_data=pd.get_dummies(training_data,drop_first=True)

test_data=pd.get_dummies(test_data,drop_first=True)

training_data.shape,test_data.shape


# In[ ]:



#Checking for correlation bw independent variables through corr matrix
training_data.corr()


# In[ ]:



x_train=training_data.drop('SellingPrice',1)
y_train=training_data['SellingPrice']

vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(x_train.values,i) for i in range(x_train.shape[1])]

vif['Features']=x_train.columns
vif


# In[ ]:



x_test=test_data.drop('SellingPrice',1)
y_test=test_data['SellingPrice']


# In[ ]:


scaler=StandardScaler()


# In[ ]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[ ]:



model_LR=LinearRegression()
model_LR.fit(x_train,y_train)


# In[ ]:


y_pred_test_LR=model_LR.predict(x_test)


# In[ ]:


#R square of model
model_LR.score(x_test,y_test)


# In[ ]:


rmse_LR=np.sqrt(mean_squared_error(y_test,y_pred_test_LR))
rmse_LR


# In[ ]:



residual_LR = y_pred_test_LR - y_test


# In[ ]:


print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred_test_LR), 'degrees.')
mape_LR = np.mean(100 * (abs(residual_LR) / y_test))
accuracy_LR = 100 - mape_LR
print("Mean Absolute percentage error",mape_LR)
print("Accuracy of model",accuracy_LR)


# In[ ]:


model_RF=RandomForestRegressor(n_estimators=1000,random_state = 10)
model_RF.fit(x_train,y_train)


# In[ ]:


y_pred_test_RF=model_RF.predict(x_test)


# In[ ]:



#R square of model
model_RF.score(x_test,y_test)


# In[ ]:



rmse_RF=np.sqrt(mean_squared_error(y_test,y_pred_test_RF))
rmse_RF


# In[ ]:


residual_RF = y_pred_test_RF - y_test


# In[ ]:



print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred_test_RF), 'degrees.')
mape_RF = np.mean(100 * (abs(residual_RF) / y_test))
accuracy_RF = 100 - mape_RF
print("Mean Absolute percentage error",mape_RF)
print("Accuracy of model",accuracy_RF)


# In[ ]:


########Gradient Boosting Mechanism

model_GB=GradientBoostingRegressor()
model_GB.fit(x_train,y_train)


# In[ ]:


y_pred_test_GB=model_GB.predict(x_test)

model_GB.score(x_test,y_test)


# In[ ]:


rmse_GB=np.sqrt(mean_squared_error(y_test,y_pred_test_GB))
rmse_GB


# In[ ]:


residual_GB = y_pred_test_GB - y_test


# In[ ]:



print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred_test_GB), 'degrees.')
mape_GB = np.mean(100 * (abs(residual_GB) / y_test))
accuracy_GB = 100 - mape_GB
print("Mean Absolute percentage error",mape_GB)
print("Accuracy of model",accuracy_GB)


# In[ ]:


predicted = pd.DataFrame(columns=["Actual","Predicted_LR","Predicted_RF","Predicted_GB"])

predicted["Actual"]=y_test
predicted["Predicted_LR"]=y_pred_test_LR
predicted["Predicted_RF"]=y_pred_test_RF
predicted["Predicted_GB"]=y_pred_test_GB

predicted


# In[ ]:




