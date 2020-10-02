#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


insurance = pd.read_csv("../input/insurance.csv")
insurance.shape


# In[ ]:


insurance.ndim


# In[ ]:


#Summary Statistics
insurance.describe()


# In[ ]:


correlation = insurance.corr(method = 'pearson')


# In[ ]:


correlation


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')


# In[ ]:


X = np.asarray(insurance)


# In[ ]:


X


# In[ ]:



variable_labels = np.asarray(insurance.columns)[0:]


# In[ ]:


var = variable_labels[0:6]


# In[ ]:


plt.scatter(insurance["sex"],insurance["charges"])


# In[ ]:


plt.scatter(insurance["age"],insurance["charges"])


# In[ ]:


plt.scatter(insurance["bmi"],insurance["charges"])


# In[ ]:


plt.scatter(insurance["children"],insurance["charges"])


# In[ ]:


plt.scatter(insurance["smoker"],insurance["charges"])


# In[ ]:


plt.scatter(insurance["region"],insurance["charges"])


# In[ ]:


#Since this a clean dataset going for datamodelling with LinearRegression
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


#Replace Categoricaldata with dummy variable
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(insurance['sex'].drop_duplicates())
insurance['sex']=encoder.transform(insurance['sex'])
encoder.fit(insurance['smoker'].drop_duplicates())
insurance['smoker']=encoder.transform(insurance['smoker'])
encoder.fit(insurance['region'].drop_duplicates())
insurance['region']=encoder.transform(insurance['region'])
insurance.head(10)


# In[ ]:



X = insurance[['age','sex','bmi','children','smoker','region']]
y = insurance['charges']
X_train = X[:-30]
X_test  = X[-30:]
y_train = y[:-30]
y_test  = y[-30:]


# In[ ]:


lm = LinearRegression()
model = lm.fit(X_train,y_train)


# In[ ]:


model.coef_


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


model.intercept_


# In[ ]:


plt.scatter(y_test,y_pred)


# In[ ]:


#LinearRegression RMSE outcome
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


#GBM to improve RMSE and increase feature importance
from sklearn.ensemble import GradientBoostingRegressor
gbm=GradientBoostingRegressor(n_estimators=100)
gbm.fit(X_train,y_train)
y_pred_gbm = gbm.predict(X_test)
error_gbm = metrics.mean_squared_error(y_test,y_pred_gbm)
print(np.sqrt(error_gbm))


# In[ ]:


gbm.feature_importances_


# In[ ]:


feat_imp = pd.Series(gbm.feature_importances_,var).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')


# In[ ]:




