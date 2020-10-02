#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15,7)
#plt.rcParams['font.cursive'] = ['Source Han Sans TW', 'sans-serif']
from sklearn.model_selection import train_test_split


# In[ ]:


housing_data_set = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


housing_data_set.head()


# In[ ]:


housing_data_set.describe()


# In[ ]:


housing_data_set.shape


# In[ ]:


#plt.figure(figsize=(17,6))
sns.heatmap(housing_data_set.corr(),cmap='viridis',annot=True)


# In[ ]:


housing_data_set.info()


# In[ ]:


# Looking for nulls
print(housing_data_set.isnull().any())
# Inspecting type
print(housing_data_set.dtypes)


# In[ ]:


# Dropping the id and date columns
house = housing_data_set.drop(['id', 'date'],axis=1)


# In[ ]:


y = house['price']
X = house.drop(['price'], axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


lm.coef_


# In[ ]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
cdf.head()


# In[ ]:


#Predictions
predication = lm.predict(X_test)
predication


# In[ ]:


plt.scatter(y_test,predication)


# In[ ]:


#Residual
sns.distplot((y_test-predication))


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test,predication)


# In[ ]:


metrics.mean_squared_error(y_test,predication)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test,predication))


# In[ ]:


#Building the optimal model using Backwards Elimination
import statsmodels.formula.api as sm


# In[ ]:


X = np.append(arr = np.ones((21613,1)).astype(int), values= X,axis=1)


# In[ ]:


X_opt = house.drop(['price'], axis=1)


# In[ ]:


lm_OLS = sm.OLS(endog= y,exog= X_opt).fit()
lm_OLS.summary()


# In[ ]:





# In[ ]:





# In[ ]:




