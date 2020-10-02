#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Ecom=pd.read_csv('../input/Ecommerce Customers')


# In[ ]:


Ecom.head()


# In[ ]:


Ecom.columns


# In[ ]:


sns.pairplot(Ecom)


# In[ ]:


Ecom.describe()


# In[ ]:


sns.jointplot(data=Ecom,x='Time on Website',y='Yearly Amount Spent')


# In[ ]:


sns.jointplot(data=Ecom,x='Time on App',y='Yearly Amount Spent')


# In[ ]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=Ecom)


# In[ ]:


Ecom.columns


# In[ ]:


y=Ecom['Yearly Amount Spent']


# In[ ]:


x=Ecom[[  'Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm=LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


lm.coef_


# In[ ]:


predictions=lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)
plt.xlabel('y test(actual values)')
plt.ylabel('predicted values')


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE',metrics.mean_absolute_error(y_test,predictions))
print('MSE',metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[ ]:


metrics.explained_variance_score(y_test,predictions)


# In[ ]:


sns.distplot(y_test-predictions)


# In[ ]:


cdf=pd.DataFrame(lm.coef_,x.columns,columns=['Coeffs'])
cdf


# In[ ]:




