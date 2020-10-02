#!/usr/bin/env python
# coding: utf-8

# **Ecommerce Customer Data Evaluation to find whether improvement of app or improvement of website will improve the sales**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


customers = pd.read_csv("../input/ecommerce-customers/Ecommerce Customers.csv")


# In[ ]:


customers.head()


# In[ ]:


customers.describe()


# In[ ]:


customers.info()


# In[ ]:


sns.jointplot(x="Time on Website",y="Yearly Amount Spent",data=customers,kind='scatter')


# In[ ]:


sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=customers,kind='scatter')


# In[ ]:


sns.jointplot(x="Time on App",y="Length of Membership",data=customers,kind='hex')


# In[ ]:


sns.pairplot(customers)


# In[ ]:


sns.lmplot(x="Length of Membership",y="Yearly Amount Spent",data = customers)


# In[ ]:


customers.columns


# In[ ]:


y = customers["Yearly Amount Spent"]


# In[ ]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


lm.coef_


# In[ ]:


prediction=lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,prediction)
plt.xlabel("Y Test(True Values)")
plt.ylabel("Predicted Values")


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE', metrics.mean_absolute_error(y_test,prediction))
print('MSE', metrics.mean_squared_error(y_test,prediction))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[ ]:


metrics.explained_variance_score(y_test,prediction)


# In[ ]:


sns.distplot((y_test-prediction),bins=50)


# In[ ]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf


# In[ ]:




