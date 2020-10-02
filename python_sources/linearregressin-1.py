#!/usr/bin/env python
# coding: utf-8

# imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# load the data

# In[ ]:


data = pd.read_csv('../input/Salary_Data.csv')
data.head(3)


# In[ ]:


data.info()


# In[ ]:


X = data.iloc[:,0].values
y = data.iloc[:,1].values

print(X.shape)
print(y.shape)


# In[ ]:


X = X.reshape(-1,1)
X.shape


# split the data into train and test data

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state = 17)
X_train.shape


# apply linear regeression

# In[ ]:


from sklearn.linear_model import LinearRegression

lnr_clf = LinearRegression()

lnr_clf.fit(X_train,y_train)


# In[ ]:


y_pred = lnr_clf.predict(X_test)
y_pred


# ploting

# In[ ]:


#plot for train data
plt.scatter(X_train,y_train,color ='red')
plt.plot(X_train,lnr_clf.predict(X_train),color = 'green')

plt.xlabel('Year of exp')
plt.ylabel('salary')
plt.show()


# In[ ]:


#plot for test data

plt.scatter(X_test,y_test,color ='red')
plt.plot(X_test,lnr_clf.predict(X_test),color ='green')
plt.xlabel('Year of exp')
plt.ylabel('salary')
plt.show()


# Y = mX+ C
# 
#  m is the coefficient of X here

# In[ ]:


print('Co-efficient = ', lnr_clf.coef_)


# mean squar error

# In[ ]:


MSE = np.mean((lnr_clf.predict(X_test) - y_test)**2)
print('MSE=',MSE)


# calculating mean squar error by sklearn 

# In[ ]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)

print('MSE=',mse)


# varience score

# In[ ]:


lnr_clf.score(X_test,y_test)


# In[ ]:




