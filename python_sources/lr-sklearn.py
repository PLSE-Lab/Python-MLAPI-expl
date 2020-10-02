#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data=pd.read_csv("../input/canada_per_capita_income.csv", header=None)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


plt.figure(figsize=(16, 8))
plt.scatter(data[0],data[1],color="b",marker="o")
plt.xticks(np.arange(1960,2030,step=10))
plt.yticks(np.arange(3000,46000,step=2000))
plt.xlabel("year")
plt.ylabel("capita")
plt.title("capita and year")


# In[ ]:


x=data[0].values.reshape(-1,1)
y=data[1].values.reshape(-1,1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[ ]:


x2=np.array(X_train**2)
x3=np.array(X_train**3)
x4=np.array(X_train**4)


# In[ ]:


X=np.append(X_train,x2,axis=1)
X=np.append(X,x3,axis=1)
X.shape


# In[ ]:


reg = LinearRegression()
reg.fit(X,y_train)


# print("The linear model is: Y = {:.5} + {:.5}X + {:.5x2} + {:.5}x3".format(reg.intercept_[0], reg.coef_[0][0],reg.coef_[1][0],reg.coef_[2][0]))

# print("The linear model is: Y = {:.5} + {:.5}X + {:.5x2} + {:.5}x3".format(reg.intercept_[0], reg.coef_[0][0],reg.coef_[1][0],reg.coef_[2][0]))

# In[ ]:


predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(X_train,y_train,c='black')
plt.plot(X_train,predictions,c='blue',linewidth=2)
plt.xlabel("Years")
plt.ylabel("Capita ($)")
plt.show()


# In[ ]:


y_pred=reg.predict(X)


# In[ ]:


metrics=np.vstack((y,y_pred)).T


# In[ ]:


df=pd.DataFrame({'Actual':y_train.flatten(),'Predicted':y_pred.flatten()})
df


# In[ ]:


print('Mean Absolute Error:',mean_absolute_error(y_train,y_pred))


# **Test Data**

# In[ ]:


x2=np.array(X_test**2)
x3=np.array(X_test**3)
x4=np.array(X_test**4)


# In[ ]:


Xa=np.append(X_test,x2,axis=1)
Xa=np.append(Xa,x3,axis=1)
Xa.shape


# In[ ]:


y_predt=reg.predict(Xa)


# In[ ]:


metrics=np.vstack((y_test,y_predt)).T


# In[ ]:


df=pd.DataFrame({'Year':X_test.flatten() ,'Actual':y_test.flatten(),'Predicted':y_predt.flatten()})
df


# In[ ]:


print('Mean Absolute Error:',mean_absolute_error(y_test,y_predt))

