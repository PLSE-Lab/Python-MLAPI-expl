#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing library
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as seabornInstance


# In[ ]:


#importing data set
dataset=pd.read_csv('../input/houseprice.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[ ]:


plt.figure(figsize=(6,4))
plt.tight_layout()
seabornInstance.distplot(y)


# In[ ]:


#Splitting the data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state = 0)


# In[ ]:


#Training
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[ ]:


#intercept and coff
y_predict=regressor.predict(x_test)
print(regressor.intercept_)
print(regressor.coef_)


# In[ ]:


#prediction
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})
df


# In[ ]:


#Actual vs prediction plotting
df1 = df.head(25)
df1.plot(kind='bar',figsize=(8,3))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


#Accuracy
import sklearn
from sklearn import metrics
print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_predict)))

