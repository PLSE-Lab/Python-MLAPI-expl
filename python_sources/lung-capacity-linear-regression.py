#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import all the reuqired libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


lc=pd.read_csv(r"//kaggle/input/lung-capacity-smoker-and-non-smoker/Lung-Capacity-Smoker.csv")


# In[ ]:


lc.head()


# In[ ]:


lc.shape


# In[ ]:


#hot encoding for categorical variables
lc.Gender.replace({"male":1,"female":0},inplace=True)
lc.Smoke.replace({"yes":1,"no":0},inplace=True)
lc.Caesarean.replace({"yes":1,"no":0},inplace=True)


# In[ ]:


#check for null values
lc.isnull().sum()


# In[ ]:


lc_x=lc.iloc[:,1:6]
lc_y=lc.iloc[:,0]


# In[ ]:


#Sampling of data into train & test
import sklearn

from sklearn.model_selection import train_test_split


# In[ ]:


lc_x_train, lc_x_test, lc_y_train, lc_y_test=train_test_split(lc_x,lc_y,test_size=0.2, random_state=101)


# In[ ]:


#linear modelling & Prediction


# In[ ]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(lc_x_train,lc_y_train) #training the algorithm
pred_val=reg.predict(lc_x_test)
pred_val


# In[ ]:


#Calulate the best values for intercept & slope
print(reg.coef_)
#slope
print(reg.intercept_)


# In[ ]:


#R-Square
reg.score(lc_x_train,lc_y_train)


# In[ ]:


## Convert test data and predicted data in Series for concatenation
X= []
for i in lc_y_test:
    X.append(i)
X = pd.Series(X)
    
pred_val= pd.Series(pred_val)
pred_val


# In[ ]:


## compare actual and predicted values
final = pd.concat({"Actual" : X,"Predicted": pred_val}, axis=1,join='outer')
final


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(lc_y_test, pred_val))  
print('Mean Squared Error:', metrics.mean_squared_error(lc_y_test, pred_val))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(lc_y_test,pred_val)))


# In[ ]:


#check the difference between actual & predicated values
df = pd.DataFrame({'Actual': X, 'Predicted': pred_val})
df1 = df.head(25)


# In[ ]:


#Plot the actual & predicated values
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




