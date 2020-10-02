#!/usr/bin/env python
# coding: utf-8

# *** Fork this Kernel. Do not edit without forking!** 
# * **Upload Your own Data**
# * **Do not forget to "Commit" in order to save your work**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


advertising_multi = pd.read_csv('../input/final.csv')
advertising_multi.head(5)


# In[ ]:


advertising_multi.info()


# In[ ]:


advertising_multi.describe()


# **Visualizing what we have**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.pairplot(advertising_multi)


# In[ ]:


sns.pairplot(advertising_multi, x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=7, aspect=0.9)


# **Assigning the x and y values**

# In[ ]:


X= advertising_multi[['TV','Radio','Newspaper']]
X.head()


# In[ ]:


y= advertising_multi['Sales']
y.head()


# In[ ]:


advertising_multi.tail() #checking the last five rows


# **Splitting Data**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100 )


# In[ ]:


# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# **Importing and Running Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression 
lr = LinearRegression()#Creating a LinearRegression object
lr.fit(X_train, y_train)


# In[ ]:


print(lr.intercept_)


# In[ ]:


coeff_df=pd.DataFrame(lr.coef_,X_test.columns,columns=['Coefficient'])
coeff_df 


# #1 unit increase in TV price raises Sales by 0.045 

# **Predicting with the test values**

# In[ ]:


y_pred = lr.predict(X_test)


# **Calculating Error**

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :' ,r_squared)


# **Checking for P_value**
# y=B      + B1 X1
#   =B X0+B1X1

# In[ ]:


import statsmodels.api as sm
X_train_sm = X_train 
X_train_sm = sm.add_constant(X_train_sm) #sm.add_constant(X) in order to add a constant.
lr_1 = sm.OLS(y_train,X_train_sm).fit() #create a fitted model in one line

lr_1.params #print the coefficients


# In[ ]:


print(lr_1.summary())#let's see the summary for p values


# P value of Newspaper is higher than 0.05 therefore is insignificant

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(5,5))
sns.heatmap(advertising_multi.corr(), annot=True)


# In[ ]:


X_train_new = X_train[['TV','Radio']]#Removing Newspaper
X_test_new = X_test[['TV','Radio']]#Removing Newspaper


# In[ ]:


lr.fit(X_train_new,y_train)


# In[ ]:


y_pred_new = lr.predict(X_test_new)#predicting


# In[ ]:


c   = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test,color="blue", linewidth=2.5,linestyle="-")
plt.plot(c,y_pred,color="red", linewidth=2.5,linestyle="-")
fig.suptitle('Actual versus Predicted', fontsize=25)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Sales', fontsize=16)


# In[ ]:


c   = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred,color="green", linewidth=2.5,linestyle="-")
fig.suptitle('Error', fontsize=25)
plt.xlabel('Index', fontsize=18)
plt.ylabel('y_test-y_pred', fontsize=16)


# In[ ]:


lr.fit(X_train_new, y_train)

