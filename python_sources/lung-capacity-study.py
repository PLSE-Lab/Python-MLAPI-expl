#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_excel("../input/dataset/Processed_Lungcap.xls")


# In[ ]:


df.head()


# In[ ]:


df.isnull().any()


# In[ ]:


df.info()


# In[ ]:


sns.pairplot(df)


# In[ ]:


from IPython.display import Image
Image("../input/dashboard/Capture1.PNG")


# In[ ]:


df.corr()


# In[ ]:


X = df[['Height']]
y = df['Lungcap']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 2)
X_train.head()


# In[ ]:


regression_model = LinearRegression()
regression_model.fit(X_train,y_train)


# In[ ]:


intercept = regression_model.intercept_
print("The intercept for our model is {}".format(intercept))


# In[ ]:


regression_model.coef_


# In[ ]:


regression_model.score(X_test,y_test)


# In[ ]:


rmse = np.sqrt(np.mean((regression_model.predict(X_test)-y_test)**2))
rmse


# In[ ]:


y_pred = regression_model.predict(X_test)


# In[ ]:


R_square = 1 - (np.sum((y_test-y_pred)**2)/(np.sum((y_test-np.mean(y_test))**2)))
R_square


# In[ ]:


np.corrcoef(y_test,y_pred)


# In[ ]:


plt.scatter(y_test,y_pred)


# In[ ]:


model2 = LinearRegression()
model2.fit(X,y)


# In[ ]:


intercept = model2.intercept_
print("The intercept for our model is {}".format(intercept))


# In[ ]:


model2.coef_


# In[ ]:


model2.score(X_test,y_test)


# In[ ]:


y_pred2 = model2.predict(X)


# In[ ]:


rmse = np.sqrt(np.mean((model2.predict(X)-y)**2))
rmse


# In[ ]:


R_square = 1 - (np.sum((y-y_pred2)**2)/(np.sum((y-np.mean(y))**2)))
R_square


# In[ ]:


np.corrcoef(y,y_pred2)


# In[ ]:


df.plot.scatter(x='Height',y='Lungcap')
plt.plot(X,y_pred2,c='red')


# In[ ]:





# In[ ]:





# In[ ]:




