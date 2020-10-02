#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Importing Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # **Importing dataset**

# In[ ]:


df=pd.read_csv('/kaggle/input/insurance/insurance.csv')


# # **Analysing the DataSet**

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # **Visualisation**

# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# # Furture analysis of the trend according to charges
# ## **Spliting the data into Testing and Training**

# In[ ]:


X=df[['age','bmi', 'children']]
y=df['charges']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
lm=LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


coef=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[ ]:


coef


# **So according to the given trend : The first factor of geting a high charge is by having more child and also the second factor for which the charge will increase is bmi**

# # **Prediction**

# In[ ]:


predictions=lm.predict(X_test)
print(predictions)


# In[ ]:


plt.scatter(y_test,predictions)


# **Residural Histogram plot**

# In[ ]:


sns.distplot((y_test-predictions),bins=50);


# **Linear Regression model**

# In[ ]:


sns.lmplot(x='bmi',y='charges',hue='children',col='sex',data=df)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # Conclusion
# 
# Conclusion states the following analysis : 
# 
# <ul><li> If you have 1 child the chances of getting fine increase to 418.900655 amount
#     <li> You will be charged 313.271462 amount per BMI Unit as predicted
#     <li> As per 1 year of age you may get 243.409566 amount charged as predicted

# # THANK YOU

# In[ ]:




