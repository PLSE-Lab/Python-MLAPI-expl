#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# #### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


customers=pd.read_csv('../input/Ecommerce Customers.csv')


# In[ ]:


customers.head()


# In[ ]:


customers.info()


# In[ ]:


customers.describe()


# #### Exploratory Data Analysis

# In[ ]:


#create a jointplot to compare the Time on Website and Yearly Amount Spent columns.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# In[ ]:


#Create jointplot to compare the Time on App column
sns.jointplot(x='Time on App',y ='Yearly Amount Spent', data = customers)


# #### Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.

# In[ ]:


sns.jointplot(x='Time on App',y ='Length of Membership', data = customers, kind='hex')


# #### Lets explore these types of relationship across the entire data set using pairplot

# In[ ]:


sns.pairplot(customers)


# #### Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.

# In[ ]:


sns.set(color_codes=True)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=customers)


# #### Training and Testing dataset

# In[ ]:



X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y= customers['Yearly Amount Spent']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# #### Training the model

# In[ ]:


from sklearn.linear_model import LinearRegression


# ##### Create an instance of linear regression model

# In[ ]:


lm=LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


lm.coef_


# #### Predicating test data

# In[ ]:


y_pred=lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,y_pred)
plt.xlabel("Predicated")
plt.ylabel("Y test")


# #### Evaluating the model

# In[ ]:


import sklearn.metrics as metrics
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, y_pred)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))


# #### Residuals

# In[ ]:


sns.distplot((y_test-y_pred))


# #### Conclusion

# ##### Recreate the dataframe below.

# In[ ]:


pd.DataFrame(lm.coef_ , X.columns, columns=['Coeffecient'])


# In[ ]:




