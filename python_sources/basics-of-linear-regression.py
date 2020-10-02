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


# **Importing Libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **Importing dataset**

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# **Getting all information regarding train and test dataset**

# > Regarding Train dataset

# In[ ]:


train_data.info()


# For 700 values of x, one value is missing in y.
# Since that row is not needed to model our data, it is not needed at all. So, it is advisable to drop that.
# Dropping that row first:

# In[ ]:


train_data.dropna(axis=0,how='any',inplace = True)


# Describing the data to get some insights on trained data...
# Here, you can see that count will now bw equal i.e. there is a value of y for an value of x.

# In[ ]:


train_data.describe()


# Let's plot the columns to see any correlation

# In[ ]:


plt.scatter(x='x',y='y',data = train_data,color = 'green')
plt.show()


# So, here can be a beautiful simple linear regression possible. Lets make the model then...

# In[ ]:


x = train_data.iloc[:,:1]
y = train_data.iloc[:,:-1]


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)


# So, our model is ready and next thing is to predict the values...

# In[ ]:


test_data.info()


# In[ ]:


x_test = test_data.iloc[:,:1]
y_test = test_data.iloc[:,:-1]


# In[ ]:


y_predict = lr.predict(x_test)


# Comapring with the help of Mean-Squared-Error and R2-score

# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)


# In[ ]:


print(mse)
print(r2)


# And our model is unexpectedly undoubtable.
# plotting y_test and y_predict : 

# In[ ]:


plt.figure(figsize=(10,7))
plt.scatter(x = x_test, y = y_predict, color = 'green',label='Predicted')
plt.scatter(x = x_test, y = y_test, color = 'red',label = 'Original')
plt.legend()
plt.show()


# And since, you can see the Original is just exact the Predicted and that is why, our model is correctly proven.

# > **Upvote if you like it...**
