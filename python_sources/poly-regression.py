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


# In[ ]:


import pandas as pd
df = pd.read_csv('../input/the-housing-data/Housing_data.csv',sep=';')


# In[ ]:


df.head()


# In[ ]:


x = df.drop(['price'],axis=1)
y = df['price'].values


# In[ ]:


from sklearn.linear_model import LinearRegression
lmodel1 = LinearRegression()


# In[ ]:


lmodel1.fit(x,y)


# In[ ]:


yp1 = lmodel1.predict(x)


# In[ ]:


y.shape
yp1.shape
#type(y)
#type(yp1)
abs(y-yp1).mean()


# In[ ]:


x
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
scl.fit(x)
x_scl = scl.transform(x)


# In[ ]:


pd.DataFrame(x_scl)[2].std()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree = 2)


# In[ ]:


pol.fit(x_scl)
x_pol = pol.transform(x_scl)


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_pol,y,test_size=0.2)


# In[ ]:


lmodel2 = LinearRegression()
lmodel2.fit(xtrain,ytrain)


# In[ ]:


#yp2 = lmodel2.predict(x_pol)
ytrain_p = lmodel2.predict(xtrain)
#abs(y-yp2).mean()
abs(ytrain-ytrain_p).mean()


# In[ ]:


len(lmodel2.coef_)


# In[ ]:


#yp2 = lmodel2.predict(x_pol)
ytest_p = lmodel2.predict(xtest)
#abs(y-yp2).mean()
abs(ytest-ytest_p).mean()


# Predictions on user given value

# In[ ]:


k = np.array([[6360 , 2 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0]])
k_t = scl.transform(k)
k_p = pol.transform(k_t)
lmodel2.predict(k_p)


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_scl,y,test_size=0.2)


# In[ ]:


errors_train = []
errors_test = []
for i in range(1,10):
    pol = PolynomialFeatures(degree = i)
    pol.fit(xtrain)
    x_train = pol.transform(xtrain)
    
    pol.fit(xtest)
    x_test = pol.transform(xtest)
        
    lmodel2 = LinearRegression()
    lmodel2.fit(x_train,ytrain)
    
    yp2_train = lmodel2.predict(x_train)
    e_train = abs(ytrain-yp2_train).mean()
    
    yp2_test = lmodel2.predict(x_test)
    e_test = abs(ytest-yp2_test).mean()
    
    errors_train.append(e_train)
    errors_test.append(e_test)


# In[ ]:


errors_train


# In[ ]:


errors_test


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(range(1,10),errors_train)
plt.plot(range(1,10),errors_test,c='red')
plt.show()


# In[ ]:


import numpy as np
a = np.array([[2,5,6,7,8,6]]).reshape(3,2)
a
pol.fit(a)
pol.transform(a)


# In[ ]:


a = np.array([[2,3,5,6,4]]).reshape(5,1)


# In[ ]:


a.mean()


# In[ ]:


a - a.mean()


# In[ ]:


b = (a-a.mean())/a.std()


# In[ ]:


b


# In[ ]:


b.std()


# In[ ]:




