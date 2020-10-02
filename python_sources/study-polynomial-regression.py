#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


Fish = pd.read_csv("../input/fish-market/Fish.csv")


# In[ ]:


Fish.info()
Fish.head()


# In[ ]:


y = Fish.Weight.values.reshape(-1,1)
x = Fish.Width.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("weight of fish in Gram")
plt.xlabel("diagonal width in cm")


# In[ ]:


#************** linear regression cizimi *********
lr = LinearRegression()
lr.fit(x,y)

#****** predict *********
y_head = lr.predict(x)

plt.plot(x,y_head, color="red", label="linear")
plt.show()
print("Predict weight of fish in 800 Gram: ", lr.predict([[800]]))


# In[ ]:


#********* Polynomial Regression *****y = b0 + b1*x1 + b2*x2 + b3*x3 +... ***********

polynomial_regression = PolynomialFeatures(degree = 15)    # 5.mertebeye kadar bakalim. Eger uygun degilse degistirmeliyiz
x_polynomial = polynomial_regression.fit_transform(x)


linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()


# This result gives overfitting to us. Then let us degree = 2 using:

# In[ ]:


#********* Polynomial Regression *****y = b0 + b1*x1 + b2*x2 + b3*x3 +... ***********

polynomial_regression = PolynomialFeatures(degree = 2) 
x_polynomial = polynomial_regression.fit_transform(x)


linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color="red",label="poly")
plt.legend()
plt.show()


# 
# Conclusion
# 
# 
#    - After this notebook, my aim is to prepare 'kernel' which is connected to Deep Learning 'not clear' data set.
#    - If you have any suggestions, please could you write for me? I wil be happy for comment and critics!
#    
#    Thank you for your suggestion and votes ;)
# 
# 
