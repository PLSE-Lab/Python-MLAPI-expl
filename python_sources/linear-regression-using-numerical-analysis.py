#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()


# In[ ]:


data = {'Size':[6,8,12,14,18],'Price':[350,775,1150,1395,1675]}
df = pd.DataFrame(data)


# In[ ]:


df


# In[ ]:


sns.lmplot(data=df, x='Size',y='Price')


# In[ ]:


df['Size_Price'] = df.Size*df.Price
df['Size_Squared'] = df.Size ** 2


# * Equation for Finding Slope Below:
# $$\Large{m = \frac{\bar{x}.\bar{y} - \overline{x.y}} {{\bar{x}^2}-{\overline{x^2}}}} $$
# * Equation for Finding Intercept by Y-Axis:
# $$\Large{ c = \bar{y} - m.\bar{x}} $$

# In[ ]:


Size_bar = np.sum(df.Size) / len(df.Size)
Price_bar = np.sum(df.Price) / len(df.Price)
Size_Price_bar = np.sum(df.Size_Price) / len(df.Size_Price)
Size_Squared_bar = np.sum(df.Size_Squared) / len(df.Size_Squared)
print(Size_bar, Price_bar,Size_Price_bar, Size_Squared_bar)


# In[ ]:


m = ((Size_bar * Price_bar) - (Size_Price_bar)) / ((Size_bar ** 2 ) - (Size_Squared_bar))
m


# In[ ]:


c = Price_bar - m * Size_bar
c


# In[ ]:


def y(m,x,c):
    return m * x + c
print(y(m, 17, c))
print(y(m,16,c))


# ### Find  R-Squared Value ($R^2$) 
# * The Equation of the $R^2$ Value is:

# In[ ]:


R_value = []
for i in range(df.shape[0]):
    R_value.append(y(m,df.Size[i], c))
print(R_value)
df['R_value'] = R_value


# In[ ]:


def R_Squared_Value(y, y_bar, y_cap):
    upper = []
    lower = []
    for i in range(y_cap.shape[0]):
        upper.append((y_cap[i] - y_bar) ** 2)
        lower.append((y[i] - y_bar) ** 2)
    return np.sum(upper) / np.sum(lower), upper, lower
R_square = R_Squared_Value(df.Price, Price_bar, df.R_value)
print(r"R^2 Value is {0:.2f}% and the Upper value is {1} and Lower Value is {2}".format(R_square[0], R_square[1],R_square[2]))
#print(R_Squared_Value(df.Price, Price_bar, df.R_value)[0])


# ### Practice 
# GPA(x) = [3.26,2.60,3.35,2.86,3.82,2.21,3.47,3.28,2.54,3.25] <br>
# Observerd Salary(y) = [33.8,29.8,33.5,30.4,36.4,27.6,35.3,35.0,26.5,33.8]<br>
# Estimated Salary($\hat{y}$) = [33.5, 29.2, 34.1,30.9,37.2,26.6,34.9,33.6,28.8,33.4]
#         
# 1. Make a Dataframe by using above data and add a row at the bottom of the dataframe which contain Mean of each column.
# 2. Estimate the least squares prediction equation of y on x.
# 3. Find the point prediction of starting salary corresponding to each of the GPAs 2.75  and 3.75.
# 4. Compare the observed and estimated salary graphiccally.
# 5. Find $R^2$ Value and Write it by using latex/markdown.
# 

# In[ ]:




