#!/usr/bin/env python
# coding: utf-8

# *THIS IS MY FIRST ATTEMPT OF SIMPLE LINEAR REGRESSION USING ONLY NUMPY*asdasfsadfasf
# 
# refrence: https://stackoverflow.com/a/27093747

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv("../input/kc-housesales-data/kc_house_data.csv")

price = df['price']asdfasdf
area = df['sqft_living']

train_price = price[:15000]
train_area = area[:15000]
test_area = area[15000:]
test_price = price[15000:]

matrix1 = np.array([[sum(train_area),-len(train_area)],[-sum((train_area)**2),sum(train_area)]])
matrix2 = np.array([sum(train_price),sum(train_area*train_price)])

result = (((sum(train_area))**2 -len(train_area)*sum((train_area)**2))**-1)* matrix1.dot(matrix2)


intercept,slope = result


# Input data files are available in the "../input/" directory.


# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# Any results you write to the current directory are saved as output.


# In[ ]:


test_price1 = intercept*train_area
test_price1 += slope


plt.scatter(test_area,test_price,color='red')
plt.plot(train_area,test_price1,color='blue')
plt.show()


# In[ ]:


test_price2 = intercept*test_area
test_price2 += slope
plt.scatter(test_area,test_price,color='red')
plt.plot(test_area,test_price2,color='green')
plt.show()

