#!/usr/bin/env python
# coding: utf-8

# # Hello everyone!
# * *I have some data for machine learning algoritms. I used this data. It's very simple and easy practice for me, about linear regression.*
# * *I create this data on excel.*
# # *Sorry for my bad english.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/energyWork_MachineL.csv",sep = ";")


# In[ ]:


df.head(10)


# In[ ]:


print(plt.style.available)
plt.style.use('ggplot')


# In[ ]:


df.info()


# * *I have values of data. Columns are "energy" and "work".*
# * *With scatterplot i check how location on scatterplot, points of values.*

# In[ ]:


plt.scatter(df.energy,df.work)
plt.xlabel("energy")
plt.ylabel("work")
plt.show()


# In[ ]:


plt.scatter(df.energy,df.work)
plt.xlabel("energy")
plt.ylabel("work")

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
x = df.energy.values.reshape(-1,1)
y = df.work.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

plt.plot(x, y_head, color = "green")

plt.show()


# * *Our result this line.*

# In[ ]:


from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y,y_head))

