#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


cars = pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")


# In[ ]:


cars.head()


# In[ ]:


# Test if there are missing values:
cars.isnull().sum()


# There are quite a lot of missing values! Let's see some statistics
# 

# In[ ]:


cars.describe()


# In[ ]:


cars.info()


# In[ ]:


cars.manufacturer.value_counts().plot.bar(figsize = (15,12))


# Most of the cars are from ford!

# In[ ]:


cars.price.plot.box()


# Some cars cost around 4 billions $, let's delete too expensive cars

# In[ ]:


cars[cars.price < 200000].price.plot.box()


# It is already a bit better, but there are still too many outliers. Let's see how it looks like without them

# In[ ]:


cars[cars.price < 50000].price.plot.box()


# Now it is far better, we can see that the median price is around 9000$

# In[ ]:


import seaborn as sns
cars = cars[cars.price < 50000]
import matplotlib.pyplot as plt
plt.figure(figsize = (15,12))
sns.boxplot(data = cars, y = "price", x = "manufacturer")
plt.xticks(rotation=90)
plt.title("Variation of the price by car type")
plt.show()


# In[ ]:




