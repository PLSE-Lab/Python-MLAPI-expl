#!/usr/bin/env python
# coding: utf-8

# ## What is Crude Oil
# 
# Crude oil is a naturally occurring, unrefined petroleum product composed of hydrocarbon deposits and other organic materials. A type of fossil fuel, crude oil can be refined to produce usable products such as gasoline, diesel and various forms of petrochemicals. It is a nonrenewable resource, which means that it can't be replaced naturally at the rate we consume it and is therefore a limited resource.
# 
# Crude oil is typically obtained through drilling, where it is usually found alongside other resources, such as natural gas (which is lighter, and therefore sits above the crude oil) and saline water (which is denser, and sinks below). It is then refined and processed into a variety of forms, such as gasoline, kerosene and asphalt, and sold to consumers. Although it is often called "black gold," crude oil has ranging viscosity and can vary in color from black and yellow depending on its hydrocarbon composition. Distillation, the process by which oil is heated and separated in different components, is the the first stage in refining.
# 
# ## History of Crude Oil Usage
# Although fossil fuels like coal have been harvested in one way or another for centuries, crude oil was first discovered and developed during the Industrial Revolution, and its industrial uses were first developed in the 19th century. Newly invented machines revolutionized the way we do work, and they depended on these resources to run. Today, the world's economy is largely dependent on fossil fuels such as crude oil, and the demand for these resources often spark political unrest, since a small number of countries control the largest reservoirs. Like any industry, supply and demand heavily affects the prices and profitability of crude oil. The United States, Saudi Arabia, and Russia are the leading producers of oil in the world.
# 
# In the late 19th and early 20th centuries, however, the United States was one of the world's leading oil producers, and U.S. companies developed the technology to make oil into useful products like gasoline. During the middle and last decades of the 20th century, however, U.S. oil production fell dramatically, and the U.S. became an energy importer. Its major supplier was the Organization of Petroleum Exporting Countries (OPEC), founded in 1960, which consists of the world's largest (by volume) holders of crude oil and natural gas, reserves. As such, the OPEC nations had a lot of economic leverage in determining supply, and therefore the price, of oil in the late 1900s.
# 
# In the early 21st century, the development of new technology, particularly hydro-fracturing, has created a second U.S. energy boom, largely decreasing the importance and influence of OPEC. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_excel('../input/Crude Oil Prices Daily.xlsx')


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,10));
data['Closing Value'].plot(kind='bar');
plt.ylabel('$ Prices ',fontsize = 18);
plt.title('Variation in Crude Prices over the years', Fontsize = 24);
plt.xticks(color = 'w');


# In[ ]:


plt.figure(figsize = (20,10));
data['Closing Value'].plot(kind='line');
plt.ylabel('$ Prices ',fontsize = 18);
plt.title('Variation in Crude Prices over the years', Fontsize = 24);


# In[ ]:


data['Closing Value'].describe()


# In[ ]:


import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns
plt.figure(figsize = (20,10))
sns.violinplot(data['Closing Value'], color = 'Orange');
plt.xlabel('Closing Value',fontsize = 18);


# In[ ]:




