#!/usr/bin/env python
# coding: utf-8

# # USA Cars Visualization
# 1.     [Brand - Price Visualization](#1)
# 2.     [Model - Price Visualization](#2)
# 3.     [Year - Price Visualization](#3)
# 4.     [Color - Price Visualization](#4)
# 5.     [State - Price Visualization](#5)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv")


# In[ ]:


df.drop(labels=["Unnamed: 0"],axis=1,inplace=True)
df.head(10)


# ## Brand - Price Visualization <a id="1"></a>

# In[ ]:


sns.factorplot(x = "brand",y="price",data=df,kind="bar",size=6)
plt.xticks(rotation=90)
plt.show()


# * Harley-Davidson brand has the most expensive cars.
# * And the least expensive cars are Peterblit.

# ## Model - Price Visualization <a id="2"></a>

# In[ ]:


df["model"] = [1 if i == "door" else 2 if i == "f-150" else 3 for i in df["model"]]


# In[ ]:


sns.factorplot(x = "model",y="price",data=df,kind="bar",size=5)
plt.xticks(rotation=90)
plt.show()


# * The majority is door, and the second greatest model is f-150, so we gave them a special class and declared the left ones as 3.
# * f-150's are the most expensive models.

# ## Year - Price Visualization <a id="3"></a>

# In[ ]:


sns.factorplot(x = "year",y="price",data=df,kind="bar")
plt.xticks(rotation=90)
plt.show()


# * Prices rise year by year, but there is an exception.
# * Classic cars have more value than newer cars.

# ## Color - Price Visualization <a id="4"></a>

# In[ ]:


sns.factorplot(x = "color",y="price",data=df,kind="bar",size=7)
plt.xticks(rotation=90)
plt.show()


# * The visualization is too complex, so we will make it easier to read.

# In[ ]:


df["color_class"] = [1 if i == "white" else 2 if i == "black" else 3 for i in df["color"]]
df.head()


# In[ ]:


sns.factorplot(x = "color_class",y="price",data=df,kind="bar",size=7)
plt.xticks(rotation=90)
plt.show()


# * The value of black cars is greater than other colors.

# ## State - Price Visualization <a id="5"></a>

# In[ ]:


sns.factorplot(x = "state",y="price",data=df,kind="bar",size=7)
plt.xticks(rotation=90)
plt.show()


# * Prices are highest in the Kentucky state.
