#!/usr/bin/env python
# coding: utf-8

# Dear Kaggle community, 
# 
# After having had a course that aim to vulgarize qualitatively the main global financial crises, I wanted to know if I could see these events differently, from an analytical point of view. (The engineer in me comes to the surface) Hence, I decided to go through this [dataset](https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspxs).
# 

# ![africa%20economy.jpg](attachment:africa%20economy.jpg)

# The datset in question describes the economic and financial situation of **70 countries**, mainly in **Africa**, and covers the period from **1800 to 2016** for each of them. The latter is described by variables such as: the country's exchange rate against the US dollar, inflation rates, Gold Standard and others that indicate whether a financial crisis has occurred and of what type.

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


df = pd.read_excel("/kaggle/input/globalcrisisdata/20160923_global_crisis_data.xlsx")
df = df.drop([0],axis = 0 )


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


print ("min :",df.Year.min())
print("max :",df.Year.max())


# If economists in particular or people with a solid knowledge of financial economics in general come across this kernel, I really expect you to read it and give me ideas, analysis that I can do, either using only the data provided in Kaggle or open data and that could help me to find insights and understand financial crises from an analytical point of view.
