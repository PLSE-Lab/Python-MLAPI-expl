#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  #Seaborn module
import matplotlib.pyplot as plt
from collections import Counter
from math import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/who_suicide_statistics.csv")


# In[ ]:


data.info()


# In[ ]:


data.head(15) #all people


# 

# In[ ]:


female = sum((data["sex"] == "female")) #Just female
female


# In[ ]:


male = sum((data["sex"] == "male")) #Just male
male 


# In[ ]:


Counter(data["age"]).keys()
    


# In[ ]:


Counter(data["age"]).values()


# In[ ]:


year_key = list()
year_values = list()

for i in Counter(data["year"]).keys():
    year_key.append(i)
    
for i in Counter(data["year"]).values():
    year_values.append(i)

plt.scatter(year_key,year_values)
plt.title("Suicide increase by years")
plt.xlabel("Year")
plt.ylabel("Suicides")
plt.grid(alpha = 0.45)
plt.show()


# In[ ]:



sns.set_style('darkgrid')
plt.xlabel("Suicide")
plt.ylabel("Frequency")
sns.distplot(year_values)


# In[ ]:


sns.set_style('darkgrid')
plt.xlabel("Year")
plt.ylabel("Frequency")
sns.distplot(year_key)

