#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/bus-dataset/bus.csv')


# # Displaying the first 5 entries in the DataFrame

# In[ ]:


df.dropna(inplace=True)
df.head(5)


# # Finding Unique Types
# 

# In[ ]:


types = df['Type'].unique()


# In[ ]:


types


# 

# # Seeing the Distribution
# 

# In[ ]:


types_dict = dict()
for item in types:
    types_dict[item]= (df['Type'].str.lower() == item.lower()).sum()


# In[ ]:


types_dict.pop('Single deck')


# In[ ]:


labels,num_data= list(types_dict.keys()),  list(types_dict.values())


# In[ ]:


x = np.arange(len(labels))
plt.bar(x,num_data)
plt.xticks(x,labels,rotation=90,fontSize=18)
plt.title("Type Distribution",fontSize=20)
plt.show()


# In[ ]:


temp = min(types_dict.values()) 
res = [key for key in types_dict if types_dict[key] == temp] 
res


# # Inference
#   - Single Deck bus is manufactured the most School bus.
#   - School bus is manufactured the least.

# # Unique Countries

# In[ ]:


countries = df['Country'].unique()


# In[ ]:


countries


# This is my first notebook do say me ur reviews and correct my mistakes
