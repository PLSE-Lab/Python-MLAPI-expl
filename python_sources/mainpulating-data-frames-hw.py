#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/pokemon.csv")


# In[ ]:


data.columns


# In[ ]:


data.head()
# data.tail()


# **indexing & slicing**

# In[ ]:


data.index = range(1,802,1) # index starts from 1
data.name[0:3] # however, zero is the index that's exclusive. show data 1st to 3rd index.


# In[ ]:


data["type1"][5] # 5th index of 'type1'


# In[ ]:


data.loc[1,"type1":"type2"]  # 1st raw of type1 to type2


# In[ ]:


data[["name","speed"]] # show name and speed throughout the list


#  ** filtering data frames**
# 

# In[ ]:


filtered = data.speed > 150


# In[ ]:


data[filtered]


# In[ ]:


filter1 = data.attack > 150
filter2 = data["defense"] > 120
data[filter1 & filter2]


# In[ ]:


data.speed[data.attack>180] # speed of pokemons of the ones having greater attack than 180.


# **transforming data**

# In[ ]:


def prod(n):
    return n**2
data["attack"].apply(prod) # squares of attack column 


# In[ ]:


data["full speed"] = data.sp_attack + data.sp_defense 
data.head()


# In[ ]:


data = pd.read_csv("../input/pokemon.csv")


# In[ ]:


print(data.index.name)  # we dont have an index name yet.
data.index.name = ["benimsin"] # name of my index is " benimsin"


# In[ ]:


data1 = data.set_index("type1")  # setting another column as index
# data.index.name = data["smth"]
data1.head()


# **hierarchical indexing **

# In[ ]:


data1 = data.set_index(["type1","type2"])
data1.head()


# **pivoting data**

# In[ ]:


dic= {"sex":["F","M","M","F"],"sizes":[13,17,12,19],"smt":["x","x","y","y"]}
df = pd.DataFrame(dic)
df.index = range(1,8,2)  # index counts two each
df


# In[ ]:


df.pivot(index = "sex",columns= "smt",values="sizes")


# **Stacking & Unstacking Data Frames**

# In[ ]:


df2 = df.set_index(["smt","sex"])
df2


# In[ ]:


df2.unstack(level=0)    # remove first index column


# In[ ]:


df3 = df2.swaplevel(0,1)
df3


# **Melting Data**

# In[ ]:


pd.melt(df,id_vars="sex",value_vars=["smt","sizes"])


#  **Category & Groupby**

# In[ ]:


df.groupby("sex").mean()


# In[ ]:


df.groupby("smt").sizes.max()

