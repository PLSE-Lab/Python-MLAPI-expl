#!/usr/bin/env python
# coding: utf-8

# **This kernel has been created under favour of DATAI data science for beginners tutorials.**

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


data = pd.read_csv('../input/StudentsPerformance.csv')
data.index = range(1, len(data) + 1)
data.head()


# In[ ]:


data["math score"][2]


# In[ ]:


#data.math score[1] ?


# In[ ]:


data.loc[4,["lunch"]]


# In[ ]:


data[["lunch","math score","parental level of education"]]


# In[ ]:


print(type(data["math score"]))
print(type(data[["reading score"]]))


# In[ ]:


data.loc[1:10,"math score"]


# In[ ]:


boolean = data.lunch == "free/reduced"
data[boolean]


# In[ ]:


first_filter = data.lunch == "free/reduced"
second_filter = data["parental level of education"] == "high school"
data[first_filter & second_filter] 


# In[ ]:


data.lunch[data["math score"]>70]


# In[ ]:


def trans(n):
    if n == "completed":
        return "hardworking"
    else:
        return "lazy"
    
data["test preparation course"].apply(trans)


# In[ ]:


data.head()


# In[ ]:


#data.HP.apply(lambda n : n/2)

data["gender"].apply(lambda n: "female" if n == "male" else "male")


# In[ ]:


# Defining column using other columns
data["mean writing and reading"] = (data["reading score"] + data["writing score"]) / 2


# In[ ]:


data.head()


# In[ ]:


print(data.index.name)
data.index.name = "index_name"
data.head()


# In[ ]:


data3 = data.copy()
data3.index = range(100,1100,1)
data3.head()
#data.info


# In[ ]:


data = pd.read_csv('../input/StudentsPerformance.csv')
data.index = range(1, len(data) + 1)
data.head()


# In[ ]:


data1 = data.set_index(["gender"],["race/ethnicity"])
data1.head()


# In[ ]:


data.head()


# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:


#pivoting
df.pivot(index="treatment",columns = "gender",values="response")


# In[ ]:


df1 = df.set_index(["treatment","gender"])
df1


# In[ ]:


df1.unstack(level=1)


# In[ ]:


df2 = df1.swaplevel(0,1)
df2


# In[ ]:


df


# In[ ]:


# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# In[ ]:


df


# In[ ]:


df.groupby("treatment").mean()


# In[ ]:


df.groupby("treatment").max()


# In[ ]:


df.groupby("treatment").min()


# In[ ]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




