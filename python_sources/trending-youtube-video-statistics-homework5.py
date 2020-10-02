#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # MANIPULATING DATA FRAMES WITH PANDAS

# * INDEXING DATA FRAMES

# In[ ]:


#read data
data=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
data.head()


# In[ ]:


# indexing using square brackets
data["likes"][0]


# In[ ]:


#using column attribute and row label
data.likes[0]


# In[ ]:


#using loc accessor
data.loc[1,["likes"]]


# In[ ]:


#selecting only some columns
data[["likes","dislikes"]].head()


# **SLICING DATA FRAME**

# In[ ]:


#slicing and indexing series
data.loc[1:10,"likes":"comment_count"]


# In[ ]:


#reverse slicing
data.loc[10:1:-1,"likes":"comment_count"]


# In[ ]:


#from something to end
data.loc[1:10,"ratings_disabled":]


# **FILTERING DATA FRAMES**

# In[ ]:


#creating boolean series
boolean=data.likes>500000
data[boolean]


# In[ ]:


#combining filters
data[(data.likes>5000000) &(data.views>500000)]


# In[ ]:


#filtering column based others
data.likes[data.views>500000]


# **TRANSFORMING DATA**

# In[ ]:


#plain python function 
def div(n):
    return n/2
data.likes.apply(div)


# In[ ]:


data["new_column"]=data.likes+data.views
data.head()


# **INDEX OBJECTS AND LABELED DATA**

# In[ ]:


#our index name is this:
print(data.index.name)
#lets change it
data.index.name="index name"
data.head()


# **HIERARCHICAL INDEXING**

# In[ ]:


data=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
data.head()


# In[ ]:


data1=data.set_index(["likes","dislikes"])
data1.head()


# **PIVOTING DATA FRAMES**

# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:


#pivoting
df.pivot(index="treatment",columns="gender",values="response")


# **STACKING and UNSTACKING DATAFRAME**

# In[ ]:


df1=df.set_index(["treatment","gender"])
df1


# In[ ]:


#level determines indexes
df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# **MELTING DATA FRAME**

# In[ ]:


df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# **CATEGORICALS AND GROUPBY**

# In[ ]:


df


# In[ ]:


df.groupby("treatment").mean()


# In[ ]:


df.groupby("treatment").age.max()


# In[ ]:


df.groupby("treatment")[["age","response"]].min()

