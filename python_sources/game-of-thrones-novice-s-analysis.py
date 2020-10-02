#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/character-deaths.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.isnull().any()


# <h4>Considering NaN value means not dead</h4>

# In[ ]:


df["Dead"]=df["Death Year"].map({np.nan:0,np.float:1})
df["Dead"].fillna(1,inplace=True)


# In[ ]:


df


# In[ ]:


df["No of Books"]=df["GoT"]+df["CoK"]+df["SoS"]+df["FfC"]+df["DwD"]
df.drop(labels=["GoT","CoK","SoS","FfC","DwD"],axis=1)


# In[ ]:


df["Allegiances"].unique()


# In[ ]:


def house(x):
    if x in ['None',"Wildling","Night's Watch"]:
        return x
    elif "House" not in x:
        return "House "+str(x)
    else:
        return x


# In[ ]:


df["Allegiances"]=df["Allegiances"].apply(house)
df["Allegiances"].unique()


# ## Death probability on basis of Allegiances

# In[ ]:


df2=df[["Dead","Allegiances"]].groupby(by="Allegiances",as_index=False).mean().sort_values(by="Dead",ascending=True)
df2


# In[ ]:


fig,ax=plt.subplots()
width=0.35
rect=ax.bar(df2["Allegiances"],df2["Dead"],color="#6B0FFF")
plt.xticks(rotation=90);


# ### Gender and Allegiances

# In[ ]:


plt.figure(figsize=(15,5))
sns.set_context(font_scale=2)
sns.violinplot(x='Allegiances',y='Dead',data=df,hue='Gender',split=True)
plt.xticks(rotation=90);


# ### Observations:
#  - Most no of deaths:Wildlings 
#  - Least no of deaths:House Tyrell

# <h2>All men must die? </h2>

# In[ ]:


df2=df[["Dead","Gender"]].groupby(by="Gender",as_index=False).mean().sort_values(by="Dead",ascending=True)
df2


# In[ ]:


fig,ax=plt.subplots()
width=0.5
#rect=ax.hist(df2,bins=20)
rect=ax.bar(df2["Gender"].map({0:"female",1:"male"}),df2["Dead"],color="#DF5BC2")
plt.xticks(rotation=0);


# ### Nobility and Gender

# In[ ]:


sns.barplot(x='Gender',y='Dead',data=df,hue="Nobility",palette="viridis")


# ### Observations:
# - The probability of dying a man in westeros is greater than that of a woman

# ## Nobility

# In[ ]:


df2=df[["Dead","Nobility"]].groupby(by="Nobility",as_index=False).mean().sort_values(by="Dead",ascending=True)
df2


# In[ ]:


fig,ax=plt.subplots()
rect=ax.bar(df2["Nobility"].map({0:"Not Noble",1:"Noble"}),df2["Dead"],color="#123456")


# In[ ]:





# ## You will live only in one book-No of appearances

# In[ ]:


df2=df[["Dead","No of Books"]].groupby(by="No of Books",as_index=False).mean().sort_values(by="Dead",ascending=True)
df2


# In[ ]:


fig,ax=plt.subplots()
rect=ax.bar(df2["No of Books"],df2["Dead"],color="#650056")


# 

# In[ ]:




