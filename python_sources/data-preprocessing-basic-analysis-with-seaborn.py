#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libralies
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


marvel = pd.read_csv("../input/marvel-wikia-data.csv")
marvel.shape


# In[ ]:


#See the loss of data in each colmns
print(marvel.shape)
print("----------------------")
print(marvel.isnull().any())
print("----------------------")
print(marvel.isnull().sum())


# In[ ]:


#drop useless columns
marvel = marvel.drop(["GSM"],axis=1)


# In[ ]:


# fill loss cells
marvel.ID = marvel.ID.fillna("unknown")
marvel.ALIGN = marvel.ALIGN.fillna("unknown")
marvel.EYE = marvel.EYE.fillna("unknown")
marvel.HAIR = marvel.HAIR.fillna("unknown")
marvel.SEX = marvel.SEX.fillna("unknown")
marvel.ALIVE = marvel.ALIVE.fillna("unknown")

#change name of column
marvel = marvel.rename(columns= {
    "FIRST APPEARANCE" : "FIRST_APPEARANCE"
    })


# In[ ]:


#See the loss of data in each colmns
print(marvel.shape)
print("----------------------")
print(marvel.isnull().any())
print("----------------------")
print(marvel.isnull().sum())


# In[ ]:


#remove rows which has "Nan"
marvel = marvel.dropna()

marvel.shape


# In[ ]:


plt.style.use('ggplot') 


# In[ ]:



#count data of SEX
plt.figure(figsize=(15, 5))
sns.countplot(x=marvel.SEX,
              data=marvel,
              palette="Pastel2" ,
              order=["Male Characters","Female Characters","Agender Characters","Genderfluid CharactersSEX","unknown"])


# In[ ]:


#count data of ["SEX"] on ["Year"]
marvel.groupby("Year")["SEX"].value_counts().unstack().plot(figsize=(15,5))


# In[ ]:


#what about ID
plt.figure(figsize=(15, 5))
sns.countplot(x=marvel.ID, data=marvel, palette="Pastel1") 


# In[ ]:


#count data of ["ID"] on ["Year"]
marvel.groupby("Year")["ID"].value_counts().unstack().plot(figsize=(15,8))


# In[ ]:


#what about EYE
plt.figure(figsize=(15, 8))
sns.countplot(x=marvel.EYE, data=marvel, palette="Pastel1") 


# In[ ]:


#what about EYE without "unknown"
remove_EYE = marvel[marvel.EYE != "unknown"]

plt.figure(figsize=(30, 10))
sns.countplot(x=remove_EYE.EYE, data=remove_EYE, palette="Pastel1")


# In[ ]:


#count data of ["ALIVE"] on ["Year"]
marvel.groupby("Year")["ALIVE"].value_counts().unstack().plot(figsize=(15,10))

