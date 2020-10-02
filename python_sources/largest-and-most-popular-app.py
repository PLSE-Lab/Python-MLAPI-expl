#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler


# In[ ]:


df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


df.head(60)


# In[ ]:


df.shape


# # Removing duplicats
# 

# In[ ]:


df.drop_duplicates(keep = False, inplace = True)


# In[ ]:


df.shape


# In[ ]:


df.info() 


# # Dropping price column as 93% of the data is 0 which correlated with Free in type column
# 

# In[ ]:


df.drop(['Price'],axis=1,inplace=True) 


# # changing object to numberic value 

# In[ ]:



df["Installs"] = df["Installs"].replace({"\W+":""},regex=True).replace({"\D+":""},regex=True).replace({"\s+":""},regex=True).replace("",np.nan).astype("float64") 
df["Size"] = df["Size"].replace({"[^\d\.']":""},regex=True).replace("",np.nan).astype("float32")*df["Size"].replace({"[^mMkK']":""},regex=True).replace({"M":1000000,"k":1000,"\s+":""}).replace("",1).astype("float64")
df["Reviews"] = df["Reviews"].replace({"[^\d\.']":""},regex=True).replace("",np.nan).astype("float32")*df["Reviews"].replace({"[^mMkK']":""},regex=True).replace({"M":1000000,"k":1000,"\s+":""}).replace("",1).astype("float64")


# In[ ]:


df.head(60)


# # % of empty value in each column

# In[ ]:


for k in df.columns:
    print(k,round((df[k].isnull().sum()/df[k].count()*100),3))


# In[ ]:


# Here we should have replaced the size and rating of NA values from mean & mode repectively.
#but here we are not replacing since we need to find largest size 

df.Size.fillna(df.Size.mean(),inplace=True)
df.Rating.fillna(df.Rating.mode()[0],inplace=True)


# In[ ]:


#removing rest of the nan data row 
for k in df.columns:
    df.dropna(subset = [k], inplace=True)
    


# In[ ]:


for k in df.columns:
    print(k,round((df[k].isnull().sum()/df[k].count()*100),3))


# In[ ]:


df.select_dtypes(include=['float64']).columns


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# # Removing outliers (Do this part only if you are feeding data to ML model)
# # for most popular category and others do not remove outliers

# In[ ]:


# #removing outliers 
# lower_bound = 0.1
# upper_bound = 0.95
# # there are lower bound and upper bound vlaues of the percentile 
# res = df.Rating.quantile([0.1,0.95])


# In[ ]:


# df1 = df[(df.Rating > res.values[0]) & (df.Rating < res.values[1])] #and df.Rating < res.values[1]


# In[ ]:


# df1.shape


# # Most Popular Category  (one with maximum installs)

# In[ ]:


df1 = df.groupby(['Category']).sum().sort_values(by="Installs",ascending=False)[:10]
df1
# game has the maximum installs 


# ## App with the largest size

# In[ ]:


df.sort_values(by="Size",ascending=False)[:20]
# here we see top size apps 


# In[ ]:


print(df.Size[df.App == "Stickman Legends: Shadow Wars"])

