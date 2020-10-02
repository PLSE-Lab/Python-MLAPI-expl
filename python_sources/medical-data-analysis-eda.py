#!/usr/bin/env python
# coding: utf-8

# # Medical data EDA

# #### Load all the essential libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# ### Load the data set

# In[ ]:


df = pd.read_csv("../input/data.csv")
df.head(5)
df.drop(["id","zipcode"],axis=1,inplace=True)
df.head(1)


# ### Findout if any Null values are present in any of the column, if FALSE then no null value(s)

# In[ ]:


df.isnull().any()


# ### Get the count of married male/femal

# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(df.gender,palette="RdBu",hue=df.marital_status)


# ### Get the count of persons for each disease type

# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(df.disease)
plt.xticks(rotation="90")


# ### Get the estimates and confidence of each disease type

# In[ ]:


plt.figure(figsize=(16,4))
sns.barplot(df.disease,df.available_vehicles)
plt.xticks(rotation="90")


# ### Get the count of male/female of each disease type

# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(df.gender,palette="husl",hue=df.disease)


# ### Get the count of Employment status Vs disease type

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.employment_status,hue=df.disease,palette="rainbow")


# ### Lets find out most common disease type for Students category

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.employment_status[df.employment_status=="student"],hue=df.disease)


# ### Get the count of various education levels

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.education)


# ### Get the count of education Vs employment_status

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.education,hue=df.employment_status)


# ### Get the count of ancestory for each country

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.ancestry)


# ### Get the count of each disease type

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.disease)
plt.xticks(rotation=90)


# ### Number of persons are more in Ireland country, so lets find out disease counts for Ireland

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.ancestry[df.ancestry=="Ireland"],hue=df.disease)


# ### Get the count disease type for each country 

# In[ ]:


# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.ancestry,hue=df.disease)


# ### Get the distribution plot (histogram) of average commute time

# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(df.avg_commute)


# ### Find out the disease count for those who wok or dont work in Military

# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(df.military_service,hue=df.disease)


# ### Get the distribution plot (histogram) of average internet usage

# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(df.daily_internet_use)


# ### Prepare a pair plot and derive some meaningful info, using KDE plot in diagonal

# In[ ]:


plt.figure(figsize=(16,4))
sns.pairplot(df,hue="disease",palette="coolwarm")


# ### Prepare a pair plot and derive some meaningful info, using histogram plot in diagonal, using disease category

# In[ ]:


plt.figure(figsize=(16,4))
sns.pairplot(df,hue="disease",palette="winter_r",diag_kind="hist")


# ### Prepare a pair plot and derive some meaningful info, using histogram plot in diagonal, using marital status category

# In[ ]:


plt.figure(figsize=(16,4))
sns.pairplot(df,hue="marital_status",palette="husl",diag_kind="hist",markers=["D","*"])


# ### Prepare a pair plot and derive some meaningful info, using histogram plot in diagonal, using ediucation category

# In[ ]:


plt.figure(figsize=(16,4))
sns.pairplot(df,hue="education",palette="gist_earth_r",diag_kind="hist",markers=["D","*","^","<",">","."])


# ### Prepare a pair plot and derive some meaningful info, using histogram plot in diagonal, using Gender category

# In[ ]:


plt.figure(figsize=(16,4))
sns.pairplot(df,hue="gender",palette="cubehelix",diag_kind="hist",markers=[">","."])


# ### Draw a KDE plot for available_vehicales Vs daily internet usage

# In[ ]:


plt.figure(figsize=(16,4))
sns.kdeplot(df.available_vehicles,df.daily_internet_use,cbar=True)


# ### Get the count of Male/Femal for each country

# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(df.gender,hue=df.ancestry)


# ## Average time spent on internet by Male and Female (using Bar graph and point plot)

# ### Using Bar graph

# In[ ]:


net_m = df[df.gender=="male"].daily_internet_use.mean()
net_f = df[df.gender=="female"].daily_internet_use.mean()
print(net_m,net_f)
plt.figure(figsize=(16,4))
sns.barplot(["male","female"],[net_m,net_f])


# ### Using point plot

# In[ ]:


plt.figure(figsize=(16,4))
sns.pointplot(df.gender,df.daily_internet_use,estimator=np.mean,markers=["*","<"],color="r",linestyles="--")


# ### Using Line plot

# In[ ]:


plt.figure(figsize=(16,4))
sns.lineplot(df.gender,df.daily_internet_use)


# In[ ]:


df.head(1)

