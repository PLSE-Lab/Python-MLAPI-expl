#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the required modules
import numpy as np 
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/energy-usage-2010.csv")
df.head(10)


# In[ ]:


df.info()


# In[ ]:


#Counting the number of occurrence of each Area
df['COMMUNITY AREA NAME'].value_counts()


# In[ ]:


#Total enrgy consumption of each  Area
sdf=df.groupby(['COMMUNITY AREA NAME']).sum()
sdf.head(10)


# In[ ]:


#Area which comsumes more THERMS energy
sdf.nlargest(10, 'TOTAL THERMS')
lig=sdf.nlargest(10, 'TOTAL THERMS').index
print(lig)
dfl=df[df['COMMUNITY AREA NAME'].isin(lig)]


# In[ ]:


#Area which comsumes more energy
sdf.nlargest(10, 'TOTAL KWH')
lig=sdf.nlargest(10, 'TOTAL KWH').index
print(lig)
dfl=df[df['COMMUNITY AREA NAME'].isin(lig)]


# Here  we see that 8  Areas are common to both the result for maximum consumtipon

# In[ ]:


#Area which comsumes less THERMS energy
sdf.nsmallest(10, 'TOTAL THERMS').iloc[:10]
lis=sdf.nsmallest(10, 'TOTAL THERMS').index
print(lis)
dfs=df[df['COMMUNITY AREA NAME'].isin(lis)]


# In[ ]:


#Area which comsumes less energy
sdf.nsmallest(10, 'TOTAL KWH').iloc[:10]
lis=sdf.nsmallest(10, 'TOTAL KWH').index
print(lis)
dfs=df[df['COMMUNITY AREA NAME'].isin(lis)]


# Here also we see that 7 out of 10  Areas are common thus we'll focus only analysis of one consumption 

# In[ ]:


fig, axs = plt.subplots(nrows=2,figsize=(20,10))
sns.countplot(x='BUILDING TYPE',data=dfs,hue='COMMUNITY AREA NAME', ax=axs[0])
axs[0].legend(loc='upper right')
axs[0].set_title("Area which consume less power")

sns.countplot(x='BUILDING TYPE',data=dfl,hue='COMMUNITY AREA NAME', ax=axs[1])
axs[1].legend(loc='upper right')
axs[1].set_title("Area which consume more power")


# From the above graphs we see that the Area which consume more power is mainly due to the reason that they have more number of plots than the area which consume less power.
# 
# It is not the population comparison

# In[ ]:


fig, axs = plt.subplots(nrows=2,figsize=(20,15))
sns.barplot(x=dfs['BUILDING TYPE'],y=dfs['TOTAL KWH'],hue=dfs['COMMUNITY AREA NAME'], ax=axs[0], ci=None)
sns.barplot(x='BUILDING TYPE',y='TOTAL KWH',data=dfl,hue='COMMUNITY AREA NAME', ax=axs[1], ci=None)

axs[0].legend(loc='upper right')
axs[0].set_ylim(0,700000)
axs[0].set_title("Area which consume less power")
axs[0].legend(loc=0)

axs[1].legend(loc=0)
axs[1].set_ylim(0,700000)
axs[1].set_title("Area which consume more power")


# From the above graph it seems that though the count for **Residential** category is more as seens above but the more consumtion is been done by **Commercial** plots for both type of Area 

# In[ ]:


fig, axs = plt.subplots(nrows=2,figsize=(20,15))
sns.barplot(x='COMMUNITY AREA NAME',y='TOTAL POPULATION',data=dfs, ax=axs[0], ci=None)
sns.barplot(x='COMMUNITY AREA NAME',y='TOTAL POPULATION',data=dfl, ax=axs[1], ci=None)


axs[0].set_title("Area which consume less power")

axs[1].set_title("Area which consume more power")


# Population play an important role in the region which consume more power apart from these region being more of commercial space as seen above

# In[ ]:


#Getting energy consumption for each month
dft=pd.concat([dfs,dfl])
dft=dft.groupby(['COMMUNITY AREA NAME']).sum()
dft=dft.iloc[:,1:13]
dft


# In[ ]:


plt.figure(figsize=(20,6))
sns.clustermap(dft,cmap='coolwarm')


# A definite conclusion can't be obtained whether Months played a specific role in comsumption or not

# ****If you like this kernel please Appreciate by an UPVOTE .Thank you****

# In[ ]:




