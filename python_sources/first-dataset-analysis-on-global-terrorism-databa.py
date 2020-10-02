#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[2]:


data=pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding = "ISO-8859-1")


# In[ ]:


data.info()


# In[3]:


new_data=data.head()
new_data


# In[ ]:


f,ax=plt.subplots(figsize=(30,30))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)
plt.show()


# In[ ]:


data.columns


# In[ ]:


data.nkill.plot(kind="line",color="red",label="Fatalities",linewidth=1,grid=True,linestyle="-")
data.nwound.plot(kind="line",color="green",label="Wounds",linewidth=1,grid=True,linestyle="-")
plt.legend(loc="upper right")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Total number of injuries and fatalities ")
plt.show()


# In[ ]:


data.plot(kind="scatter",x="nkill",y="nwound",color="violet")
plt.xlabel("Fatalities")
plt.ylabel("Injuries")
plt.title("Fatalities and Injuries Scatter Plot")
plt.show()


# In[ ]:


data.weaptype1.plot(kind="hist",bins=25,figsize=(11,11),grid=True)
plt.title("Usage of Different Weapons ")
plt.show()


# In[ ]:



data[np.logical_and(data["nkill"]>400,data["nwound"]>100)] #Biggest Terrorist Attacks


# In[ ]:


melted = pd.melt(frame=new_data,id_vars = 'country_txt', value_vars= ['nkill','nwound'])
melted


# In[ ]:


melted.pivot(index = 'country_txt', columns = 'variable',values='value')


# In[ ]:


data.dtypes


# In[ ]:



data.describe()


# In[ ]:





# In[ ]:




