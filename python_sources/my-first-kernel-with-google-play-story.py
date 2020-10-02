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


data = pd.read_csv('../input/googleplaystore.csv')


# # First examine to data

# In[ ]:


data.info()


# ## After that clean the data from object
# - Remove the '+'   and   ',' from Installs columns for convert numeric
# - After that remove the other objects from numeric columns for convert numeric
# 

# In[ ]:


data.Installs=data.Installs.apply(lambda x: x.replace('+','') if '+' in str(x) else x)
data.Installs=data.Installs.apply(lambda x: x.replace(',','') if ',' in str(x) else x)
data.Installs=data.Installs.replace('Free',np.nan)
data.Installs=pd.to_numeric(data.Installs)



# In[ ]:


data.Reviews = data.Reviews.apply(lambda x: x.replace('M','') if 'M' in str(x) else x)
data.Reviews = data.Reviews.apply(lambda x: x.strip('.') if '.' in str(x) else x)
data.Reviews = pd.to_numeric(data.Reviews)


# In[ ]:


data.Size = data.Size.apply(lambda x: x.strip('M') if 'M' in str(x) else x)
data.Size = data.Size.apply(lambda x: x.strip('Varies with device') if 'Varies with device' in str(x) else x)
data.Size = data.Size.apply(lambda x: x.strip('k') if 'k' in str(x) else x)
data.Size = data.Size.apply(lambda x: x.strip('+') if '+' in str(x) else x)
data.Size = data.Size.apply(lambda x: x.replace(',','') if ',' in str(x) else x)
data.Size = pd.to_numeric(data.Size)


# In[ ]:


data.Price = data.Price.apply(lambda x: x.strip('Prise') if 'Prise' in str(x) else x)
data.Price = data.Price.apply(lambda x: x.strip('$') if '$' in str(x) else x)
data.Price = data.Price.apply(lambda x: x.strip('Everyone') if 'Everyone' in str(x) else x)
data.Price = pd.to_numeric(data.Price)


# In[ ]:



data.plot()
plt.show()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot = True,linewidths = .5, fmt = '.3f',ax=ax)
plt.show()


# In[ ]:


data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in data.columns]


# In[ ]:


data.Content_Rating.unique()      


# In[ ]:


Everyone = data[data.Content_Rating == "Everyone"]
Teen =  data[data.Content_Rating == "Teen"]
Everyone_10 = data[data.Content_Rating == "Everyone 10+"]
Mature_17  =  data[data.Content_Rating == "Mature 17+"]
Adults_only_18 = data[data.Content_Rating == "Adults only 18+"]
Unrated =  data[data.Content_Rating == "Unrated"]

Everyone1 = float(len(Everyone))
Teen1 =  float(len(Teen))
Everyone_101 =float(len(Everyone_10))
Mature_171  =  float(len(Mature_17))
Adults_only_181 = float(len(Adults_only_18))
Unrated1 =  float(len(Unrated))

x = np.array(['Everyone','Teen','Everyone_10','Mature_17','Adults_only_18','Unrated'])
y = np.array([Everyone1,Teen1,Everyone_101,Mature_171,Adults_only_181,Unrated1])


# In[ ]:


plt.bar(x,y)
plt.xticks(x,x,rotation=90)
plt.show()


# In[ ]:


data.Installs.plot(kind = 'line',color = 'b',label = 'Installs',linewidth = 1,alpha = .5,grid = True,linestyle = '-')
plt.legend(loc='upper right')

plt.show()


# ## Copying the data  because of dont change data
# - Then sorting Installs and Apps 

# In[ ]:


dataexam = data.copy()


# In[ ]:


dataexam = dataexam.sort_values(['Installs','App'],ascending = False)


# In[ ]:


dataexam.head()


# In[3]:


most_categories = data.Category.value_counts().sort_values(ascending = False)


# In[ ]:


label = np.array(most_categories.index)
value = np.array(most_categories.values)


# In[ ]:


plt.subplots(figsize=(18, 18))
plt.bar(label,value,align = 'center',width= 0.8)
plt.xticks(label,label,rotation = 90)
plt.show()


# In[ ]:


dataexam1 = data.loc[:,['Installs' , 'Rating']]
dataexam1.plot()


# In[ ]:


data.App[dataexam.Category =='FAMILY']


# In[ ]:




