#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the nessesary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading datasets
import os
files  = os.listdir("../input")
print(files)

# Any results you write to the current directory are saved as output.


# In[ ]:


#crop state,crop production, crop varity, crop year, produced
crop_state = pd.read_csv("../input/data 1.csv")
crop_produce = pd.read_csv("../input/data 2.csv")
crop_varity = pd.read_csv("../input/data 3.csv")
crop_year = pd.read_csv("../input/data 4.csv")
produced = pd.read_csv("../input/produced.csv")


# In[ ]:


crop_state.head()


# In[ ]:


crop_produce.head()


# In[ ]:


crop_varity.head()


# In[ ]:


crop_year.head(12)


# In[ ]:


produced.head()


# ### So till now we know about all the data sets 
# ### Now we will explore and plot each data sets and analyse them

# ### 1.Working on crop_produce

# In[ ]:


crop_produce.columns


# In[ ]:


k = crop_produce[['Crop             ', 'Production 2006-07', 'Production 2007-08',
       'Production 2008-09', 'Production 2009-10', 'Production 2010-11']].groupby('Crop             ')
index = list(k.indices.keys())
index[:]


# In[ ]:


# So it is clear that for taking only the total part 
index[-8:-2]


# In[ ]:


# Now plotting the Year wise production of agricultural crop
k.sum()[:-9].plot(figsize=(20,12), kind='bar');
# -9 for eliminating all the total part
plt.title('Year wise production of agricultural crop')
plt.ylabel('Production in Quintal');


# ### Here we can clearly see the year wise production from 2006 to 2011 of different crops 

# In[ ]:


#k.mean().plot(figsize=(20,10), kind='bar');
#plt.figure(figsize=(12,6))


# In[ ]:


l = len(k['Crop             '])
# l for enumerating throgh all crops
fig, arraxes = plt.subplots(1,4, figsize=(12,12), sharey=True)
plt.setp(arraxes, yticks=range(len(index)), yticklabels = index)

for axes, p in zip(arraxes.flat,['Production 2006-07', 'Production 2007-08','Production 2008-09', 'Production 2009-10', 'Production 2010-11']):
    axes.barh(range(l), k[p].head())
    axes.set_title(p)
#     axes.tick_params(axis='x',  rotation=90)
    axes.set_xlabel("production in Quantal")
fig.set_figwidth(20)


# In[ ]:


kc = crop_produce[['Crop             ','Area 2006-07', 'Area 2007-08', 'Area 2008-09', 'Area 2009-10',
       'Area 2010-11']].groupby('Crop             ')
kc.sum().plot(figsize=(20,18), kind='barh', stacked= True);


# In[ ]:


k.head()['Crop             '].values


# ### Working on crop_state

# In[ ]:


crop_state.groupby('State').sum()


# ### so here is the information about the crop produced , agricultural areas,cost of production and Yeild of the states

# In[ ]:


cols = crop_state.columns


# In[ ]:


crop_state.groupby('Crop')[cols[:-1]].sum().plot(kind='bar', figsize=(12,6));


# In[ ]:


crop_state.groupby('State')[cols[5:6]].sum().plot(kind='bar', figsize=(12,6));


# In[ ]:


crop_state.groupby('State')[cols[:-1]].sum().plot(kind='bar', figsize=(12,6));


# In[ ]:


crop_state.head()


# In[ ]:


crop_state.groupby('State')[cols[-1]].sum().plot(kind='pie', figsize=(14,14));


# ### for crop_state

# In[ ]:


plt.title('Crop-wise '+cols[-1], color='red', fontsize=20)
crop_state.groupby('Crop')[cols[-1]].sum().plot(kind='pie', figsize=(20,20));


# In[ ]:


crop_varity.head()


# In[ ]:


plt.title('No of varieties per crop in India', fontsize=40, color='orange')
crop_varity['Crop'].value_counts().plot(kind='bar', figsize= (12,12));


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




