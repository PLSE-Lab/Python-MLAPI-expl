#!/usr/bin/env python
# coding: utf-8

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

import matplotlib.pyplot as plt

import os
import math
import decimal

data=pd.read_csv('../input/countries of the world.csv')
Worlddata=pd.DataFrame(data)
Worlddata.head(10)

# Any results you write to the current directory are saved as output.


# In[ ]:


Worlddata.info()


# In[ ]:


Worlddata.describe()


# In[ ]:


sorts=Worlddata.sort_values(['GDP ($ per capita)'], ascending=False)
sorts.head(10)


# In[ ]:


plt.figure(figure_size=(14,6)


# In[ ]:


plt.figure(figsize=(20,10))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns  
sns.barplot(x='Country',y='GDP ($ per capita)',data=Worlddata[:10])


# In[ ]:


Worlddata.info()


# In[ ]:


Worlddata['Density'] = Worlddata['Pop. Density (per sq. mi.)']
Worlddata['coastline'] = Worlddata['Coastline (coast/area ratio)'] 
Worlddata['migration'] = Worlddata['Net migration']
Worlddata['infant_mortality'] = Worlddata['Infant mortality (per 1000 births)']
Worlddata['literacy'] = Worlddata['Literacy (%)']
Worlddata['phones'] = Worlddata['Phones (per 1000)']
Worlddata['arable'] = Worlddata['Arable (%)']
Worlddata['crops'] = Worlddata['Crops (%)']
Worlddata['other'] = Worlddata['Other (%)']


# In[ ]:


Worlddata.country = Worlddata.Country.astype('category')
Worlddata.region = Worlddata.Region.astype('category')
Worlddata.density = Worlddata.Density.str.replace(",",".").astype(float)
Worlddata.coastline = Worlddata.coastline.str.replace(",",".").astype(float)
Worlddata.migration = Worlddata.migration.str.replace(",",".").astype(float)
Worlddata.infant_mortality = Worlddata.infant_mortality.str.replace(",",".").astype(float)
Worlddata.literacy = Worlddata.literacy.str.replace(",",".").astype(float)
Worlddata.phones = Worlddata.phones.str.replace(",",".").astype(float)
Worlddata.arable = Worlddata.arable.str.replace(",",".").astype(float)
Worlddata.crops = Worlddata.crops.str.replace(",",".").astype(float)
Worlddata.other = Worlddata.other.str.replace(",",".").astype(float)
Worlddata.climate = Worlddata.Climate.str.replace(",",".").astype(float)
Worlddata.birthrate = Worlddata.Birthrate.str.replace(",",".").astype(float)
Worlddata.deathrate = Worlddata.Deathrate.str.replace(",",".").astype(float)
Worlddata.agriculture = Worlddata.Agriculture.str.replace(",",".").astype(float)
Worlddata.industry = Worlddata.Industry.str.replace(",",".").astype(float)
Worlddata.service = Worlddata.Service.str.replace(",",".").astype(float)


# In[ ]:


Worlddata.head()


# In[ ]:


Worlddata.info()


# In[ ]:


x = Worlddata.loc[:,["Region","GDP ($ per capita)","infant_mortality","Birthrate","phones","literacy","Service"]]
sns.pairplot(x,hue="Region",palette="inferno")


# In[ ]:


Worlddata.plot(kind='scatter',x='infant_mortality'.y='')


# In[ ]:




