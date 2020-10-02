#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


os.getcwd()


# In[ ]:


os.chdir("/kaggle/input/india-trade-data")


# In[ ]:


imported = pd.read_csv("2018-2010_import.csv")


# In[ ]:


imported.head()


# In[ ]:


exp = pd.read_csv("2018-2010_export.csv")


# In[ ]:


exp.head()


# In[ ]:


imported.shape
exp.shape


# In[ ]:


group = imported.groupby('country').sum()[['value']]


# In[ ]:


# gives the countries from which we import at most
group.sort_values(['value'],axis=0,ascending=False)


# In[ ]:


expgroup = exp.groupby('country').sum()[['value']]


# In[ ]:


# countries in which we export at most
expgroup.sort_values('value',axis = 0,ascending = False)


# In[ ]:


#from above we can see that in which country how much we import and export & also see top import and export countries,


# In[ ]:


# now we see that which sector import and export in other countrise
impsector = imported.groupby('country').count()[['Commodity']]


# In[ ]:


print(impsector)


# In[ ]:


impsector1 = imported.groupby('Commodity').count()[['country']]


# In[ ]:


print(impsector1)


# In[ ]:



impsector2 = imported.groupby('Commodity').sum()[['value']]


# In[ ]:


print(impsector2)


# In[ ]:


# shows highest imported sectors
impsector2.sort_values('value',axis = 0,ascending = False)


# In[ ]:


imp1 = imported.groupby(['year','country']).sum()['value']


# In[ ]:


print(imp1)


# In[ ]:


imp1.shape


# In[ ]:


imp1.head()


# In[ ]:


# each sector's share in importing 
imp2 = pd.pivot_table(imported,'value',['country','year','Commodity'])


# In[ ]:


print(imp2)


# In[ ]:


imp3 = imported.sort_values('value',ascending = False,axis = 0)


# In[ ]:


imp3.head()


# In[ ]:


pd.pivot_table(imp3,'value',['country','year','Commodity'])


# In[ ]:


pd.pivot_table(imported,'value',['year','country'])


# In[ ]:


all_import = imported.set_index(['year','country','value']).sort_index()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# graph shows the highest import in year
plt.plot(imported['year'],imported['value'])
plt.xlabel("year") 
plt.ylabel("value") 
plt.show


# In[ ]:


pd.pivot_table(exp,'value',['year','country'])


# In[ ]:


pd.pivot_table(imported,'value',['year','Commodity','country'])


# In[ ]:


exp1 = exp.groupby('Commodity').sum()[['value']]


# In[ ]:


print(exp1)


# In[ ]:


# in which we export at most
exp1.sort_values('value',ascending = False,axis = 0)


# In[ ]:


# graph shows in which sector we export at most in particular year
plt.plot(exp['year'],exp['value'])
plt.xlabel('year')
plt.ylabel('value')
plt.show()

