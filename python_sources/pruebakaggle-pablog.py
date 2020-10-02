#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

ds = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")


# In[ ]:


#Actividad 1:
ds.tail(10)


# In[ ]:


#Actividad 2:
ds.loc[ds.title == 'Top 10 Black Friday 2017 Tech Deals']


# In[ ]:


#Actividad 3:
ds.iloc[5000]


# In[ ]:


#Actividad 4:
ds.loc[ds.likes >= 5000000]


# In[ ]:


#Actividad 5:
#ds.loc[ds.channel_title == 'iHasCupquake']
sum(ds.likes[ds.channel_title == 'iHasCupquake'])


# In[ ]:


#Actividad 6:
import matplotlib.pyplot as plt
import pandas as pd

ax=plt.gca()

ds.loc[ds.channel_title == 'iHasCupquake'].plot(kind='bar', x='trending_date', y='likes', ax=ax)
plt.show()


# In[ ]:


#Actividad Extra 1:

#Objetivo: Ver toda la informacion de la celda 'Description' usando los ID de los videos
ds_describe = ds.pd.read_csv([c for c in ds_select1]).describe().toPandas().transpose()

