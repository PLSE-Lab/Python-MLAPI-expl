#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd 

ds=pd.read_csv("../input/youtube-new/USvideos.csv")


# In[ ]:


#Actividad 1
ds.tail(10)






# In[ ]:


#Actividad 2
ds.loc[ds.channel_title == 'The Deal Guy']


# In[ ]:


#Actividad3
ds.iloc[5000]


# In[ ]:


#Actividad4
ds.loc[ds.likes>= 5000000]


# In[ ]:


#Actividad5
sum(ds.likes[ds.channel_title == 'iHasCupquake'])


# In[ ]:


ds.plot.hist(ds.likes[ds.channel_title == 'iHasCupquake'])


ds.plot.hist(ds.likes[ds.channel_title == 'iHasCupquake'])
ds.plot(kind='trending_date',legend='likes')
ds.plot.hist(ds.likes[ds.channel_title == 'iHasCupquake'] ,(ds.likes[trending_date])

ds.plot.hist(ds.likes[ds.channel_title == 'iHasCupquake'] ,(ds.likes[trending_date])
             

