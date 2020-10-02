#!/usr/bin/env python
# coding: utf-8

# The retail,recreation,transit,workplace,residential change graph between Bangladesh and some developed country and New York.
# 
# There some controversy that Bangladeshi are not well maintained the social distance but the google mobility data show that Bangladeshi are well maintained the social distance even better than developed and more social aware country like United States, also Bngladesh and New York got the infection almost similar time where New York has too many confirmed cases and deaths but Bangladeshi and New York people almost maintained same level social distance which are found by Google Mobility Data.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/GMR.csv',parse_dates=True)


# In[ ]:


df.head(5)


# In[ ]:


df2=df.copy()
df3=df2.fillna('x',inplace=True)


# In[ ]:


df3=df2.copy()


# In[ ]:


df3.head(3)


# In[ ]:


bd=df3[df3['country_region']=='Bangladesh']
tai=df3[df3['country_region']=='Taiwan']
uk=df3[df3['country_region']=='United Kingdom']
us=df3[df3['country_region']=='United States']
ita=df3[df3['country_region']=='Italy']
fra=df3[df3['country_region']=='France']
ger=df3[df3['country_region']=='Germany']


# In[ ]:


uk2=uk[uk['sub_region_1']=='x']
us2=us[us['sub_region_1']=='x']
ita2=ita[ita['sub_region_1']=='x']
fra2=fra[fra['sub_region_1']=='x']
ger2=ger[ger['sub_region_1']=='x']


# In[ ]:


tai.shape


# The retail,recreation,transit,workplace,residential change graph between Bangladesh and some developed country.

# In[ ]:


plt.figure(figsize=(30,12))
plt.title('retail and recreation percent change from baseline')
plt.plot(bd['date'],bd['retail_and_recreation_percent_change_from_baseline'])
plt.plot(tai['date'],tai['retail_and_recreation_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['retail_and_recreation_percent_change_from_baseline'])
plt.plot(us2['date'],us2['retail_and_recreation_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['retail_and_recreation_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('retail_and_recreation_percent_change_from_baseline.png')


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('residential percent change from baseline')
plt.plot(bd['date'],bd['residential_percent_change_from_baseline'])
plt.plot(tai['date'],tai['residential_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['residential_percent_change_from_baseline'])
plt.plot(us2['date'],us2['residential_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['residential_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('residential_percent_change_from_baseline.png')


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('workplaces percent change from baseline')
plt.plot(bd['date'],bd['workplaces_percent_change_from_baseline'])
plt.plot(tai['date'],tai['workplaces_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['workplaces_percent_change_from_baseline'])
plt.plot(us2['date'],us2['workplaces_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['workplaces_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('workplaces_percent_change_from_baseline.png')


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('grocery and pharmacy percent change from baseline')
plt.plot(bd['date'],bd['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(tai['date'],tai['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(us2['date'],us2['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['grocery_and_pharmacy_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('grocery_and_pharmacy_percent_change_from_baseline.png')


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('parks percent change from baseline')
plt.plot(bd['date'],bd['parks_percent_change_from_baseline'])
plt.plot(tai['date'],tai['parks_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['parks_percent_change_from_baseline'])
plt.plot(us2['date'],us2['parks_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['parks_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('parks_percent_change_from_baseline.png')


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('transit stations percent change from baseline')
plt.plot(bd['date'],bd['transit_stations_percent_change_from_baseline'])
plt.plot(tai['date'],tai['transit_stations_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['transit_stations_percent_change_from_baseline'])
plt.plot(us2['date'],us2['transit_stations_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['transit_stations_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('transit_stations_percent_change_from_baseline.png')


# Comparing here Bangladesh and New York.
# Because they have got infection almost similar time.

# In[ ]:


ny=us[us['sub_region_1']=='New York']


# In[ ]:


ny=ny[ny['sub_region_2']=='x']


# In[ ]:


ny.shape


# In[ ]:


ny.head(3)


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('retail_and_recreation_percent_change_from_baseline')
plt.plot(bd['date'],bd['retail_and_recreation_percent_change_from_baseline'])
plt.plot(ny['date'],ny['retail_and_recreation_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('residential_percent_change_from_baseline')
plt.plot(bd['date'],bd['residential_percent_change_from_baseline'])
plt.plot(ny['date'],ny['residential_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('workplaces_percent_change_from_baseline')
plt.plot(bd['date'],bd['workplaces_percent_change_from_baseline'])
plt.plot(ny['date'],ny['workplaces_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('transit_stations_percent_change_from_baseline')
plt.plot(bd['date'],bd['transit_stations_percent_change_from_baseline'])
plt.plot(ny['date'],ny['transit_stations_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('parks_percent_change_from_baseline')
plt.plot(bd['date'],bd['parks_percent_change_from_baseline'])
plt.plot(ny['date'],ny['parks_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)


# In[ ]:


plt.figure(figsize=(30,12))
plt.title('grocery_and_pharmacy_percent_change_from_baseline')
plt.plot(bd['date'],bd['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(ny['date'],ny['grocery_and_pharmacy_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)

