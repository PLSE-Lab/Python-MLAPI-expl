#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly as ply


# In[ ]:





# ### Data Pre-processing 

# In[ ]:


appdata=pd.read_csv('../input/AppleStore.csv')


# In[ ]:


appdata.columns


# #### As data contains two trends for analysis i-e ratings for current version and other for all released versions, I will explore the current version and its rating trend. 

# In[ ]:


appdata.drop('Unnamed: 0',axis=1)


# In[ ]:


appdata=appdata.drop(['Unnamed: 0','vpp_lic','currency'],axis=1)


# In[ ]:


appdata


# In[ ]:


appdata['Size_GB']=appdata['size_bytes']/(1024*1024)


# In[ ]:


appdata


# In[ ]:


appdata.rename(columns={'track_name':'app_name','cont_rating':'content_rate',
                        'prime_genre':'genre','rating_count_tot':'versions_rating',
                       'rating_count_ver':'version_rating','sup_devices.num':'supp_devices','ipadSc_urls.num':'screen_shots_displayed',
                       'lang.num':'supp_lang_num'},inplace=True)


# In[ ]:


appdata


# ### DATA CLEANING

# In[ ]:


appdata=appdata.loc[:,['app_name','genre','user_rating_ver','version_rating','price','supp_devices','screen_shots_displayed','size_bytes']]


# In[ ]:


appdata


# In[ ]:


appdata.head()


# In[ ]:


appdata=appdata.sort_values(by=['user_rating_ver','version_rating'],ascending=False)


# In[ ]:


appdata.head(10)


# #### Top paid Apps

# In[ ]:


paidapps=appdata[appdata['price']>0.0]


# In[ ]:


paidapps.count()


# In[ ]:


paidapps=paidapps.sort_values(by=['price'],ascending=False)


# In[ ]:


paidapps.head()


# ### Paid Apps by Category 

# In[ ]:


paid_apps=paidapps.groupby(['genre']).count()


# In[ ]:


paid_apps['app_name'].plot(kind='barh',
                          figsize=(10,6),
                          alpha=0.98)
plt.xlabel('Frequency Count')
plt.ylabel('Category')
plt.title('Paid Apps Category Wise')
plt.show()


# ### To find the ratings of the apps related to games

# In[ ]:


games=appdata.loc[appdata['genre']=='Games']


# In[ ]:


games


# In[ ]:


gamesapps=games.groupby(['user_rating_ver']).count()


# In[ ]:


gamesapps['app_name'].plot(kind='barh',
                          figsize=(10,6),
                          alpha=0.98)
plt.xlabel('Frequency Count')
plt.ylabel('Rating')
plt.title('Games Classified by User Rating')
plt.show()


# ### Here we will find the the mostly used category of Apps which were rated five star by both Parametres i-e (Paid & free) 

# In[ ]:


top_rated=appdata.loc[appdata['user_rating_ver']==5.0]
top_rated


# In[ ]:


paid_apps=top_rated[top_rated['price']>0]
rated_paid_apps=paid_apps.sort_values('version_rating',ascending=False)
top_rated_paid_apps=rated_paid_apps.groupby(by='genre').count()
top_rated_paid_apps=top_rated_paid_apps['app_name']
top_rated_paid_apps


# In[ ]:


free_apps=top_rated[top_rated['price']==0.0]
rated_free_apps=free_apps.sort_values('version_rating',ascending=False)
top_rated_free_apps=rated_free_apps.groupby(by='genre').count()
top_rated_free_apps=top_rated_free_apps['app_name']
top_rated_free_apps


# In[ ]:


genre=np.unique(appdata['genre'])
genre


# In[ ]:





# In[ ]:


frame={'top_rated_free':top_rated_free_apps,'top_rated_paid':top_rated_paid_apps}


# In[ ]:


combined=pd.DataFrame(frame,index=genre)


# In[ ]:


combined.plot(kind='barh',
             figsize=(10,6))
plt.xlabel('Rating Counts')
plt.ylabel('Genre')
plt.show()


# ### As majority of games were rated 4.5 star let's explore them

# In[ ]:


four_rated=games.loc[games.user_rating_ver==4.5]


# In[ ]:


four_rated


# In[ ]:


four_paid_apps=four_rated[four_rated['price']>0]
four_rated_paid_apps=four_paid_apps.sort_values('version_rating',ascending=False)
four_rated_paid_apps=four_rated_paid_apps.groupby(by='genre').count()
four_rated_paid_apps=four_rated_paid_apps['app_name']
four_rated_paid_apps


# In[ ]:


four_free_apps=four_rated[four_rated['price']==0.0]
four_rated_free_apps=four_free_apps.sort_values('version_rating',ascending=False)
four_rated_free_apps=four_rated_free_apps.groupby(by='genre').count()
four_rated_free_apps=four_rated_free_apps['app_name']
four_rated_free_apps


# ### There may be numerous aspects or trends in this data but these were major that I analyzed, Happy Coding

# In[ ]:





# In[ ]:




