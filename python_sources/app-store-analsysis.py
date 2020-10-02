#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


appstore_df = pd.read_csv("../input/AppleStore.csv")
appstore_df.drop(columns=['Unnamed: 0'], inplace = True)
appstore_df.head()


# In[ ]:


appstore_df.shape


# In[ ]:


appstore_df.info()


# **Top Free Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 100000) 
            & (appstore_df.price == 0)][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(20)


# **Top Paid Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 100000) 
            & (appstore_df.price > 0)][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(20)


# **Top Free Games**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 100000)
           & (appstore_df.prime_genre == 'Games') 
           & (appstore_df.price == 0)][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Paid Games**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 100000)
           & (appstore_df.prime_genre == 'Games') 
           & (appstore_df.price > 0)][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0)


# **Top Games which have 17+ Content Rating**

# In[ ]:


appstore_df[(appstore_df.prime_genre == 'Games') 
           & (appstore_df.cont_rating == '17+')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Games which have 12+ Content Rating**

# In[ ]:


appstore_df[(appstore_df.prime_genre == 'Games') 
           & (appstore_df.cont_rating == '12+')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Games which have 4+ Content Rating**

# In[ ]:


appstore_df[(appstore_df.prime_genre == 'Games') 
           & (appstore_df.cont_rating == '4+')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Entertainment Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 10000)
           & (appstore_df.prime_genre == 'Entertainment')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Education Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 10000)
           & (appstore_df.prime_genre == 'Education')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Productivity Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 10000)
           & (appstore_df.prime_genre == 'Productivity')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Photo & Video Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 10000)
           & (appstore_df.prime_genre == 'Photo & Video')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# **Top Utilities Apps**

# In[ ]:


appstore_df[(appstore_df.rating_count_tot > 10000)
           & (appstore_df.prime_genre == 'Utilities')][['track_name','rating_count_tot','user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# <h1>The Worst Apps </h1><br>
# **Apps which have recived less than 2 ratings **

# In[ ]:


appstore_df[(appstore_df.user_rating < 2)
           & (appstore_df.rating_count_tot > 100)][['track_name','rating_count_tot', 'user_rating']].sort_values('rating_count_tot', ascending=0).head(25)


# <h1>The Worst Games </h1><br>
# **Games which have recived less than 2.5  ratings **

# In[ ]:


appstore_df[(appstore_df.user_rating < 2.5)
           & (appstore_df.rating_count_tot > 100)
           & (appstore_df.prime_genre == 'Games')][['track_name','rating_count_tot', 'user_rating']].sort_values('rating_count_tot', ascending=0)


# In[ ]:


free_app = appstore_df[appstore_df.price == 0].track_name.count()
paid_app = appstore_df[appstore_df.price >0].track_name.count()

print('Total Number of Free Apps =', free_app)
print('Total Number of Paid Apps =', paid_app)

apps = {'Free Apps' : [free_app],
        'Paid Apps' : [paid_app]
       }
df_apps = pd.DataFrame.from_dict(apps)
df_apps.plot(kind='bar', colormap='Spectral')


# In[ ]:


pd.DataFrame.from_dict(appstore_df.cont_rating.value_counts(sort=True, ascending=False))


# In[ ]:


print(pd.DataFrame.from_dict(appstore_df.cont_rating.value_counts(sort=True, ascending=False)))
plt.figure(figsize=(8,7))
sns.set_style("darkgrid")
sns.countplot(appstore_df.cont_rating, palette='Spectral')
plt.grid(True)
plt.xlabel('Content Ratings', fontsize=14, color='#191970')
plt.ylabel('Number of Apps', fontsize=14, color='#191970')


# In[ ]:





# In[ ]:


df1 = appstore_df[['id', 'track_name', 'rating_count_tot']].sort_values('rating_count_tot', ascending=0)

plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
sns.barplot(x=df1['track_name'].head(12), y=df1['rating_count_tot'], linewidth=1, edgecolor="k"*len(df1)) 
plt.grid(True)
plt.xticks(rotation=30)
plt.xlabel('Apps', fontsize=15, color='#191970')
plt.ylabel('Total Number of Ratings', fontsize=15, color='#191970')
plt.title('User Rating counts', fontsize=15, color='#191970')
plt.show()


# In[ ]:


plt.figure(figsize=(13,8))
sns.set_style("darkgrid")

sns.countplot(y=appstore_df.prime_genre, linewidth=1, edgecolor="k"*len(appstore_df), palette='Spectral') 
plt.grid(True)
plt.xticks(rotation=10)
plt.xlabel('Number of Apps', fontsize=15, color='#191970')
plt.ylabel('Prime Genres', fontsize=15, color='#191970')
plt.show()

