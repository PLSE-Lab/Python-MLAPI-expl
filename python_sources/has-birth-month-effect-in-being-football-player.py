#!/usr/bin/env python
# coding: utf-8

# A colleage told me a story about professional players, that they are usually born in the first months of the year. The thing is that when they are young they are usually the olders in the class, so they are more "developed" and better than younger players.  Let's see if this is true with footballers

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

footballers_dataset =  pd.read_csv('../input/complete-fifa-2017-player-dataset-global/FullData.csv', header=0)
general_dataset =  pd.read_csv('../input/births-in-us-1994-to-2003/births.csv', header=0)
world_cup_players_dataset =  pd.read_csv('../input/fifa-world-cup-2018-players/wc2018-players.csv', header=0)


# Let's get the data: general population, FIFA players db, and World Cup players db.

# In[ ]:


footballers_birth_dates_months = footballers_dataset.apply(lambda x: datetime.strptime(x.Birth_Date, "%m/%d/%Y").month, axis=1) 
world_cup_footballers_birth_dates_months = world_cup_players_dataset.apply(lambda x: datetime.strptime(x['Birth Date'], "%d.%m.%Y").month, axis=1) 


# In[ ]:


general_by_month=general_dataset.groupby(['month'])['births'].sum()
general_by_month.plot.bar();


# Ok, there is not a lot of deviation in the general population birth month. How about footballers?

# In[ ]:


plt.title('FIFA Footballers birth month histogram', fontsize=18)
_ = plt.hist(footballers_birth_dates_months, 10, alpha=0.5, label='Month')


# In[ ]:


plt.title('World Cup Footballers birth month histogram', fontsize=18)
_ = plt.hist(world_cup_footballers_birth_dates_months, 10, alpha=0.5, label='Month')


# What about Spain? (My country)

# In[ ]:


spanish_footballers_birth_dates_months = footballers_dataset[footballers_dataset['Nationality']=='Spain'].apply(lambda x: datetime.strptime(x.Birth_Date, "%m/%d/%Y").month, axis=1) 
plt.title('FIFA Spanish Footballers birth month histogram', fontsize=18)
_ = plt.hist(spanish_footballers_birth_dates_months, 10, alpha=0.5, label='Month')


# In[ ]:


spanish_world_cup_footballers_birth_dates_months = world_cup_players_dataset[world_cup_players_dataset['Team']=='Spain'].apply(lambda x: datetime.strptime(x['Birth Date'], "%d.%m.%Y").month, axis=1) 
plt.title('World Cup Spanish Footballers birth month histogram', fontsize=18)
_ = plt.hist(spanish_world_cup_footballers_birth_dates_months, 10, alpha=0.5, label='Month')


# **So it seems that there are some kind of effect. Footballers born in January have more probability of being a professional player than the ones born in the following months. I was born in April, that explains a lot of things.**

# In[ ]:




