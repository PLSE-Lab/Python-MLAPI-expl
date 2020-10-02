#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


events=pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')


# In[ ]:


events.head()


# In[ ]:


region=pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')


# In[ ]:


region.head()


# # LET US MERGE EVENTS AND REGION BASED ON NOC:

# In[ ]:


merged = pd.merge(events, region, on='NOC', how='left')


# In[ ]:


merged.head()


# # NOW LET US LOOK AT THE GOLD MEDALISTS OUT OF THIS:

# In[ ]:


GOLDMEDALS = merged[(merged.Medal == 'Gold')]
GOLDMEDALS.head()


# In[ ]:


GOLDMEDALS.isnull().sum()


# # AGE VS GOLDMEDALS

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(GOLDMEDALS['Age'])
plt.title('Distribution of Gold Medals')


# In[ ]:


GOLDMEDALS['ID'][GOLDMEDALS['Age'] > 50].count()


# This clearly shows that around 65 people who are aged above 50 won a gold medal.

# In[ ]:


GOLDMEDALS['ID'][GOLDMEDALS['Age'] > 60].count()


# 6 people who are above 60 won a gold medal.

# # LET'S LOOK AT THE DIFFERENT SPORTS DISCIPLINES:

# In[ ]:


DISCIPLINES = GOLDMEDALS['Sport'][GOLDMEDALS['Age'] > 50]


# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(DISCIPLINES)
plt.title('Gold Medals for Athletes Over 50')


# So the maximum people won medals for Horse riding,followed by Sailing.

# #  GOLD MEDALS WON BY EACH NATION:

# In[ ]:


GOLDMEDALS.region.value_counts().reset_index(name='Medal').head(5)


# It is clear from the above table that maximum medals was won by USA followed by Russia.
# 
# Let's look at this graphically to get a better understanding about it.

# In[ ]:


TOTALGOLDMEDALS = GOLDMEDALS.region.value_counts().reset_index(name='Medal').head(5)
g = sns.catplot(x="index", y="Medal", data=TOTALGOLDMEDALS,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_xlabels("Top 5 countries")
g.set_ylabels("Number of Medals")
plt.title('Medals per Country')


# This clearly shows USA bagged the maximum number of medals here.
# 
# USA won by a huge margin compared to other countries.
# 
# The second position was secured by Russia.

# # Let us look in which all disciplines did USA won maximum gold medals:

# In[ ]:


GOLDMEDALSUSA = GOLDMEDALS.loc[GOLDMEDALS['NOC'] == 'USA']


# In[ ]:


GOLDMEDALSUSA.Event.value_counts().reset_index(name='Medal').head(20)


# In[ ]:





# Thus we can see that Basketball tops this list.
# 
# Maximum number of medals was won by the Men's Basketball team.

# # Let's look at the Male Athletes:

# In[ ]:


BASKETBALLGOLDUSA = GOLDMEDALSUSA.loc[(GOLDMEDALSUSA['Sport'] == 'Basketball') & (GOLDMEDALSUSA['Sex'] == 'M')].sort_values(['Year'])


# In[ ]:


BASKETBALLGOLDUSA.head()


# # LET US LOOK AT THE HEIGHT AND WEIGHT:

# In[ ]:


NOTNullMedals = GOLDMEDALS[(GOLDMEDALS['Height'].notnull()) & (GOLDMEDALS['Weight'].notnull())]


# In[ ]:


NOTNullMedals.head()


# In[ ]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="Height", y="Weight", data=NOTNullMedals)
plt.title('Height vs Weight of Olympic Medalists')


# The more the weight, the more the height.

# In[ ]:


NOTNullMedals.loc[NOTNullMedals['Weight'] > 160]


# These are the people with weight greater than 160, and we can see that these people are weightlifters.

# In[ ]:




