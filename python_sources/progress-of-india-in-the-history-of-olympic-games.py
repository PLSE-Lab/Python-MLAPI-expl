#!/usr/bin/env python
# coding: utf-8

# # Exploring the progress of India in the history of Olympic Games
# 
# 
# Thanks for stopping by my kernel.
# 
# 
# ### This work is inspired by the work of many kagglers who have worked on this dataset. If you like my findings on the dataset, please, leave an upvote. Also, This is my first kernel on Kagle community. Feedback on my findings is highly appreciated! 
# 
# I am using the dataset 120 years of Olympic history: athletes and results.
# 
# ## Content 
# 
# The file athlete_events.csv contains 271116 rows and 15 columns. Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events). The columns are:
# 
# 1. ID - Unique number for each athlete;
# 2. Name - Athlete's name;
# 3. Sex - M or F;
# 4. Age - Integer;
# 5. Height - In centimeters;
# 6. Weight - In kilograms;
# 7. Team - Team name;
# 8. NOC - National Olympic Committee 3-letter code;
# 9. Games - Year and season;
# 10. Year - Integer;
# 11. Season - Summer or Winter;
# 12. City - Host city;
# 13. Sport - Sport;
# 14. Event - Event;
# 15. Medal - Gold, Silver, Bronze, or NA.
# 
# 
# ![Image on web](https://3.bp.blogspot.com/-ABt6G4yEy1A/V6OCGTCuwnI/AAAAAAAAEmM/qIrshFpUGX42zNJb1CxGQ08xHH3leZlaACLcB/s1600/Capture.JPG)
# 
# 
# # Version 
# 
# version 1.0
# version 2.0 (Fixed the hockey medal count problem.)
# 
# 
# ## Index of content
# 
# 1. Importing the modules
# 2. Data Importing 
# 3. Explore and extract information
# 4. Joining the dataframes
# 5. Extract data of Indian Atheletes
# 6. Medals won according to year
# 7. Indian Women in Athletics
# 8. Total Medals and number of Medals won by year
# 9. Conclusion
# 
# ### Related Kernels
# 
# Related kernel: [marcogdepinto](https://www.kaggle.com/marcogdepinto/let-s-discover-more-about-the-olympic-games)

# # 1. Importing the modules

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os 
print(os.listdir("../input"))


# # 2. Data Importing

# In[ ]:


data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')


# # 3. Explore and Extract Information

# In[ ]:


data.head(5)


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


regions.head()


# # 4. Joining the dataframes
# 
# Pandas 'merge' function will help us to join these two dataframes easily.

# In[ ]:


merged = pd.merge(data, regions, on='NOC', how='left')


# In[ ]:


merged.head()


# # 5. Extract data of Indian Atheletes
# 
# Let's make a new dataframe including only Indian atheletes since this notebook findinds is about India.

# In[ ]:


dataIndia = merged[merged.NOC == 'IND']
dataIndia.head(5)


# Since we can see a lot of NaN values in the medals section. Let's filter those values out. In case where NaN values are not visible easily we can use the below method.

# In[ ]:


dataIndia.isnull().any()


# Let's filter out NaN values. We don't need them.

# In[ ]:


dataIndia = dataIndia[dataIndia.Medal.notnull()]


# In[ ]:


dataIndia.head()


# ## 6. Medals won according to year

# In[ ]:


plt.figure(figsize=(12, 6))
plt.tight_layout()
sns.countplot(x='Year', hue='Medal', data= dataIndia)
plt.title('Distribution of Medals')


# Seems like India won so many gold medals before 90s. Let's find out in which sport it won the gold medals.

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(x='Sport', data= dataIndia)
plt.title('Distribution of Medals')


# It seems that Hockey was the main sport responsible for India's medal haul.
# 
# It makes sense: Hockey being India's national sport.
# 
# ## Based on the suggestion of a fellow kaggler. I will count 1 medal for a hockey team instead of medal of every player of the team.
# 
# ### There are two ways to solve this medal problem of team. The easy way is to divide the total number of medal won in hockey by team size (i.e. 14). However, I will be solving it by other way so that where this dividing rule doesn't apply, people can use this method.
# 
# Let's draw this graph according to year.

# In[ ]:


hockeyPlayersMedal = dataIndia.loc[(dataIndia['Sport'] == 'Hockey')].sort_values(['Year'])
hockeyPlayersMedal.head(30)


# ## Let's count only one medal for a team in a particular game.

# In[ ]:


hockeyTeamMedal = hockeyPlayersMedal.groupby(['Year']).first()
hockeyTeamMedal.head()


# Now count the number of team medals in Hockey

# In[ ]:


hockeyTeamMedal['ID'].count()


# Not bad! I have verified this with [Wikipedia page](https://en.wikipedia.org/wiki/India_men%27s_national_field_hockey_team)
# 
# And we have fixed the problem.

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(x='Year', hue='Sport', data= dataIndia)
plt.title('Distribution of Medals by sports')


# It seems like Indian youth lost interest in Hockey after 90's. Sad :(

# 
# 
# 
# # 7. Indian Women in Athletics
# 
# 
# 
# We can study the data and draw conclusions about the involvement of women in India
# 
# 
# Let's create a filtered dataset:

# In[ ]:


womenInOlympics = merged[(merged.Sex == 'F') & (merged.NOC == 'IND')]


# In[ ]:


womenInOlympics.head()


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=womenInOlympics)
plt.title('Women participation per edition of the Games')


# It seems like Women started their journey way back from 1924 but Let's see when did the women wom their first medal.

# In[ ]:


womenInOlympics = dataIndia[dataIndia.Sex == 'F']
womenInOlympics.head()


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', hue='Medal', data=womenInOlympics)
plt.title('Women Medals per edition of the Games')


# Women in India won their first medal in the year 2000 and continued to win more medals in the upcoming games.

# # 8. Total Medals and number of Medals won by year
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


dataIndia.Medal.value_counts()


# It seems like Indians loved Gold.

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(x='Medal', hue='Year', data= dataIndia)
plt.title('Distribution of Medals by year')


# # 9. Conclusions
# 
# #### Thank you so much for reading. If you liked my work, please, upvote! This is my first kernel and I will be sharing more of my work regularly from now.
# 
# I will review and update the kernel periodically following your feedbacks. 
# 
# 
# Got a question? Ask me! :)

# In[ ]:




