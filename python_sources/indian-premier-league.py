#!/usr/bin/env python
# coding: utf-8

# ## Fetch data

# In[27]:


#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')
# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# In[28]:


#read csv into a pandas dataframe.
matches = pd.read_csv("../input/matches.csv")


# In[29]:


#check the first 5 rows.
matches.head()


# ## Clean data

# #### Check null columns

# In[30]:


matches.isnull().sum()


# The above query yields that there are 5 columns which have null values for some rows. it doesn't look like that we will need to delete any entries in order to have a clean dataset, we can work to replace the null values with appropriate data.

# #### Check rows with city column as null

# In[31]:


matches[matches.city.isnull() == True]


# We can easliy see from the above data that all of the matches for which city value is missing were held at 'Dubai International Cricket Stadium'.
# 
# We now need to check whether any other data row has a city value available with thricket e venue as 'Dubai International Cricket Stadium'.

# In[32]:


matches[matches.venue == 'Dubai International Cricket Stadium'].city.value_counts(dropna=False)


# So we couldn't find a city value from the available data. So we would be best placed if we update the city value as 'Dubai'.

# In[33]:


#Update city as 'Dubai' wherever it is null.
matches['city'].fillna(value='Dubai',inplace=True)


# #### Check rows with winner column as null

# In[34]:


matches[matches.winner.isnull() == True]


# So we have 3 rows having winner as null, looking at the result column gives us the info that this is because there was no result for the game. The same is true for player of the match column, so we need not check it separately now.
# 
# We can update both winner and player of the match columns with 'no result' to have a clean dataset.

# In[35]:


#Update winner and player_of_match columns as 'no result' wherever they are null.
matches['winner'].fillna(value='no result',inplace=True)
matches['player_of_match'].fillna(value='no result',inplace=True)


# #### Check rows with umpire1 and umpire2 as null

# In[36]:


matches[matches.umpire1.isnull() == True]


# Looking at the data doesn't yield any specific reason why umpire1 and umpire2 are not available. i think we are fine if we just update the missing values as 'not available'.

# In[37]:


#Update umpire1 and umpire2 as 'not available' wherever their value is null.
matches['umpire1'].fillna(value='not available',inplace=True)
matches['umpire2'].fillna(value='not available',inplace=True)


# #### Check rows with umpire3 column value as null. 

# We don't have a single row having not null value for umpire3, so this particular column should be dropped from the dataset.

# In[38]:


#drop the null column
matches.dropna(axis=1,inplace=True)


# Check the final shape of the dataset

# In[39]:


matches.shape


# ## Analyse data

# ### 1. Player of the match

# #### Top 10 'player of the match' total for all seasons.

# In[40]:


matches.player_of_match.value_counts().head(10).sort_values().plot(kind='barh')


# #### Player with maximum number of 'player of the match' in a season

# In[41]:


vc = matches.groupby('season').player_of_match.value_counts()
imax = vc.idxmax()
print('\n\nYear       :', imax[0], '\nPlayer     :',imax[1], '\nNo. of PoM :', vc[vc.idxmax()])


# #### Host city when CH Gayle was 'Player of the match'.

# In[42]:


matches[matches.player_of_match == 'CH Gayle'].city.value_counts().plot(kind='barh')


# #### Winner team when CH Gayle was 'player of the match'

# In[43]:


matches[matches.player_of_match == 'CH Gayle'].winner.value_counts().plot(kind='barh')


# #### Opposition team against which CH Gayle got the most 'player of the match' awards.

# In[44]:


a = matches[(matches.player_of_match == 'CH Gayle') & (matches.team1 == 'Royal Challengers Bangalore')].team2
b = matches[(matches.player_of_match == 'CH Gayle') & (matches.team2 == 'Royal Challengers Bangalore')].team1
a.append(b).value_counts().plot(kind='barh')


# ### 2. Matches 

# #### Number of matches in each season

# In[45]:


matches.season.value_counts().sort_index().plot(kind='bar')


# ### 3. Win by wickets analysis

# In[46]:


matches[matches.win_by_wickets > 0].win_by_wickets.plot.hist(bins=5)
plt.xlabel('Win by wickets')


# In[49]:


matches[['winner','win_by_wickets']][matches.win_by_wickets > 0].boxplot(vert=False, by='winner')


# ### 4. Win by runs analysis

# In[48]:


matches[matches.win_by_runs > 0].win_by_runs.plot.hist(bins=15)
plt.xlabel('Win by runs')


# In[50]:


matches[['winner','win_by_runs']][matches.win_by_runs > 0].boxplot(vert=False, by='winner')

