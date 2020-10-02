#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

draft_years = []
for year in range(2009, 2019):
    df = pd.read_csv("../input/{}.csv".format(year))
    
    df['LinearDraftValue'] = len(df) - df['Overall']
    
    draft_years.append(df)
    
last_10_years = pd.concat(draft_years)

# Fill NA values that we'll do math on later
last_10_years = last_10_years.fillna({'GP': 0.0})

# Cleanup for moved/renamed teams.
last_10_years = last_10_years.replace('Phoenix Coyotes', 'Arizona Coyotes')
last_10_years = last_10_years.replace('Atlanta Thrashers', 'Winnipeg Jets')


# # Draft Pick EDA
# 
# I started by looking at NHL draft data from the last 10 years. Here is a sample of what we have to work with for each year (with some additional stats removed).

# In[ ]:


df[['Overall', 'Team', 'Player', 'Nat.', 'Pos', 'GP']].head(10)


# ## Scoring Value of Draft Picks by Team
# 
# Being an Edmonton Oilers fan, I wanted to see just how their high draft picks actually compared to other teams if I applied a points value to each pick. I started with a linear scoring system that assumes the #1 draft pick is worth the most, and the last draft pick is basically worthless.
# 
# Obviously this system doesn't represent the value of high draft pics very well, but I thought it would be interesting to compare.

# In[ ]:


teams = last_10_years.groupby('Team')['LinearDraftValue'].agg('sum').sort_values()

plt.figure(figsize=(10,7))
teams.plot(kind='barh', title='Linear Value of Overall Draft Pick Scoring by Team')
plt.show()


# ## Need a Better Scoring System
# 
# No surprise that the oilers come out on top, and teams like the Caps would come last. I would have guessed Coyotes and Sabres would be high as well, but I'm a little surprised to see Chicago as high as it is.
# 
# Obviously the real value of draft picks shouldn't be linear like that since there's a pretty quick dropoff in the players ability to make the NHL if they aren't taken in the first round.
# 
# I wanted to examine that, and I think the graph below show just how far and fast it drops off when we look at average NHL games played over the years by each pick. I think it's fair to say that 20+ years of data would probably smooth this out quite a bit, but the distribution seems fine.

# In[ ]:


gp_by_draft_order = last_10_years.groupby('Overall')['GP'].agg('mean')

plt.figure(figsize=(10,7))
gp_by_draft_order.plot(title="Overall Draft Pick Value by Average Games Played")
plt.show()


# ## Taking Another Look
# 
# So now we can apply this games played data as a new way to weight the score of each draft pick. The data starts to get a little more interesting. I think the noise in the data may be causing some subtle anomolies, but it's pretty interesting that the Oilers are no longer the clear cut leader based on their #1 overall picks, which is actually the opposite of what I was expecting based on the new scoring system.

# In[ ]:


last_10_years['DraftValue'] = last_10_years['Overall'].apply(lambda x: gp_by_draft_order.loc[x])

teams = last_10_years.groupby('Team')['DraftValue'].agg('sum').sort_values()

plt.figure(figsize=(10,7))
teams.plot(kind='barh', title='Value of Overall Draft Picks by Team')
plt.show()

