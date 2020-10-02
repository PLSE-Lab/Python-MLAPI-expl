#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ![](https://live.staticflickr.com/8453/8054927083_e503602d8b_b.jpg)This is a brief starter kernel looking at the Triple Crown dataset. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style = 'white')


# In[ ]:


conditions = pd.read_csv("../input/triple-crown-of-horse-races-2005-2019/TrackConditions.csv")
races = pd.read_csv("../input/triple-crown-of-horse-races-2005-2019/TripleCrownRaces_2005-2019.csv")


# # Looking at the Data
# First, let's see what the data looks like.

# In[ ]:


races.head(20)


# In[ ]:


races.dtypes


# In[ ]:


conditions.head(20)


# In[ ]:


conditions.dtypes


# # Merging Track Conditions with Races
# If you want to see how horses fare under different conditions, you'll need to merge the two data sets on `year` and `race`.

# In[ ]:


df = races.merge(conditions, on = ['year', 'race'])


# In[ ]:


df.head()


# # Plotting the Data
# First, we can take a look at the distribution of the betting odds.

# In[ ]:


plt.hist(df['Odds'])
plt.xlabel('Odds')


# Now, we can see how the payout for a win bet varies according to a horses odds. Only horses who actually win have information on their payout for a win.

# In[ ]:


df1 = df[df['final_place'] == 1]
plt.scatter(df1['Odds'], df1['Win'])


# We can also look at how the racetrack conditions on a given day affect a horses chances of winning. To to this, I will create a categorical variable for the horse's final place (Top 3 finish or Rest of the field). Then I will create a new dataframe that contains the mean odds for horses grouped by their final place and track conditions.

# In[ ]:


df['final_place_cat'] = pd.cut(df['final_place'], [0, 3, 22], labels = ['Top 3', 'Rest'], right = True) 

grouped_df = df.groupby(['final_place_cat', 'track_condition'])['Odds'].mean().reset_index()

barwidth = 0.25

bars1 = grouped_df['Odds'][grouped_df['track_condition'] == 'Fast']
bars2 = grouped_df['Odds'][grouped_df['track_condition'] == 'Muddy']
bars3 = grouped_df['Odds'][grouped_df['track_condition'] == 'Sloppy']

r1 = np.arange(len(bars1))
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]

fig, ax = plt.subplots(figsize = (10,6))
fig.tight_layout()
fig.subplots_adjust(bottom = 0.25, top = 0.9)
ax.bar(r1, bars1, color = 'b', width = barwidth, label = 'Fast')
ax.bar(r2, bars2, color = 'r', width = barwidth, label = 'Muddy')
ax.bar(r3, bars3, color = 'g', width = barwidth, label = 'Sloppy')
ax.set_ylabel('Starting Odds', fontsize = 16)
ax.set_yticklabels([0, 5, 10, 15, 20, 25], fontsize = 16)
ax.set_xlabel('Finishing Place', fontsize = 16)
ax.set_xticks([0.25, 1.25])
ax.set_xticklabels(['Top 3', 'Rest of Field'], fontsize = 16)
fig.legend(loc = 'lower center', ncol = 3, facecolor = 'white', edgecolor = 'white', fontsize = 16)
fig.suptitle("Starting Odds of Top 3 vs. Rest of the Field Based on Track Conditions", fontsize = 16)


# We can see here that on days in which the track was not considered fast (i.e. optimal), the starting odds of the top 3 tend to be higher. On a muddy day at the track, the top three finishers tend to have **much** higher odds, meaning they were rated as less likely to win. So, there is some evidence that the track conditions may level the playing field a bit.

# # Conclusion
# OK, so I hope this gave you a taste of the dataset. I would love to see what the community comes up with using these data.
