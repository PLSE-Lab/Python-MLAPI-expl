#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ![Worldcup](http://img.fifa.com/images/fwc/2018/opengraph/generic.png)
# 
# 
# **[FIFA](https://en.wikipedia.org/wiki/FIFA_World_Cup)** 2018 just begun a few days ago. People keep guessing who is going to be the Champion of FIFA World Cup 2018. They even got a cat for this job. However, I just want to know more about the teams. Legend says that Germany is going to win again this time. Rumor says that Belgium is underestimated, and they are going to surprised everybody. I hope my favorite team, Argentina, can have a better grade this time. 
# 
# This data includes how many times each team have made it to the final, semifinal, and even win the championship. I'm going to explore this data and try to predict who can make it to the top 16. May the force be with the strongest team.

# In[ ]:


df = pd.read_csv('../input/World Cup 2018 Dataset.csv')
df = df.dropna(axis='rows',how='all')
df = df.dropna(axis='columns',how='all')


# In[ ]:


df


# In[ ]:


df = df.fillna(0)


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df['score'] = (df['Previous \nappearances'] + 4 * df['Previous\n semifinals'] + 8 * df['Previous\n finals'] + 16 * df['Previous \ntitles'])
df['group_win'] = (df['history with \nfirst opponent\n W-L'] + df['history with\n second opponent\n W-L'] + df['history with\n third opponent\n W-L'])
df['group_goals'] = (df['history with\n first opponent\n goals'] + df['history with\n second opponent\n goals'] + df['history with\n third opponent\n goals'])


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.barh(df.Team, df.score)


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.barh(df.Team, df.group_win)


# Basically, I think the country with bar on the right side can pass the first round

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.barh(df.Team, df.group_goals)


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

barWidth = 0.9

plt.bar(df.Team, df.score, width = barWidth, label='Total score')
plt.bar(df.Team, df.group_win, width = barWidth, label='Wins to opponenets')
plt.bar(df.Team, df.group_goals, width = barWidth, label='Goals against opponents')

plt.legend()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom= 0.2, top = 0.98)
 
plt.show()


# In[ ]:


df_tree = df.drop(df.index[[13,25]])


# In[ ]:


import squarify

df_sub = df_tree.loc[(df_tree!=0).any(axis=1)]

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

squarify.plot(sizes=df_tree['Previous \nappearances'], label=df_tree['Team'])
plt.axis('off')
plt.title('Distribution of appearances')
plt.show()


# ## Radar chart
# 
# As those magazine or report you may have seen, I want to create radar charts to scale each team's strength and make prediction about who will make it to the top 16
# Their is five index
# 
# * Rank = (70 - df.Current FIFA rank)/7
# * Score = (max = 200) /20
# * wins(max = 18,min = -17) = (17 + df.group_win)/3.5
# * Goals(max = 72,min = -72) = 72 + df.group_goals/14
# * Appearance(max = 20) = df.Previous appearances / 2
# 
# #Round to integer

# In[ ]:


df.describe()


# In[ ]:


df.head(4)


# In[ ]:


#Rank = (70 - df.Current FIFA rank)/7
#Score = (max = 200) /20
#wins(max = 18,min = -17) = (17 + df.group_win)/3.5
#Goals(max = 72,min = -72) = 72 + df.group_goals/14
#Appearance(max = 20) = df.Previous appearances / 2
#Round to integer
df_radar = pd.DataFrame({
    'group': ['Russia','Saudi Arabia','Egypt','Uruguay'],
    'Rank': [1, 1, 6, 7],
    'Score': [1, 0, 0, 4],
    'Wins': [5, 4, 6, 5],
    'Goals': [5, 5, 5, 5],
    'Appearance': [5, 2, 1, 6]
})


# In[ ]:


# Refer to https://python-graph-gallery.com/radar-chart/

from math import pi

def make_spider( row, title, color):
 
    # number of variable
    categories=list(df_radar)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,2,row+1, polar=True, )
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8], ["2","4","6","8"], color="grey", size=7)
    plt.ylim(0,10)
 
    # Ind1
    values=df_radar.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.1)
 
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
my_palette = plt.cm.get_cmap("Set2", len(df.index))
# Loop to plot
for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# From the radar chart I made, I think for Group A, Uruguay will make it to the top 16. The other three is pretty equal based on the radar chart.

# In[ ]:


df[4:8]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['Portugal','Spain','Morocco','Iran'],
    'Rank': [10, 9, 4, 5],
    'Score': [1, 2, 0, 0],
    'Wins': [2, 10, 3, 5],
    'Goals': [3, 8, 5, 5],
    'Appearance': [3, 7, 2, 2]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# For group B, Spain and Porugal is definitely gonna make it.

# In[ ]:


df[8:12]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['France','Australia','Peru','Denmark'],
    'Rank': [9, 5, 8, 8],
    'Score': [3, 0, 0, 0],
    'Wins': [6, 4, 5, 5],
    'Goals': [6, 5, 5, 5],
    'Appearance': [7, 2, 2, 2]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# Group C, France is defenitely gonna make it. Though the rank of Australia isn't as good as the other 3, based on thier performance in thier fist game against France, it's hard to tell which of them will be another team that make it to the top 16.

# In[ ]:


df[12:16]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['Argentina','Iceland','Croatia','Nigeria'],
    'Rank': [9, 7, 8, 3],
    'Score': [5, 0, 0, 0],
    'Wins': [6, 4, 5, 4],
    'Goals': [5, 5, 6, 5],
    'Appearance': [8, 0, 2, 3]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# For group D, Argentian totally can nail it. Croatia probably can pass the first round. However, I personally think Iceland has chance to win few games.

# In[ ]:


df[16:20]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['Brazil','Switzerland','Costarica','Serbia'],
    'Rank': [10, 9, 6, 5],
    'Score': [10, 1, 0, 1],
    'Wins': [8, 5, 3, 5],
    'Goals': [7, 5, 4, 5],
    'Appearance': [10, 5, 2, 6]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# For group E, Brazil and Switzerland will make it to the top 16.

# In[ ]:


df[20:24]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['Germany','Mexico','Sweden','Korea'],
    'Rank': [10, 8, 7, 2],
    'Score': [10, 1, 2, 1],
    'Wins': [7, 4, 5, 4],
    'Goals': [7, 5, 5, 4],
    'Appearance': [9, 8, 6, 5]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# For group F, I think Germany already got the winning ticket. I think another one that get to the top 16 is Mexico.

# In[ ]:


df[24:28]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['Belgium','Panama','Tunisia','England'],
    'Rank': [9, 2, 6, 8],
    'Score': [1, 0, 0, 2],
    'Wins': [0, 5, 5, 10],
    'Goals': [0, 5, 5, 10],
    'Appearance': [6, 0, 2, 7]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# For group G, England is going to the second round. The radar chart of Belgium is pretty bad most because they lose to England so many time. Since the way to win first round is to have the higher score as possible, Belgium is still probably another team that go to the second round.

# In[ ]:


df[28:32]


# In[ ]:


df_radar = pd.DataFrame({
    'group': ['Poland','Senegal','Columbia','Japan'],
    'Rank': [9, 7, 8, 2],
    'Score': [1, 0, 0, 0],
    'Wins': [4, 4, 4, 2],
    'Goals': [5, 5, 5, 4],
    'Appearance': [4, 1, 3, 3]
})


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))


# For group H, I think Poland and Columbia will be the last two that get the tickets to the top 16.

# ## Now, the top 16 teams are settled. Let me see how many teams I predicted made it.
# 
# The list I predicted :
# 
# **Uruguay/Spain/Porugal/France/Argentian/Croatia/Brazil/Switzerland/Germany/Mexico/England/Belgium/Poland/Columbia**
# 
# **True/True/True/True/True/True/True/True/False/True/True/True/False/True**
# 
# ### 12 out of 14 are right
# 
# The accuracy of my prediction is 85%
# 
# While overall accuracy is 75%
# 
# ### Now, I'll use another scale to make comparison of each match
# 
# I'll keep appearance and rank, while use new wins, GS, and GA as those of the last three games.

#  ### Matches
#  
#  * France vs Argentina
#  * Uruguay vs Portugal
#  * Spain vs Russia
#  * Croatia vs Denmark
#  * Brazil vs Mexico
#  * Belgium vs Japan
#  * Switzerland vs Sweden
#  * Columbia vs England

# In[ ]:


match = pd.DataFrame({
    'Team': ['France','Argentina','Uruguay','Portugal','Spain','Russia','Croatia','Denmark','Brazil','Mexico','Belgium','Japan','Switzerland','Sweden','Columbia','England'],
    'Appearance': [7,8,6,3,7,5,2,8,10,8,6,3,5,6,3,7],
    'Rank': [9,9,7,10,9,1,8,8,10,8,9,2,9,7,8,8],
    'Wins': [2,1,3,1,1,2,3,1,2,2,3,1,1,2,2,2],
    'GS': [3,7,5,5,6,8,7,2,5,3,9,4,5,5,5,8],
    'GA': [9,9,10,6,5,6,9,9,9,6,8,6,6,8,7,7]
})


# **Radar Chart (overlapped)**
# 
# refer to : https://python-graph-gallery.com/391-radar-chart-with-several-individuals/

# In[ ]:


# Refer to https://python-graph-gallery.com/radar-chart/

from math import pi

def make_spider( row, title, color):
 
    # number of variable
    categories=list(match)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,8,row+1, polar=True, )
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8], ["2","4","6","8"], color="grey", size=7)
    plt.ylim(0,10)
 
    # Ind1
    values=match.loc[row].drop('Team').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.4)
 
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(match.index)):
    make_spider(row=row, title='group' + match['Team'][row], color=my_palette(row))


# **In my opinion, I think these teams will win the matches :**
# 
# * Argentina X
# * Uruguay V
# * Spain X
# * Croatia V
# * Brazil V
# * Belgium V
# * Sweden V
# * England V
# 
# ### OK, so now all top 8s are settled. I predicted 6 out of 8 right. I need to be honest, Russia did surprise me with their performance.

# In[ ]:


match = pd.DataFrame({
    'Team': ['France','Uruguay','Russia','Croatia','Brazil','Belgium','Sweden','England'],
    'Appearance': [7,6,5,2,10,5,6,7],
    'Rank': [9,7,1,8,10,9,7,8],
    'Wins': [2,3,2,3,2,1,2,2],
    'GS': [3,5,8,7,5,5,5,8],
    'GA': [9,10,6,9,9,6,8,7]
})


# In[ ]:


# Refer to https://python-graph-gallery.com/radar-chart/

from math import pi

def make_spider( row, title, color):
 
    # number of variable
    categories=list(match)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,4,row+1, polar=True, )
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8], ["2","4","6","8"], color="grey", size=7)
    plt.ylim(0,10)
 
    # Ind1
    values=match.loc[row].drop('Team').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.4)
 
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(match.index)):
    make_spider(row=row, title='group' + match['Team'][row], color=my_palette(row))


# ## Upvote if you find it interesting. Let's see if I predict it right.

# In[ ]:




