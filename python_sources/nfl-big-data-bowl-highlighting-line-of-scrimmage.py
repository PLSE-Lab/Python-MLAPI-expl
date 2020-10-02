#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl
# ## Programmatically Highlighting The Line of Scrimmage
# 
# This notebook expands on the work done by [Rob Mulla](https://www.kaggle.com/robikscube) in his [NFL Big Data Bowl  Plotting Player Position](https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position) notebook. 
# 
# My aim is to programmatically calculate the line of scrimmage and highlight it. In the original notebook it was marked by the code "highlight_line_number=yl+54" which adds the required 54 yards to move the scrimmage line from the left side of the field to the right as was needed for the selected play.
# 
# The only section of this notebook that I will edit therefore is the "Highlight the line of scrimmage" section.
# 

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches
pd.set_option('max_columns', 100)

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# ## Function to Create The Football Field

# In[ ]:


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

create_football_field()
plt.show()


# # Adding Players For a Play

# In[ ]:


fig, ax = create_football_field()
train.query("PlayId == 20170907000118 and Team == 'away'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
train.query("PlayId == 20170907000118 and Team == 'home'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
plt.title('Play # 20170907000118')
plt.legend()
plt.show()


# # Highlight the line of scrimmage

# First I have to amend the data so that the abbreviations used in PossessionTeam match those used in HomeTeamAbbbr and VisitorTeamAbbr.

# In[ ]:


train['PossessionTeam'].replace(to_replace = "ARZ", value = "ARI", inplace=True)
train['PossessionTeam'].replace(to_replace = "BLT", value = "BAL", inplace=True)
train['PossessionTeam'].replace(to_replace = "CLV", value = "CLE", inplace=True)
train['PossessionTeam'].replace(to_replace = "HST", value = "HOU", inplace=True)


# I then created a TeamIdentifier column and use this to compare against PossessionTeam, calculate the average X position for the team in possession on the play before sorting the data so that for each GameId and PlayId the NaN's are listed first and backfilled with the PossessionTeamAvgXPos.

# In[ ]:


train['TeamIdentifier'] = np.where(train['Team'] == 'home', train['HomeTeamAbbr'], train['VisitorTeamAbbr'])
train['PossessionTeamAvgXPos'] = train.query('TeamIdentifier == PossessionTeam').groupby(['PlayId', 'Team'])['X'].transform('mean')
train.sort_values(by=['GameId','PlayId','PossessionTeamAvgXPos'])
train['PossessionTeamAvgXPos'].fillna(method='backfill', inplace=True)


# Finally I add a column holding the scrimmage line value for each play. As the field visualisation has an X length of 120 due to the endzones the 50 yard line is X = 60 so when the team in possession has an average position greater than 60 I calculate the X position of the scrimmage line as being YardLine + (2 * (50 - YardLine)).

# In[ ]:


train['ScrimmageLine'] = np.where(train['PossessionTeamAvgXPos'] > 60, (train['YardLine'] + (2 * (50 - train['YardLine']))), train['YardLine'])


# To show that this moves the line of scrimmage to the correct yard marker on the visualised field I'll take a game between Kansas City Chiefs and New England Patriots from the dataset and visualise each running play for that game.

# In[ ]:


testdata = train[train['GameId'] == 2017090700]
plays = testdata.PlayId.unique()


# In[ ]:


def get_play_visualisation(play):
    playid = play
    
    hometeam = train.query("PlayId == @playid")['HomeTeamAbbr'].tolist()[0]
    visitorteam = train.query("PlayId == @playid")['VisitorTeamAbbr'].tolist()[0]
    
    yl = train.query("PlayId == @playid")['YardLine'].tolist()[0]
    fig, ax = create_football_field(highlight_line=True,
                                    highlight_line_number=train.query('PlayId == @playid')['ScrimmageLine'].values[0])
    train.query("PlayId == @playid and Team == 'away'")         .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=50, legend='Away')
    train.query("PlayId == @playid and Team == 'home'")         .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=50, legend='Home')
    plt.title(f'Play # {playid}, {visitorteam} at {hometeam}')
    plt.legend()
    plt.show()

for p in plays:
    get_play_visualisation(p)

