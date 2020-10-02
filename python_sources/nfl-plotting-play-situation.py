#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl
# The following code visualizes the game to get a grasp of the different play situations. You only have to define the PlayId you are looking for. I added some features and functionalities to the football field made by Rob Mulla (https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position ). In addition I changed it to a layerwise plotting module.
# ## Plotting the field

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# ## Create The Football Field
# First we create the football field with the endzones (again, thanks to Rob for this awesome plot). I got rid of a few parameters and prepared the plot for various plotting layers but the main structure is still the same. 

# In[ ]:


PlayId = 20170907000395
def create_football_field(linenumbers=True, fifty_is_los=False):

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=(24, 12.66))
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    ez1 = patches.Rectangle((0, 0), 10, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
    ez2 = patches.Rectangle((110, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
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
            plt.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white')
            plt.text(x, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white')
            
    hash_range = range(11, 110)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')
    
    return fig, ax

create_football_field()
plt.show()


# # Adding Players and their motion information
# This is the most useful part to get an overview of the play situation. The players are plotted with their orientation and their moving direction. The length of the moving direction arrow differs to visualize the speed of the sepcific player. In addition the position of the player is shown. 

# In[ ]:


def add_players(PlayId, show_position=True, show_orientation=True, show_motion_direction=True):
    play_information = train.query("PlayId == @PlayId")
    train.query("PlayId == @PlayId and Team == 'away'").plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=200, legend='Away')
    train.query("PlayId == @PlayId and Team == 'home'").plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=200, legend='Home')
    
    if show_position:
        for index, row in play_information.iterrows():
            plt.annotate(row['Position'], (row['X']-0.5, row['Y']-0.5), color='white')
    
    if show_orientation:
        for index, row in play_information.iterrows():
            x_arrow = 1.25*np.sin(np.deg2rad(row['Orientation']))
            y_arrow = 1.25*np.cos(np.deg2rad(row['Orientation']))
            plt.arrow(row['X'], row['Y'], x_arrow, y_arrow, head_width=0.5, length_includes_head=True,
                      color = 'orange' if row['Team'] == 'away' else 'blue')

    if show_motion_direction:
        for index, row in play_information.iterrows():
            x_arrow = -row['S']*np.sin(np.deg2rad(row['Dir']))
            y_arrow = row['S']*np.cos(np.deg2rad(row['Dir']))
            plt.arrow(row['X'], row['Y'], x_arrow, y_arrow, head_width=0.5, length_includes_head=True,
                      head_starts_at_zero=True, 
                      facecolor='orange' if row['Team'] == 'away' else 'blue')
    plt.title('PlayId: ' + str(PlayId))

fig, ax = create_football_field()
add_players(PlayId)
plt.show()


# # Highlight lines and more information
# Some additional information to the play situation is presented in the last function, e.g. the line of scrimmage, play direction and the current game score. But at the end we are interested in the actual achieved yards... so they are presented as a red bar starting from the line of scrimmage. 

# In[ ]:


def highlight_lines(PlayId, show_scrimmage_line=True, show_play_direction=True, show_yards=True, show_teams=True):
    
    play_information = train.query("PlayId == @PlayId")
    yl = play_information['YardLine'].tolist()[0]
    yl_left = yl + 10 
    yl_right = 110 - yl
    distance_left_scrimmage = 0
    distance_right_scrimmage = 0
    for index, rows in play_information.iterrows():
        distance_left_scrimmage += abs(rows['X'] - yl_left)
        distance_right_scrimmage += abs(rows['X'] - yl_right)

    if distance_left_scrimmage <= distance_right_scrimmage:
        yl = yl_left
    else:
        yl = yl_right

    if show_scrimmage_line:
        plt.plot([yl, yl], [0, 53.3], color='yellow')
        plt.text(yl + 1, 46, 'Line of scrimmage', color='yellow', fontsize=20)
        
    direction = play_information['PlayDirection'].tolist()[0]
    if show_play_direction:
        if 'right' in direction:
            plt.arrow(55, 2, 10, 0, head_width=0.5, length_includes_head=True, color='red')
        else:
            plt.arrow(65, 2, -10, 0, head_width=0.5, length_includes_head=True, color='red')
        plt.text(54, 3, 'Play direction', color='red', fontsize=20)
    
    if show_yards:
        if 'right' in direction:
            sign = 1
        else: 
            sign = -1
        yard_strip = patches.Rectangle((yl, 52.3), sign*play_information['Yards'].tolist()[0], 1, linewidth=0.1,  facecolor='red')
        ax.add_patch(yard_strip)
        plt.text(yl + sign*(3**-sign), 50, 'Yards: {}' .format(play_information['Yards'].tolist()[0]), color='red', fontsize=20)
    
    if show_teams:
        visitor = (play_information['VisitorTeamAbbr'].tolist()[0],
                   play_information['VisitorScoreBeforePlay'].tolist()[0],
                  'orange')
        home = (play_information['HomeTeamAbbr'].tolist()[0], 
                play_information['HomeScoreBeforePlay'].tolist()[0],
                'blue')

        if play_information['PossessionTeam'].tolist()[0] == visitor[0]:
            if 'right' in direction:
                team_l, score_l, color_l = visitor
                team_r, score_r, color_r = home
            else:
                team_r, score_r, color_r = visitor
                team_l, score_l, color_l = home
        else:
            if 'right' in direction:
                team_r, score_r, color_r = visitor
                team_l, score_l, color_l = home
            else:
                team_l, score_l, color_l = visitor
                team_r, score_r, color_r = home
        
        plt.text(1, 26, '{0}: {1}' .format(team_l, score_l), color=color_l, fontsize=30)
        plt.text(111, 26, '{0}: {1}' .format(team_r, score_r), color=color_r, fontsize=30)

fig, ax = create_football_field()
add_players(PlayId)
highlight_lines(PlayId)
plt.show()


# ## Reference
# 1. https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position
