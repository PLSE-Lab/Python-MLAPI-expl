#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# thanks to my dude Savvas Tjortjoglou for creating this sweet blog
# http://savvastjortjoglou.com/nba-shot-sharts.html

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df_full = pd.read_csv('../input/data.csv')
df_sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


print(df_full.head())
print('\n')
print(df_sample.head())


# In[ ]:


df_train = df_full[df_full['shot_made_flag'].notnull()]
print('full dataset size:', df_full.shape)
print('train dataset size:', df_train.shape)


# In[ ]:


# creating a basic scatter plot to show the data
sns.set_style('white')
sns.set_color_codes()
plt.figure(figsize=(12,11))
plt.scatter(df_train['loc_x'],df_train['loc_y'])
# note that x-axis values are the inverse of what they actually should be
# only showing shots up to 50 feet away
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.show()


# In[ ]:


# time to add basketball court lines for context
from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

# let's draw the court
plt.figure(figsize=(12,11))
plt.scatter(df_train['loc_x'],df_train['loc_y'])
draw_court(outer_lines=True)

# and now draw the shots
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()


# In[ ]:


# ^ so he basically shoots from everywhere but doesn't like to shoot just inside the 3-point
# line. He also doesn't like the left corner too much.

# now time for a heatmap of his FGA since so many of the dots are overlapping
cmap=plt.cm.YlOrRd_r 

# n_levels sets the number of contour lines for the main kde plot
joint_shot_chart = sns.jointplot(df_train['loc_x'],df_train['loc_y'], stat_func=None,
                                 kind='kde', space=0, color=cmap(0.1),
                                 cmap=cmap, n_levels=50)

joint_shot_chart.fig.set_size_inches(12,11)

# A joint plot has 3 Axes, the first one called ax_joint 
# is the one we want to draw our court onto and adjust some other settings
ax = joint_shot_chart.ax_joint
draw_court(ax)

# Adjust the axis limits and orientation of the plot in order
# to plot half court, with the hoop by the top of the plot
ax.set_xlim(-250,250)
ax.set_ylim(422.5, -47.5)

# Get rid of axis labels and tick marks
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelbottom='off', labelleft='off')

# Add a title
ax.set_title('Kobe Bryant Career FGA', 
             y=1.2, fontsize=18)


plt.show()


# In[ ]:


# the above image has been flipped from the previous views and the color map
# is much more helpful to know kobe's preferences. By far, he loves to shoot 
# at the rim. He also likes the remainder of the paint

# it's amazing how balanced his shot chart is. He can operate from the left or right
# side of the court.
