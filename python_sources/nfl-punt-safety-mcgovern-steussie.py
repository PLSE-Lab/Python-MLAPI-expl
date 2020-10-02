#!/usr/bin/env python
# coding: utf-8

# # NFL Punt Rule Change Analysis
# ## Kevin McGovern and Todd Steussie
# This notebook contains visualizations used to evaluate NFL punt plays in 2016 and 2017 and support a suggested rule change.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ipywidgets import interactive
from IPython.display import HTML

from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_notebook, show, push_notebook
from bokeh.models import CustomJS, Slider, Button, BoxAnnotation, LabelSet, LinearAxis, Range1d
from bokeh.models.markers import Triangle
from bokeh.layouts import row, widgetbox
import bokeh.models as bmo

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

output_notebook()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/nfl-punt-analysis-data-output"))

# Load full data set
puntDF = pd.read_csv('../input/nfl-punt-analysis-data-output/playerMvmt-level-data.csv')
playLvlDF = pd.read_csv('../input/nfl-punt-analysis-data-output/play-level-data.csv')
playLvlDF.loc[:,('Block Type')] = ['Blindside' if x == 'TRUE' else 'Non-Blindside' for x in playLvlDF['blindsideBlock']]
playLvlCleanDF = playLvlDF[(playLvlDF['Primary_Impact_Type'] != 'Unclear') & (playLvlDF['InjCtrlFlag'] == 'Injury')]


# In[ ]:


"""Return a data frame

Function to return a data frame that contains a set of movements over the entire life of a play, for a specific play and set of actors

Parameters: PlayID, actors (list of GSISIDs)
"""
def get_data_over_time(playID = 3746, actors = ['27654.0','33127.0']):
    limitedSample = puntDF.loc[(puntDF['PlayID'] == playID)]
    return limitedSample[limitedSample['GSISID_y'].isin(actors)]

"""Return a data frame

Function to return a data frame that contains a single play at a particular point in time for all actors

Parameters: PlayID, timeNum (integer describing deciseconds from start of play)
"""
def update_sample(playID = 3746, timeNum = 1):
    return puntDF.loc[(puntDF['PlayID'] == playID) & (puntDF['TimeNum'] == timeNum)]

"""Return a data frame

Function to return a data frame that contains only the injured and primary actor. Used for drawing distance line between players

Parameters: limitedSample (data frame)
"""
def update_interactors(limitedSample = puntDF.loc[(puntDF['PlayID'] == 3746)]):
    return limitedSample[(limitedSample['GSISID_x'] == limitedSample['GSISID_y']) | (limitedSample['Primary_Partner_GSISID'] == limitedSample['GSISID_y'])]

"""Return a data frame

Function to return a data frame that contains only the data for a specific play

Parameters: playID
"""
def get_play(playID = 3746):
    return playLvlDF.loc[(playLvlDF['PlayID'] == playID)]

"""Return a string

Function to return a string that explains what happened on a particular play

Parameters: playID
"""
def generatePlayText(playID = 3746):
    playCalled = get_play(playID)
    return str(playCalled['GSISID'].iloc[0]) + ' was injured in ' + str(playCalled['Season_Year'].iloc[0]) + '. The contact was made by ' + playCalled['Player_Activity_Derived'].iloc[0] + ' on ' + playCalled['Primary_Impact_Type'].iloc[0] + ' contact.'

# Create a dict of impacts
data = {
    '1': [1683, 30665.0, 90, 110, 'Non-Earhole'],
    '2': [3746, 29510.0, 265, 285, 'Non-Earhole'],
    '3': [1045, 28698.0, 175, 195, 'Non-Earhole'],
    '4': [1976, 32807.0, 190, 210, 'Earhole'],
    '5': [2341, 32998.0, 140, 160, 'Earhole'],
    '6': [2792, 31317.0, 185, 205, 'Earhole'],
    '7': [1045, 31756.0, 140, 160, 'Earhole'],
    '8': [1088, 31999.0, 335, 355, 'Earhole'],
    '9': [1526, 30789.0, 125, 145, 'Earhole'],
    '10': [1683, 25503.0, 125, 145, 'Earhole'],
    '11': [2489, 32851.0, 120, 140, 'Earhole'],
    '12': [2587, 31059.0, 160, 180, 'Earhole'],
    '13': [2764, 31930.0, 105, 125, 'Earhole'],
}

# Create a dataframe from above dictionary
impactPlays = pd.DataFrame.from_dict(data, orient='index',
                       columns=['PlayID', 'GSISID_y', 'ImpactStart', 'ImpactEnd', 'Block Type'])

# merge time-base puntDF with impactPlays df created above
impactPlayTime = pd.merge(impactPlays, puntDF, how="inner", on=['PlayID', 'GSISID_y'])

# only keep plays of windows where contact occurred
impactPlayTime = impactPlayTime.loc[(impactPlayTime['TimeNum'] >= impactPlayTime['ImpactStart']) & (impactPlayTime['TimeNum'] <= impactPlayTime['ImpactEnd'])]

# group time by every five deciseconds to smooth data
impactPlayTime['TimeFloor'] = impactPlayTime['TimeNum']//5

# group by newly created time dimension
impactPlayTime = impactPlayTime.groupby(['TimeFloor','PlayID','Block Type','GSISID_y'], as_index=False)['Speed'].mean()

# multiply time by five so new timenum aligns to old timenum
impactPlayTime['TimeNum'] = impactPlayTime['TimeFloor']*5


# In[ ]:


"""Return a statement

Function to update all charts when changes are made to notebook interactive elements. Will update the three charts at bottom of the page

Parameters: PlayID, timeNum (integer describing deciseconds from start of play)
"""
def update_plot(timeNum, playID):
    #update circles
    limitedSample = update_sample(playID, timeNum)
    x = limitedSample['x']
    y = limitedSample['y']
    colorMap = ['#ff0000' if x in(limitedSample['GSISID_x'].values) else '#0000ff' if x in(limitedSample['Primary_Partner_GSISID'].values) else '#545454' for x in limitedSample.GSISID_y]

    # update circle data in circle figure
    circlePlt.data_source.data['x'] = x
    circlePlt.data_source.data['y'] = y
    circlePlt.data_source.data['fill_color'] = colorMap
    
    #update lines using similar method as above
    distanceLine = update_interactors(limitedSample)
    line_x = distanceLine['x']
    line_y = distanceLine['y']
    lines.data_source.data['x'] = line_x
    lines.data_source.data['y'] = line_y
    
    # push notebook changes to chart
    push_notebook(handle=bokeh_handle)
    
    # Update distance chart
    distanceDF = get_data_over_time(playID, actors = [get_play(playID)['GSISID'].iloc[0]])
    x = distanceDF['TimeNum']
    y = distanceDF['Distance']

    # update line data in line figure
    distanceLines.data_source.data['x'] = x
    distanceLines.data_source.data['y'] = y
    
    push_notebook(handle=bokeh_handle2)
    
    # Update speed chart
    distanceDF = get_data_over_time(playID, actors = [get_play(playID)['GSISID'].iloc[0]])
    x = distanceDF['TimeNum']
    y = distanceDF['Speed']
    injSpeedlines.data_source.data['x'] = x
    injSpeedlines.data_source.data['y'] = y

    # do the same thing as above but for primary partner
    distanceDF = get_data_over_time(playID, actors = [get_play(playID)['Primary_Partner_GSISID'].iloc[0]])
    x = distanceDF['TimeNum']
    y = distanceDF['Speed']
    primActorSpeedlines.data_source.data['x'] = x
    primActorSpeedlines.data_source.data['y'] = y
    
    push_notebook(handle=bokeh_handle3)
    
    return generatePlayText(playID)


# In[ ]:


"""Return a Bokeh figure

Function to create Bokeh x-y movement charts over time

Parameters: PlayID, actors (list of relevant players), figure (bokeh chart figure reference)
"""
def create_bokeh_play_x_y_lines(playID, actors, figure):

    # calculate base data set
    limitedSample = get_data_over_time(playID = playID, actors = actors)
    x = limitedSample[limitedSample['GSISID_y'] == float(actors[0])]['x']
    y = limitedSample[limitedSample['GSISID_y'] == float(actors[0])]['y']
    x2 = limitedSample[limitedSample['GSISID_y'] == float(actors[1])]['x']
    y2 = limitedSample[limitedSample['GSISID_y'] == float(actors[1])]['y']

    # add line renderers with color, width, and alpha
    figure.line(x, y, line_color='#b22222',line_width=3, alpha=0.8)
    figure.line(x2, y2, line_color='#4682b4',line_width=3, alpha=0.8)
    #plt2 = figure.circle(x, y, size=8, fill_color=color_map, alpha=0.8)

    # create boxes that segment the field
    low_box = BoxAnnotation(right=10, fill_alpha=0.2, fill_color='#555555')
    mid_box = BoxAnnotation(left=10, right=110, fill_alpha=0.2, fill_color='green')
    high_box = BoxAnnotation(left=110, fill_alpha=0.2, fill_color='#555555')

    # add a line used to create lines on field and a consistent window
    figure.line([10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,
              100,100,110,110,120,120],
             [55.5,0,0,55.5,55.5,0,0,55.5,55.5,0,0,55.5,55.5,0,0,55.5,
             55.5,0,0,55.5,55.5,0,0,0], line_color='grey')

    figure.add_layout(low_box)
    figure.add_layout(mid_box)
    figure.add_layout(high_box)

    return figure

"""Return a Bokeh figure

Function to create Bokeh line charts over time

Parameters: PlayID, actors (list of relevant players), X (x-axis string ref), Y (y-axis string ref), lineFig (bokeh chart figure reference), 
color (color of line), rightAxis (reference to right axis of chart), rightAxisName (name entered inside paren for chart axis)
"""
def create_bokeh_lines(playID, actor, X, Y, lineFig, color='black', rightAxis=None, rightAxisName=""):
    # calculate base data set
    distanceDF = get_data_over_time(playID = playID, actors = [actor])
    distanceDF['TimeFloor'] = distanceDF['TimeNum']//5
    distanceDF = distanceDF.groupby(['TimeFloor','PlayID'], as_index=False)[Y].mean()
    distanceDF['TimeNum'] = distanceDF['TimeFloor']*5
    x = distanceDF[X]
    y = distanceDF[Y]

    # return the line renderer with a color, width, and alpha
    if rightAxis==None:
        return lineFig.line(x, y, line_color = color, line_width = 2, alpha = 0.8)
    else:
        # Setting the second y axis range name and range
        minval = y.min()*.8 if y.min()>0 else -y.max()*.05 if y.min()==0 else y.min()*1.2
        lineFig.extra_y_ranges = {rightAxis: Range1d(minval,y.max()*1.2)}
        lineFig.add_layout(LinearAxis(y_range_name=rightAxis, axis_label=Y + ' ('+ rightAxisName +')'), 'right')
        return lineFig.line(x, y, line_color = color, line_width = 2, alpha = 0.8, y_range_name=rightAxis)


# In[ ]:


# Create figure plot
distanceLineFig = figure(plot_width=800, plot_height=400)
# Create distance lines
distanceLines = create_bokeh_lines(3746, '27654.0', 'TimeNum', 'Distance', distanceLineFig)

# Create injured player and primary actor speed charts
speedLineFig = figure(plot_width=800, plot_height=400)
injSpeedlines = create_bokeh_lines(3746, '27654.0', 'TimeNum', 'Speed', speedLineFig, 'red')
primActorSpeedlines = create_bokeh_lines(3746, '33127.0', 'TimeNum', 'Speed', speedLineFig, 'blue')


# # Play Analysis
# Before we decided on a rule change, we reviewed each play to determine the players' interactions and activities prior to the injury. The analysis that preceeded our rule recommendation is listed below.
# 
# Note: we excluded one injury from our analysis where the cause was listed as 'unclear', as it did not inform the high-level analysis.

# The breakdown of primary impact shows that the vast marjority of concussions occurred during either helmet-to-body or helmet-to-helmet collisions.

# In[ ]:


## Primary Impact
plt.figure(figsize=(10,5))
sns.set_style("darkgrid")
sns.set_palette("YlGnBu", n_colors=3)
countplt = sns.countplot(x="Primary_Impact_Type", data=playLvlCleanDF)


# The source of the injury was roughly split between tackling and blocking.

# In[ ]:


## Primary Impact
plt.figure(figsize=(10,5))
sns.set_style("darkgrid")
sns.set_palette("YlGnBu", n_colors=4)
countplt2 = sns.countplot(x="Player_Activity_Derived", data=playLvlCleanDF)


# ## Of the plays provided by the NFL for this competition, a significant percentage of concussions occurred when the player was blocking the opponent with helmet-to-body contact.

# In[ ]:


uniform_data = pd.pivot_table(playLvlCleanDF, values='PlayID', index=['Player_Activity_Derived'], columns=['Primary_Impact_Type'], aggfunc='count')
plt.figure(figsize=(10,8))
ax = sns.heatmap(uniform_data, annot=True, cmap="YlGnBu")


# ### Some trends start to emerge but it wasn't until we looked at game film that we saw a theme.
# Analysis of the game film showed what was not immediately clear from the data. A significant number of the injuries occured when a player blocked a "defenseless player". Although applied to a different situational context, Article 7 of the NFL Rulebook establishes criteria for defenseless player status, stating that a player should be considered defenseless until "the player is capable of avoiding or warding off the impending contact of an opponent". These types of plays are also commonly referred to as blindside blocks.

# In[ ]:


## Block type on blocking plays
blockedDF = playLvlCleanDF[(playLvlCleanDF['Player_Activity_Derived'] == 'Blocked') | (playLvlCleanDF['Player_Activity_Derived'] == 'Blocking')]
uniform_data = pd.pivot_table(blockedDF, values='PlayID', index=['Player_Activity_Derived'], columns=['Block Type'], aggfunc='count')
plt.figure(figsize=(10,8))
ax2 = sns.heatmap(uniform_data, annot=True, cmap="YlGnBu")


# ## Of the 37 plays that lead to a concussion in 2016 and 2017, 11 of these were caused by a block occurring when the opponent does not have an opportunity to anticipate the impact. This is often a result of the opponent being oriented in a different direction, and as a result, was not aware of the imminent collision.
# 
# ## If adopted, this rule change would impact 65% of the injuries resulting from blocking on punt plays (11 of 17).
# 

# In[ ]:


HTML('<video width="600" height="400" controls> <source src="http://a.video.nfl.com//films/vodzilla/153252/44_yard_Punt_by_Justin_Vogel-n7U6IS6I-20181119_161556468_5000k.mp4" type="video/mp4"></video>')


# # Suggested Rule Change
# ## Rule: Penalize blocks where the opponent does not have an opportunity to anticipate the impact. These types of blocks occur when the opponent is oriented in a different direction, and as a result is not aware of an imminent collision. This rule change would extend defenseless player protection to all participants on punt plays.
# 
# Under current NFL rules, it is illegal to initiate contact against a player who is in a defenseless posture.  As our analysis has shown, a significant portion of injuries on punt plays have resulted from blocks against defenseless players. Although applied to a different situational context, Article 7 of the NFL Rulebook establishes criteria for defenseless player status, stating that a player is to be considered defenseless until "the player is capable of avoiding or warding off the impending contact of an opponent". 
# 
# It is our contention that emposing defenseless player protection status for all players during punt plays would significantly reduce injuries. We believe that this rule change is a logical extension of the protections already enacted to protect defenseless players. Adopting this rule change is likely to have a significant positive impact on player safety as previous rules protecting defenseless players, with minimal risk of negative externalities associated with changes to strategy or player behavior. Players running downfield that do not see an incoming player are particularly vulnerable to injury. Not only are these blocks dangerous to the defenseless player, but they often result in injuries to the player attempting the block as well.
# 

# ## Rule Change Example #1
# ### CAR-WAS 2016 Week 15 example of blindside block on the targeted player.
# The charts below show how in a 2016 week 15 Panthers vs. Redskins game, a player suffered a concussion while being blocked by a player running downfield.
# 

# In[ ]:


# Distance vs speed
LineFigPlay1 = figure(plot_width=800, plot_height=400, title="CAR-WAS Distance Between Impacted Players (blue) vs. Speed of Injured Player (red)")
# Create distance lines
LineFigPlay1Rtn = create_bokeh_lines(2341, '32007.0', 'TimeNum', 'Distance', LineFigPlay1, color='#008ac2')
LineFigPlay1.yaxis.axis_label = "Distance (yards)"
LineFigPlay1.xaxis.axis_label = "Play Time (milliseconds)"
LineFigPlay1.background_fill_color = "#efefef"
LineFigPlay1.background_fill_alpha = 0.5
#LineFigPlay1.legend.location = "top_right"

LineFigPlay1Rtn2 = create_bokeh_lines(2341, '32007.0', 'TimeNum', 'Speed', LineFigPlay1, color='#de425b', rightAxis='rightRange', rightAxisName='red')

LineFigPlay1show = show(row(LineFigPlay1), notebook_handle=True)


# In the chart above, we compared speed of the injured player and distance between the injured player and the impacting player. As you can see the player is running at a high speed from the 0.6 second mark until the point of contact at 1.5 seconds. The two players are significantly far apart but quickly converged due to the high speed of both players.

# In[ ]:


# Speed vs acceleration
# Create injured player and primary actor speed charts
speedLineFigPlay1 = figure(plot_width=800, plot_height=400, title="CAR-WAS Speed (red) vs. Acceleration (green) of Injured Player")
injSpeedlinesPlay1 = create_bokeh_lines(2341, '32007.0', 'TimeNum', 'Speed', speedLineFigPlay1, color='#de425b')
speedLineFigPlay1.yaxis.axis_label = "Speed"
speedLineFigPlay1.xaxis.axis_label = "Play Time (milliseconds)"
speedLineFigPlay1.background_fill_color = "#efefef"
speedLineFigPlay1.background_fill_alpha = 0.5

accelSpeedlinesPlay1 = create_bokeh_lines(2341, '32007.0', 'TimeNum', 'Acceleration', speedLineFigPlay1, color='#00FA9A', rightAxis='rightRange', rightAxisName='green')

Play1show = show(row(speedLineFigPlay1), notebook_handle=True)


# In the chart above we compare the speed of the injured player against the acceleration of the injured player. The injured player actually briefly decelerates, immediately prior to the moment of impact but not with nearly enough time to make a difference in his overall speed. He then then accelerates again as the two make contact at the 1.5 second mark. Because he makes contact at such a high speed, damage is maximized and he is immediately immobilized.

# In[ ]:


# Paths
XYFigPlay1 = figure(plot_width=800, plot_height=400, x_range=(0,120), y_range=(0, 53.3), title="CAR-WAS Path of Injured (red) and Impacting (blue) Players")
XYFigPlay1upd = create_bokeh_play_x_y_lines(playID = 2341, actors = ['32007.0', '32998.0'], figure = XYFigPlay1)
bokeh_handle6 = show(row(XYFigPlay1upd), notebook_handle=True)


# The above chart shows the path of the injured player (red) compared with the path of the impacting player (blue). The contact occurs downfield and you can see the injured player does not move far after the contact, while the blocking player runs through the block and off the field.

# In[ ]:


HTML('<iframe width="700" height="400" src="https://streamable.com/s/p7exm/mepwve" frameborder="0" allowfullscreen></iframe>')


# In[ ]:


HTML('<iframe width="700" height="400" src="https://streamable.com/s/qu3q4/azcmbw.mp4" frameborder="0" allowfullscreen></iframe>')


# View the video above to see the exact contact that led to the concussion.

# ### Let's compare the motion metrics from the play above with a median impact from a typical block on a punt play. The typical block represents the replacement scenario if the rule change is adopted.

# In[ ]:


# Speed vs acceleration
# Create injured player and primary actor speed charts
speedLineFigPlay3 = figure(plot_width=800, plot_height=400, title="NYG-KC Speed (red) vs. acceleration (green) of blocked player on Ideal Block")
injSpeedlinesPlay3 = create_bokeh_lines(1683, '25503.0', 'TimeNum', 'Speed', speedLineFigPlay3, color='#de425b')
speedLineFigPlay3.yaxis.axis_label = "Speed"
speedLineFigPlay3.xaxis.axis_label = "Play Time (milliseconds)"
speedLineFigPlay3.background_fill_color = "#efefef"
speedLineFigPlay3.background_fill_alpha = 0.5

accelSpeedlinesPlay3 = create_bokeh_lines(1683, '25503.0', 'TimeNum', 'Acceleration', speedLineFigPlay3, color='#00FA9A', rightAxis='rightRange', rightAxisName='green')

Play3show = show(row(speedLineFigPlay3), notebook_handle=True)


# On a play we would call an 'typical block' the above chart shows the speed and acceleration of a player engaging on a block. The contact with the player happens at roughly 0.9 seconds. There is a large deceleration period (green line) before engaging in the block, and once the two players release, the player accelerates again and begins to speed up to continue the remainder of the play.

# In[ ]:


# Speed vs acceleration
# Create injured player and primary actor speed charts
speedLineFigPlay32 = figure(plot_width=800, plot_height=400, title="NYG-KC Speed (red) vs. acceleration (green) of blocking player on Ideal Block")
injSpeedlinesPlay32 = create_bokeh_lines(1683, '30665.0', 'TimeNum', 'Speed', speedLineFigPlay32, color='#de425b')
speedLineFigPlay32.yaxis.axis_label = "Speed"
speedLineFigPlay32.xaxis.axis_label = "Play Time (milliseconds)"
speedLineFigPlay32.background_fill_color = "#efefef"
speedLineFigPlay32.background_fill_alpha = 0.5

accelSpeedlinesPlay32 = create_bokeh_lines(1683, '30665.0', 'TimeNum', 'Acceleration', speedLineFigPlay32, color='#00FA9A', rightAxis='rightRange', rightAxisName='green')

Play32show = show(row(speedLineFigPlay32), notebook_handle=True)


# For the blocked player, because he is able to anticipate the block he decelerates leading into the the 0.9 second contact period and is able to accelerate again after he is free from the block. This contact is significantly safer than contact where one player does not anticipate being blocked.

# In[ ]:


plt.figure(figsize=(10,8))
ax = sns.boxplot(x="Block Type", y="Speed", hue="Block Type",
                  data=impactPlayTime, palette="YlGnBu")


# The above chart shows the speed of the blocking player on the plays with an blindside block vs. the speed of player on what we considered ideal blocks during the point of contact. As you can see the speed of the player on blindside blocks is signficantly higher as he engages with a defenseless blocker.

# ## Rule Change Example #2
# ### NO-MIA 2017 Week 4 example of blindside block leading to concussion for the blocking player.
# 
# In the below example a Saints player makes a block against a Dolphins player. The contact leads to a concussion for the blocking player.

# In[ ]:


HTML('<video width="560" height="315" controls> <source src="http://a.video.nfl.com//films/vodzilla/153272/Haack_42_yard_punt-iP6aZSRU-20181119_165050694_5000k.mp4" type="video/mp4"></video>')


# In[ ]:


# Distance vs speed
LineFigPlay2 = figure(plot_width=800, plot_height=400, title="NO-MIA Distance Between Impacted Players (blue) vs. Speed of Injured Player (red)")
# Create distance lines
LineFigPlay2Rtn = create_bokeh_lines(2792, '33838.0', 'TimeNum', 'Distance', LineFigPlay2, color='#008ac2')
LineFigPlay2.yaxis.axis_label = "Distance (yards)"
LineFigPlay2.xaxis.axis_label = "Play Time (milliseconds)"
LineFigPlay2.background_fill_color = "#efefef"
LineFigPlay2.background_fill_alpha = 0.5
#LineFigPlay2.legend.location = "top_right"

LineFigPlay2Rtn2 = create_bokeh_lines(2792, '33838.0', 'TimeNum', 'Speed', LineFigPlay2, color='#de425b', rightAxis='rightRange', rightAxisName='red')

LineFigPlay2show = show(row(LineFigPlay2), notebook_handle=True)


# The above chart shows that both distance between the injured and impacting player and the speed of the injured player are maximized at roughtly the same time. The moment of impact occurs around the 1.9 second mark and leads to a quick drop in the speed of the injured player. The contact from this is most signficant because the high speed causes the players to collide at an injury-inducing pace.

# In[ ]:


# Speed vs acceleration
# Create injured player and primary actor speed charts
speedLineFigPlay2 = figure(plot_width=800, plot_height=400, title="NO-MIA Speed (red) vs. Acceleration (green) of Injured Player")
injSpeedlinesPlay2 = create_bokeh_lines(2792, '33838.0', 'TimeNum', 'Speed', speedLineFigPlay2, color='#de425b')
speedLineFigPlay2.yaxis.axis_label = "Speed"
speedLineFigPlay2.xaxis.axis_label = "Play Time (milliseconds)"
speedLineFigPlay2.background_fill_color = "#efefef"
speedLineFigPlay2.background_fill_alpha = 0.5

accelSpeedlinesPlay2 = create_bokeh_lines(2792, '33838.0', 'TimeNum', 'Acceleration', speedLineFigPlay2, color='#00FA9A', rightAxis='rightRange', rightAxisName='green')

Play2show = show(row(speedLineFigPlay2), notebook_handle=True)


# The above chart shows speed and acceleration compared for the injured player. Acceleration increases rapidly when the ball is kicked as the player initially backpedals to block. As the ball goes down field, the player accelerates again and maintains that speed until the point of contact near the sideline down field. At the moment of impact (190-200) the player quickly decelerates as he quickly comes to a stop after making contact with the player running downfield.
# 
# ### This example shows that the blocking player, not the unsuspecting player, sustains the injury. Blocks on players who are caught off guard often occur at a high speed  and endanger both the player being blocked and the blocking player.

# In[ ]:


## Create X-Y Chart showing path of impacted players

XYFigPlay2 = figure(plot_width=800, plot_height=400, x_range=(0,120), y_range=(0, 53.3), title="NO-MIA Path of Injured (red) and Impacting (blue) Players")
XYFigPlay2 = create_bokeh_play_x_y_lines(playID = 2792, actors = ['33838.0', '31317.0'], figure = XYFigPlay2)
bokeh_handle4 = show(row(XYFigPlay2), notebook_handle=True)


# In[ ]:


HTML('<iframe width="700" height="400" src="https://streamable.com/s/vdd1h/djxdgd" frameborder="0" allowfullscreen></iframe>')


# In[ ]:


HTML('<iframe width="700" height="400" src="https://streamable.com/s/rgdx5/jewmgc.mp4" frameborder="0" allowfullscreen></iframe>')


# Please note: the video is shown from the opposite side of the player tracking graphs above

# The two examples above illustrate particular situations where a low block and a low dive resulted in concussions. It is our belief that these concussions could have been avoided had the players not gone to the ground on the play.

# One note of caution, because the data we evaluated only used the 2016 and 2017 seasons, it is possible that this sample size is not representative of a more comprehensive dataset. To fully test this theory, additional seasons would have to be evaluated.

# ## Interactive Section
# The charts below allow the selection of other plays to navigate through time and view different results. The controls to change inputs are located immediately following the charts.
# 
# ### Please note: to run this workbook with fully interactivity, you must first fork the notebook and run all cells. The last cells has interactive elements for PlayID and Time during play 
# 
# 

# ### Distance between injured player and primary actor

# In[ ]:


bokeh_handle2 = show(row(distanceLineFig), notebook_handle=True)


# ### Speed of injured player (red) vs. primary actor (blue)

# In[ ]:


bokeh_handle3 = show(row(speedLineFig), notebook_handle=True)


# ### Play Motion Chart

# In[ ]:


# Create chart for plays
# calculate base data set
limitedSample = update_sample(playID = 3746)
x = limitedSample['x']
y = limitedSample['y']

# get data for distance line
distanceLine = update_interactors(limitedSample)
line_x = distanceLine['x']
line_y = distanceLine['y']

fig = figure(plot_width=800, plot_height=400, x_range=(0,120), y_range=(0, 53.3))

color_map = ['#ff0000' if x in(limitedSample['GSISID_x'].values) else '#0000ff' if x in(limitedSample['Primary_Partner_GSISID'].values) else '#545454' for x in limitedSample.GSISID_y]

# add a circle renderer with a size, color, and alpha
lines = fig.line(line_x, line_y, line_color='black',line_width=2, alpha=0.6)
circlePlt = fig.circle(x, y, size=8, fill_color=color_map, alpha=0.8)

# add boxes to either end represented endzones
low_box = BoxAnnotation(right=10, fill_alpha=0.2, fill_color='#555555')
mid_box = BoxAnnotation(left=10, right=110, fill_alpha=0.2, fill_color='green')
high_box = BoxAnnotation(left=110, fill_alpha=0.2, fill_color='#555555')

# add lines to field
fig.line([10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,
          100,100,110,110,120,120],
         [55.5,0,0,55.5,55.5,0,0,55.5,55.5,0,0,55.5,55.5,0,0,55.5,
         55.5,0,0,55.5,55.5,0,0,0], line_color='grey')

fig.add_layout(low_box)
fig.add_layout(mid_box)
fig.add_layout(high_box)

bokeh_handle = show(row(fig), notebook_handle=True)


# ### Change the time or play viewed above (must edit/fork the notebook first)

# In[ ]:


interactive(update_plot, playID = playLvlDF[playLvlDF['InjCtrlFlag'] == 'Injury']['PlayID'].sort_values().values, timeNum = (0,350))


# In[ ]:




