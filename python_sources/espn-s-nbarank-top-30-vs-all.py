#!/usr/bin/env python
# coding: utf-8

# <b>This analysis takes a look at how the NBA's brightest stars of the 2014-2015 season stacked up with the rest of the league across various metrics.</b>
# 
# <em>Disclaimer: Data science baby's first attempt at analysis, harsh yet gentle criticism warmly welcome</em>
# 
# Every year ESPN releases it's famous <a href='http://www.espn.com/nba/story/_/id/11754398/#nbarank-2014-players-1-500' target='_blank'>#NBARank results</a> to predict players' relative performances in the upcoming season. Conveniently for us, this provides us just the group of individuals we can give the title of "stars", who we can put under the microscope and scrutinize for any shortcomings we uncover through a potpourri of metrics (seen below). I decided to use the top 30 players on #NBARank to put in that "star" group, which averages out to 1 per NBA team.
# 
# 1. <a href="#section1">Offensive FG% vs. Defensive FG%</a>
# * <a href="#section2">Average Shot Distance vs. Field Goal %</a>
# * <a href="#section3">Who's Clutch?</a>
# * <a href="#section4">A Closer Look at the Clutchness of the Top 30</a>
# * <a href="#section5">Lockdown Defenders</a>
# * <a href="#section6">Defender Distance vs FG%</a>

# <h1><a id="section1"></a>Offensive FG% vs. Defensive FG%</h1>

# Here we can compare players not only on their typical offensive field goal percentage, but also the field goal percentage of players on the defensive end. By this I mean the amount of field goals made when a player is the closest defender to the shot divided by the amount of field goals attempted when the player is the closest defender to the shot.

# In[ ]:


import pandas as pd
import numpy as np

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

data = pd.read_csv('../input/shot_logs.csv')

playerIDList= list(data['player_id'].unique())
defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())
attackerIDList = list(data['player_id'].unique())

playerData = pd.DataFrame(index=playerIDList, columns=['player', 'made', 'missed', 'fg_percentage', 
                                                       'made_against', 'missed_against', 'fg_percentage_against'])

for defenderID in defenderIDList:
    name= data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['CLOSEST_DEFENDER'].iloc[0]
    made = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='made')
    missed = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='missed')
    percentage = made/(made+missed)
    playerData.at[defenderID, 'player'] = name
    playerData.at[defenderID, 'made_against'] = made
    playerData.at[defenderID, 'missed_against'] = missed
    playerData.at[defenderID, 'fg_percentage_against'] = percentage

for attackerID in attackerIDList:
    made = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='made')
    missed = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='missed')
    percentage = made/(made+missed)
    playerData.at[attackerID, 'made'] = made
    playerData.at[attackerID, 'missed'] = missed
    playerData.at[attackerID, 'fg_percentage'] = percentage
    
newPlayerData = playerData.sort_values('fg_percentage_against')
newPlayerData2 = newPlayerData.drop(newPlayerData[newPlayerData.missed_against < 200].index)


# <h2>(Most of) The League</h2>

# In[ ]:


newPlayerData2


# In[ ]:


ESPNRankTop30 = ['James, LeBron', 'Paul, Chris', 'Davis, Anthony', 'Westbrook, Russell', 'Griffin, Blake', 'Curry, Stephen',
                'Love, Kevin', 'Durant, Kevin', 'Harden, James', 'Howard, Dwight', 'Anthony, Carmelo', 'Noah, Joakim',
                'Aldridge, LaMarcus', 'Gasol, Marc', 'Parker, Tony', 'Lillard, Damian', 'Nowitzki, Dirk', 'Wall, John',
                'Cousins, DeMarcus', 'Bosh, Chris', 'Duncan, Tim', 'Jefferson, Al', 'Irving, Kyrie', 'Leonard, Kawhi', 
                'Ibaka, Serge', 'Horford, Al', 'Dragic, Goran', 'Rose, Derrick', 'Lowry, Kyle', 'Drummond, Andre']
newPlayerData3 = newPlayerData[newPlayerData['player'].isin(ESPNRankTop30)]
newPlayerData3 = newPlayerData3.sort_values('player')
newPlayerData3['ranking'] = 0
for i in range(len(ESPNRankTop30)):
    newPlayerData3.loc[newPlayerData3['player'] == ESPNRankTop30[i], 'ranking'] = str(i+1)


# <h2>#NBARank's Top 30</h2>

# In[ ]:


newPlayerData3


# In[ ]:


line = Scatter(
            x= [0,1],
            y= [0,1],
            marker = dict(
                size=1,
                color='rgba(200, 200, 200, .5)'
            ),
            name = "Line of Neutrality"
        )

trace1 = Scatter(
            x=newPlayerData2['fg_percentage'],
            y=newPlayerData2['fg_percentage_against'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(132, 123, 255, .9)',
                line = dict(
                    width = 2,
                )
            ), 
            name='League',
            text= newPlayerData2['player']
        )

trace2 = Scatter(
            x=newPlayerData3['fg_percentage'],
            y=newPlayerData3['fg_percentage_against'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(255, 123, 132, .9)',
                line = dict(
                    width = 2,
                ),
            ),
            name='#NBARank Top 30',
            text = newPlayerData3['player'] + ' (#' + newPlayerData3['ranking'] + ')'
        )

data = [line, trace1, trace2]

layout = Layout(
    hovermode = 'closest',
    annotations=Annotations([
        Annotation(
            x=0.5004254919715793,
            y=-0.16191064079952971,
            showarrow=False,
            text='Made Field Goal %',
            xref='paper',
            yref='paper'
        ),
        Annotation(
            x=-0.05944728761514841,
            y=0.4714285714285711,
            showarrow=False,
            text='Allowed Field Goal %',
            textangle=-90,
            xref='paper',
            yref='paper'
        )
    ]),
    autosize=True,
    margin=Margin(
        b=100
    ),
    title='Made Vs. Allowed FG%',
    xaxis=XAxis(
        autorange=False,
        range=[0.35, 0.72],
        type='linear'
    ),
    yaxis=YAxis(
        autorange=False,
        range=[0.35, 0.55],
        type='linear'
    )
)

graph = Figure(data=data, layout=layout)
iplot(graph)


# Here we have the plot comparing offensive FG% with defensive FG% among all players in the #NBARank Top 30 along with players in the rest of the league who have forced at least 200 missed shots (around 150 players). The gray line marks the "Line of Neutrality," where a player along that line essentially allows exactly the proportion of field goals as they make. Players above the line let in a higher proportion of field goals on defense than they make on offense, implying that they may be a liability, and players below the line make a higher proportion of field goals on offense than they let in on defense, which is obviously desirable. The further a player is from the Line of Neutrality, the larger the scale of their good/bad-ness. Plotly plots have the nice feature that you can isolate each subset by double clicking its title on the legend pane seen on the right hand side of the plot, so go ahead and give that a go ya crazy goof.
# 
# We can see from the get go that the stars don't vastly stand out from the rest of the league in this metric. All the Top 30 players (seen in red dots as opposed to blue) are kind of jumbled up around the center with the rest of the league's players. Those among the stars that stand out the most are Anthony Davis (made 54.3% of field goals, let in 40.6%), Stephen Curry, Al Horford, and Dwight Howard. However, at least in this metric, even these standout stars pale in comparison to the likes of not-Top 30 players such as Deandre Jordan, Tyson Chandler, and Rudy Gobert. Tony Allen also has his own impressive spot on the plot, with the very low Allowed FG% of 35.5%.
# 
# Also standing out among the stars in a not-so-desirable way are Dirk Nowitzski and Carmelo Anthony, who find themselves at the far end on the wrong side of the Line of Neutrality. Interestingly enough, they are just a couple players within a surprisingly large group of stars who have let in a higher proportion of field goals on defense than they make on offense. Luckily for them they have the likes of Channing Frye (made 39.1% of field goals, let in 51.2%), Trevor Ariza, and Trey Burke to keep them from looking too terribly in this metric.

# <h1><a id="section2"></a>Average Shot Distance vs. Field Goal %</h1>

# In this section we take a look at players' field goal percentage in accordance with their average shot distance, and once again see if the #NBARank Top 30 set themselves apart form the rest of the league in any way in the resulting plot.

# In[ ]:


data = pd.read_csv('../input/shot_logs.csv')
defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())
attackerIDList = list(data['player_id'].unique())

playerData = pd.DataFrame(index=playerIDList, columns=['player', 'made', 'missed', 'fg_percentage', 
                                                       'fg_distance'])

for attackerID in attackerIDList:
    name= data[(data['player_id'] == attackerID)]['player_name'].iloc[0]
    spacePos = name.find(' ')
    firstname = name[0].upper() + name[1:spacePos]
    lastname = name[spacePos+1].upper() + name[spacePos+2:]
    name = firstname + ' ' + lastname
    made = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='made')
    missed = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='missed')
    percentage = made/(made+missed)
    averageDist = np.mean(data[(data['player_id'] == attackerID)]['SHOT_DIST'])
    playerData.at[attackerID, 'player'] = name
    playerData.at[attackerID, 'made'] = made
    playerData.at[attackerID, 'missed'] = missed
    playerData.at[attackerID, 'fg_percentage'] = percentage
    playerData.at[attackerID, 'fg_distance'] = averageDist
    
newPlayerData = playerData.sort_values('fg_distance', ascending=False)
newPlayerData2 = newPlayerData.drop(newPlayerData[newPlayerData.made < 200].index)
newPlayerData2


# If you don't mind, I'd just like to take a moment to acknowledge how Kyle Korver's <em><b>average</b></em> field goal distance is just over the distance from the corner of the 3 point line to the basket. You go Kyle! <img src='http://www.ajc.com/rf/image_lowres/Pub/p8/AJC/2017/03/03/Images/030417%20hawks%20photos%20CC27.JPG' href="kyle" width='300px'>

# In[ ]:


import plotly

import plotly.plotly as py
from plotly.graph_objs import *
from ipywidgets import widgets 
from IPython.display import display, clear_output, Image
from plotly.graph_objs import *
from plotly.widgets import GraphWidget

ESPNRankTop30 = ['Lebron James', 'Chris Paul', 'Anthony Davis', 'Russell Westbrook', 'Blake Griffin', 'Stephen Curry',
                'Kevin Love', 'Kevin Durant', 'James Harden', 'Dwight Howard', 'Carmelo Anthony', 'Joakim Noah',
                'Lamarcus Aldridge', 'Marc Gasol', 'Tony Parker', 'Damian Lillard', 'Dirk Nowtizski', 'John Wall',
                'Demarcus Cousins', 'Chris Bosh', 'Tim Duncan', 'Al Jefferson', 'Kyrie Irving', 'Kawhi Leonard', 
                'Serge Ibaka', 'Al Horford', 'Goran Dragic', 'Derrick Rose', 'Kyle Lowry', 'Andre Drummond']

trace1 = Scatter(
            x=newPlayerData2['fg_distance'],
            y=newPlayerData2['fg_percentage'],
            mode = 'markers',
            marker = dict(
                size = newPlayerData2['made']/20,
                color = 'rgba(132, 123, 255, .9)',
                line = dict(
                    width = 2,
                ),
            ),
            name='League',
            text = newPlayerData2['player']
        )

newPlayerData3 = newPlayerData2[newPlayerData2.player.isin(ESPNRankTop30)]
trace2 = Scatter(
            x=newPlayerData3['fg_distance'],
            y=newPlayerData3['fg_percentage'],
            mode = 'markers',
            marker = dict(
                size = newPlayerData3['made']/20,
                color = 'rgba(255, 123, 132, .9)',
                line = dict(
                    width = 2,
                ),
            ),
            name='#NBARank Top 30',
            text = newPlayerData3['player']
        )

data = [trace1, trace2]

layout = Layout(
    hovermode = 'closest',
    annotations=Annotations([
        Annotation(
            x=0.5004254919715793,
            y=-0.16191064079952971,
            showarrow=False,
            text='Average Shot Distance (Feet)',
            xref='paper',
            yref='paper'
        ),
        Annotation(
            x=-0.06944728761514841,
            y=0.4714285714285711,
            showarrow=False,
            text='Field Goal %',
            textangle=-90,
            xref='paper',
            yref='paper'
        )
    ]),
    autosize=True,
    margin=Margin(
        b=100
    ),
    title='Comparing Players\' FG% and Average Shot Distance (Minimum 200 Made Shots)'
)

graph = Figure(data=data, layout=layout)
iplot(graph)


# Unlike the first metric, many of the stars actually can find themselves in pretty impressive positions here. The #NBARank Top 30 (seen in red rather than blue) can be seen to populate much of the upper levels of FG percentages for each given average shot distance. Some of the star big men including Anthony Davis and Al Horford absolutely own the 10-12 foot shot distance, while Steph, Chris Paul, and Dirk all make strong cases for themselves with shots at longer distance. However, fellow star Andre Drummond does seem to be one that gets outperformed in this metric by other low-post players with average shots taken within 5 feet from the basket. Drummond, with a field goal percentage of 50.8%, gets nowhere close to that of Tyson Chandler (67.6%) or Deandre Jordan (71.2%), although he does make more shots (as represented by the size of each circle). Other stars getting somewhat outperformed by players with the same average shot distance include Russell Westbrook, Derrick Rose, and Kyle Lowry.

# <h1><a id="section3"></a>Who's Clutch?</h1>

# Lots of star players have that tag of "clutchness" attached to them, but more often than not this recognition is given to stars by overenthusiastic fans that herald a player's willingness to take the late-game shots rather than whether the ball actually goes through the net. In this section, we'll take a look at how each player's field goal percentage fluctuates through each quarter, and see if there are any interesting things that stand out, including any tips or dips that happen come the fourth quarter.

# In[ ]:


data = pd.read_csv('../input/shot_logs.csv')
attackerIDList = list(data['player_id'].unique())
playerIDList = []
for ID in attackerIDList:
    for period in range(1,4):
        playerIDList.append(ID+period/10)

playerData = pd.DataFrame(index=playerIDList, columns=['player', 'period', 'made', 'missed', 'fg_percentage'])

for attackerID in attackerIDList:
    name= data[(data['player_id'] == attackerID)]['player_name'].iloc[0]
    spacePos = name.find(' ')
    firstname = name[0].upper() + name[1:spacePos]
    lastname = name[spacePos+1].upper() + name[spacePos+2:]
    name = firstname + ' ' + lastname
    for period in range(1,5):
        made = np.sum(np.logical_and(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='made',
                                     data[(data['player_id'] == attackerID)]['PERIOD']==period))
        missed = np.sum(np.logical_and(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='missed',
                                       data[(data['player_id'] == attackerID)]['PERIOD']==period))
        percentage = made/(made+missed)
        playerData.at[attackerID+period/10, 'player'] = name
        playerData.at[attackerID+period/10, 'period'] = period
        playerData.at[attackerID+period/10, 'made'] = made
        playerData.at[attackerID+period/10, 'missed'] = missed
        playerData.at[attackerID+period/10, 'fg_percentage'] = percentage
    
newPlayerData = playerData.sort_values('player', ascending=True)
inelligibleNames = newPlayerData[newPlayerData.made < 50]['player']
inelligibleNames = inelligibleNames.unique()
newPlayerData2 = newPlayerData[~newPlayerData.player.isin(inelligibleNames)]
newPlayerData2


# In[ ]:


from ipywidgets import widgets 
from IPython.display import display, clear_output, Image
from plotly.graph_objs import *
from plotly.widgets import GraphWidget

ESPNRankTop30 = ['Lebron James', 'Chris Paul', 'Anthony Davis', 'Russell Westbrook', 'Blake Griffin', 'Stephen Curry',
                'Kevin Love', 'Kevin Durant', 'James Harden', 'Dwight Howard', 'Carmelo Anthony', 'Joakim Noah',
                'Lamarcus Aldridge', 'Marc Gasol', 'Tony Parker', 'Damian Lillard', 'Dirk Nowtizski', 'John Wall',
                'Demarcus Cousins', 'Chris Bosh', 'Tim Duncan', 'Al Jefferson', 'Kyrie Irving', 'Kawhi Leonard', 
                'Serge Ibaka', 'Al Horford', 'Goran Dragic', 'Derrick Rose', 'Kyle Lowry', 'Andre Drummond']

trace1 = Scatter(
            x=newPlayerData2['period'],
            y=newPlayerData2['fg_percentage'],
            mode = 'markers',
            marker = dict(
                size = newPlayerData2['made']/5,
                color = 'rgba(132, 123, 255, .9)',
                line = dict(
                    width = 2,
                ),
            ),
            name='League',
            text = newPlayerData2['player']
        )

newPlayerData3 = newPlayerData2[newPlayerData2.player.isin(ESPNRankTop30)]
trace2 = Scatter(
            x=newPlayerData3['period'],
            y=newPlayerData3['fg_percentage'],
            mode = 'markers',
            marker = dict(
                size = newPlayerData3['made']/5,
                color = 'rgba(255, 123, 132, .9)',
                line = dict(
                    width = 2,
                ),
            ),
            name='#NBARank Top 30',
            text = newPlayerData3['player']
        )

data = [trace1,trace2]

layout = Layout(
    height = 700,
    hovermode = 'closest',
    annotations=Annotations([
        Annotation(
            x=0.5004254919715793,
            y=-0.16191064079952971,
            showarrow=False,
            text='Quarter',
            xref='paper',
            yref='paper'
        ),
        Annotation(
            x=-0.06944728761514841,
            y=0.4714285714285711,
            showarrow=False,
            text='Field Goal %',
            textangle=-90,
            xref='paper',
            yref='paper'
        )
    ]),
    autosize=True,
    margin=Margin(
        b=100
    ),
    title='Comparing Players\' FG% per Quarter (Minimum 50 Shots per Quarter)'
)

graph = Figure(data=data, layout=layout)
iplot(graph)


# Just looking at the plot as a whole, we can see that the range of field goal percentages among players taking at least 50 shots per quarter (70 players) is smallest in the first quarter and largest in the fourth. We can take this to infer that players don't necessarily get too hot or cold in the first quarter, and the fourth is where performance really starts to diverge. Also interesting to see is that the number of shots the players make in the quarter (represented by the size of the player's circle) are much higher in the first quarter and get dramatically smaller by the fourth. This can likely be attributed to fewer shot attempts that individual players are comfortable taking up late in the game, where ball-hogging is much less socially acceptable.
# 
# The stars once again find themselves performing very diversely with relation to the rest of the league in this metric. Star standouts on the positive end of the spectrum include Anthony Davis (again) who easily leads the league in field goal percentage in the fourth quarter with a whopping 60.5%, along with Al Horford (56.5%) and Tim Duncan (56.3%). Nikola Vucevic really looks like a 1st quarter player in this plot, by not only having the best 1st quarter field goal percentage among the represented players, but also be among the highest in made 1st quarter field goals (notice his huge circle). Of course, Vucevic's diminishing activity and performance in the remaining quarters doesn't hurt the 1st-quarter-player argument for him either.
# 
# Star standouts on the negative end of the spectrum include Kyle Lowry, who finds himself near the bottom of fourth quarter FG% with an abysmal 37.1%, along with Damian Lillard (39.1%) and Derrick Rose (39.2%). To my surprise Klay Thompson and Joe Johnson also can be found alongside them with sub-39% fourth quarter FG percentages. Trey Burke may also have been shooting a little too much in the fourth, and is the only represented player that shot under 33% in the fourth quarter.

# <h1><a id="section4"></a>A Closer Look at the Clutchness of the Top 30</h1>

# The prior plot allowed us to see standout anomolies in each quarter, but didn't really give us an easy way of tracking each individual player's flunctuation of FG% between quarters. We can make a plot that does allow us to put each individual star's quarter-by-quarter performance under the microscope here.

# In[ ]:


ESPNRankTop30 = ['Lebron James', 'Chris Paul', 'Anthony Davis', 'Russell Westbrook', 'Blake Griffin', 'Stephen Curry',
                'Kevin Love', 'Kevin Durant', 'James Harden', 'Dwight Howard', 'Carmelo Anthony', 'Joakim Noah',
                'Lamarcus Aldridge', 'Marc Gasol', 'Tony Parker', 'Damian Lillard', 'Dirk Nowtizski', 'John Wall',
                'Demarcus Cousins', 'Chris Bosh', 'Tim Duncan', 'Al Jefferson', 'Kyrie Irving', 'Kawhi Leonard', 
                'Serge Ibaka', 'Al Horford', 'Goran Dragic', 'Derrick Rose', 'Kyle Lowry', 'Andre Drummond']
newPlayerData3 = newPlayerData[newPlayerData.player.isin(ESPNRankTop30)]

newPlayerData3 = newPlayerData3.sort_values('player')
newPlayerData3['ranking'] = 0
for i in range(len(ESPNRankTop30)):
    newPlayerData3.loc[newPlayerData3['player'] == ESPNRankTop30[i], 'ranking'] = i+1
    
newPlayerData3 = newPlayerData3.sort_values('period', ascending = True)
newPlayerData3


# In[ ]:


from plotly.graph_objs import *
from ipywidgets import widgets 
from IPython.display import display, clear_output, Image
from plotly.graph_objs import *
from plotly.widgets import GraphWidget

data = []
newPlayerData3 = newPlayerData3.sort_values(['ranking','period'], ascending = [True, True])
for player in list(newPlayerData3['player'].unique()):
    data.append(Scatter(
        x=newPlayerData3[newPlayerData3['player']==player]['period'],
        y=newPlayerData3[newPlayerData3['player']==player]['fg_percentage'],
        mode = 'lines+markers',
        line = dict(
            color = 'rgba(100, 100, 100, .7)',
            width = 1
#             dash = 'dot'
        ),
        marker = dict(
            size = newPlayerData3['made']/5,
            color = 'rgba(132, 123, 255, .9)',
            line = dict(
                color = 'rgb(250,250,250)',
                width = 2
            ),
        ),
        name = player + ' #' + str(newPlayerData3[newPlayerData3['player']==player]['ranking'].iloc[0]),
        text = (player + ' #' + newPlayerData3[newPlayerData3['player']==player]['ranking'].astype(str) + '<br>' +
                'Made: ' + newPlayerData3[newPlayerData3['player']==player]['made'].astype(str) + '<br>' +
                'Missed: ' + newPlayerData3[newPlayerData3['player']==player]['missed'].astype(str))
    ))

layout = Layout(
    hovermode = 'closest',
    annotations=Annotations([
        Annotation(
            x=0.5004254919715793,
            y=-0.16191064079952971,
            showarrow=False,
            text='Quarter',
            xref='paper',
            yref='paper'
        ),
        Annotation(
            x=-0.06944728761514841,
            y=0.4714285714285711,
            showarrow=False,
            text='Field Goal %',
            textangle=-90,
            xref='paper',
            yref='paper'
        )
    ]),
    autosize=True,
    margin=Margin(
        b=100
    ),
    title='Comparing Players\' FG% per Quarter (ESPNRank Top 30)'
)

graph = Figure(data=data, layout=layout)
iplot(graph)


# Note: You can isolate each individual's performance by double clicking on their name in the legend pane on the right.
# 
# Now that we have nifty lines between markers and individual traces for each player, we can track how each player's FG% changed across the game's quarters. One notable individual of curiousity included Stephen Curry, who actually trended down in FG% every quarter, from 51.4% in the first to 44% in the fourth. Most players such as Lebron James and John Wall don't necessarily follow any sort of clear trend through the quarters, but many stars did have their lowest FG% occuring in the fourth quarter. The only stars that had their highest FG% occuring in the fourth quarter were Anthony Davis, Tim Duncan, Demarcus Cousins, and Serge Ibaka. Carmelo Anthony, who is often considered a clutch star, doesn't necessarily have his highest FG% occuring in the fourth quarter, but does see a rise in FG% there after a decline through the first three quarters. Other players that are often considered clutch stars, such as Kyrie Irving and Chris Paul, had lackluster results in the fourth when observed from the season-wide scale. 

# <h1><a id="section5"></a>Lockdown Defenders</h1>

# It's no secret that the stars we've been looking at can score the basketball, but one interesting question to pose is who are the best, and perhaps worst, men to defend them. We take our shot at figuring that out here.

# In[ ]:


data = pd.read_csv('../input/shot_logs.csv')
ESPNRankTop30 = ['lebron james', 'chris paul', 'anthony davis', 'russell westbrook', 'blake griffin', 'stephen curry',
                'kevin love', 'kevin durant', 'james harden', 'dwight howard', 'carmelo anthony', 'joakim noah',
                'lamarcus aldridge', 'marc gasol', 'tony parker', 'damian lillard', 'dirk nowtizski', 'john wall',
                'demarcus cousins', 'chris bosh', 'tim duncan', 'al jefferson', 'kyrie irving', 'kawhi leonard', 
                'serge ibaka', 'al horford', 'goran dragic', 'derrick rose', 'kyle lowry', 'andre drummond']
data = data[data.player_name.isin(ESPNRankTop30)]
defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())

playerData = pd.DataFrame(index=defenderIDList, columns=['player', 'made_against', 'missed_against', 'fg_percentage_against'])

for defenderID in defenderIDList:
    name= data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['CLOSEST_DEFENDER'].iloc[0]
    made = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='made')
    missed = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='missed')
    percentage = made/(made+missed)
    playerData.at[defenderID, 'player'] = name
    playerData.at[defenderID, 'made_against'] = made
    playerData.at[defenderID, 'missed_against'] = missed
    playerData.at[defenderID, 'fg_percentage_against'] = percentage

playerData2 = playerData.drop(playerData[np.logical_and(playerData.missed_against < 20, playerData.made_against < 20)].index)


# <h1>Defenders that can make the stars miss</h1> 

# In[ ]:


playerData2 = playerData2.sort_values('fg_percentage_against', ascending = True)
playerData2.head(10)


# Caron Butler comes in as the number 1 overall star-stopper, by incredibly letting in only 32.4% of the shots against him by star players. I would say that it's a surprising result, but we always knew he was <a href='https://www.youtube.com/watch?v=X6YYDDsNreY' target='_blank'>a tricky trickster. </a><img src="http://i.imgur.com/tnVsoIT.gif" alt="silly Caron" width="500px">

# Fellow vet and notable perimeter-defender Tony Allen also makes his presence known in this metric, coming in second place among represented players (players with at least 20 misses or makes against them) with a very impressive 33.3% defensive FG% when defending the stars. Also worthy of shoutout is Jeff Teague, who not only had a very impressive defensive FG%, but also had an impressive number of sheer matchups against the star players, being the only player with over 100 attempts against him on this list.

# <h1>Defenders that can't</h1>

# In[ ]:


playerData2 = playerData2.sort_values('fg_percentage_against', ascending = False)
playerData2.head(10)


# Coming in the top 10 worst star stoppers is star player Carmelo Anthony himself, who sits pretty at number 3 by letting in 67.7% of the field goal attempts against him by other players in #NBARank's Top 30. Nevertheless, no matter how hard he doesn't try, Melo is still overshadowed by Bojan Bogdanovic, who comes in at the number 1 spot for worst defender of star players with an incredible-yet-horrendous defensive FG% of 75%. However, you may be happy to hear that he has <a href="https://8points9seconds.com/2017/12/20/bojan-bogdanovic-defense/" target="_blank">reportedly improved in his defensive abilities</a>. Thabo Sefolosha, widely regarded as a top wing defender, makes a surprising appearance on this list at number 6.

# <h1>Every Star's Kryptonite</h1>

# Every star on our list is a Superman to their team, but even Superman has a Kryptonite he's weak to. Here we try to find each star's matchup nightmare by seeing which defender they shoot worst against. And just like Superman, who has the power to move planets but is somehow weak to a green rock, each star's kryptonite may not be exactly who we'd expect.
# <img src="http://1.bp.blogspot.com/-ydY0O-mdcAs/Te5oOJWX3VI/AAAAAAAAALw/FFrdd-wXI-k/s1600/superman.jpg" alt="superman" width="300px">

# In[ ]:


data = pd.read_csv('../input/shot_logs.csv')
stars = ['lebron james', 'chris paul', 'anthony davis', 'russell westbrook', 'blake griffin', 'stephen curry',
                'kevin love', 'kevin durant', 'james harden', 'dwight howard', 'carmelo anthony', 'joakim noah',
                'lamarcus aldridge', 'marc gasol', 'tony parker', 'damian lillard', 'dirk nowtizski', 'john wall',
                'demarcus cousins', 'chris bosh', 'tim duncan', 'al jefferson', 'kyrie irving', 'kawhi leonard', 
                'serge ibaka', 'al horford', 'goran dragic', 'derrick rose', 'kyle lowry', 'andre drummond']

kryptoniteData = pd.DataFrame(index=stars, columns=['kryptonite', 'made_against', 'missed_against', 
                                                            'fg_percentage_against'])

for star in stars:
    data = pd.read_csv('../input/shot_logs.csv')
    data = data[data.player_name.isin([star])]
    defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())

    playerData = pd.DataFrame(index=defenderIDList, columns=['player', 'made_against', 'missed_against', 'fg_percentage_against'])
    try:
        for defenderID in defenderIDList:
            name= data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['CLOSEST_DEFENDER'].iloc[0]
            made = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='made')
            missed = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='missed')
            percentage = made/(made+missed)
            playerData.at[defenderID, 'player'] = name
            playerData.at[defenderID, 'made_against'] = made
            playerData.at[defenderID, 'missed_against'] = missed
            playerData.at[defenderID, 'fg_percentage_against'] = percentage

        playerData2 = playerData.drop(playerData[(playerData.missed_against + playerData.made_against) < 8].index)
        playerData2 = playerData2.sort_values('fg_percentage_against', ascending = True)
        kryptoniteData.at[star, 'kryptonite']  = playerData2['player'].iloc[0]
        kryptoniteData.at[star, 'made_against']  = playerData2['made_against'].iloc[0]
        kryptoniteData.at[star, 'missed_against']  = playerData2['missed_against'].iloc[0]
        kryptoniteData.at[star, 'fg_percentage_against']  = playerData2['fg_percentage_against'].iloc[0]
    except:
        kryptoniteData.at[star, 'kryptonite']  = 'N/A'
        kryptoniteData.at[star, 'made_against']  = 'N/A'
        kryptoniteData.at[star, 'missed_against']  = 'N/A'
        kryptoniteData.at[star, 'fg_percentage_against']  = 'N/A'
        
kryptoniteData


# Above is the list of stars along with their corresponding "kryptonite" defensive matchup. There needed to be a minimum of 8 observed field goal attempts against the defending player in order to be a valid matchup. Pretty self explanatory chart, but I will highlight the fact that the only matchup that resulted in a 0% field goal percentage for the star occured to the King himself, Lebron James. And who was the King slayer you may ask? That would be Travis Wear, also known as this fella:
# 
# <img src="https://static01.nyt.com/images/2015/02/23/sports/23ARATONjpsub/23ARATONjpsub-master1050.jpg" alt="Travis" height="500" width="500" title="Travis Wear, Lebron's worst nightmare">
# 
# All hail King Wear!

# <h1><a id="section6"></a>Defender Distance vs FG%</h1>

# The final metric we'll be looking at is how the distance from the nearest defender affects the stars' field goal percentage. We'll look at field goal percentages when the defender is between 0-1 feet away, 1-2 feet away, 2-3 feet away, 3-4 feet away, 4-5 feet away, and over 5 feet away from the shot. We can also compare this with the league-wide average field goal percentages for shots taken with each of these given amounts of space.

# In[ ]:


data = pd.read_csv('../input/shot_logs.csv')
stars = ['lebron james', 'chris paul', 'anthony davis', 'russell westbrook', 'blake griffin', 'stephen curry',
                'kevin love', 'kevin durant', 'james harden', 'dwight howard', 'carmelo anthony', 'joakim noah',
                'lamarcus aldridge', 'marc gasol', 'tony parker', 'damian lillard', 'dirk nowtizski', 'john wall',
                'demarcus cousins', 'chris bosh', 'tim duncan', 'al jefferson', 'kyrie irving', 'kawhi leonard', 
                'serge ibaka', 'al horford', 'goran dragic', 'derrick rose', 'kyle lowry', 'andre drummond']

playerData = pd.DataFrame(index=list(range(len(stars)+1)), columns=['player', 'made_1ft', 'missed_1ft', 'fg_percentage_1ft',
                                                        'made_2ft', 'missed_2ft', 'fg_percentage_2ft',
                                                        'made_3ft', 'missed_3ft', 'fg_percentage_3ft',
                                                        'made_4ft', 'missed_4ft', 'fg_percentage_4ft',
                                                        'made_5ft', 'missed_5ft', 'fg_percentage_5ft',
                                                        'made_over_5ft', 'missed_over_5ft', 'fg_percentage_over_5ft'])

for defenderDist in range(1,6):
    made = np.sum(data[np.logical_and(data['CLOSE_DEF_DIST'] <= defenderDist, 
                                      data['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='made')
    missed = np.sum(data[np.logical_and(data['CLOSE_DEF_DIST'] < defenderDist, 
                                            data['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='missed')
    percentage = made/(made+missed)
    playerData.loc[len(stars), 'player'] = 'League Average'
    playerData.loc[len(stars), ('made_' + str(defenderDist) + 'ft')] = made
    playerData.loc[len(stars), ('missed_' + str(defenderDist) + 'ft')] = missed
    playerData.loc[len(stars), ('fg_percentage_' + str(defenderDist) + 'ft')] = percentage
    
made = np.sum(data[data['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='made')
missed = np.sum(data[data['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='missed')
percentage = made/(made+missed)
playerData.loc[len(stars), 'made_over_5ft'] = made
playerData.loc[len(stars), 'missed_over_5ft'] = missed
playerData.loc[len(stars), 'fg_percentage_over_5ft'] = percentage    
    
for star in stars:
    stardata = data[data.player_name == star]
    playerData.loc[stars.index(star), 'player'] = star
    for defenderDist in range(1,6):
        try:
            made = np.sum(stardata[np.logical_and(stardata['CLOSE_DEF_DIST'] <= defenderDist, 
                                                  stardata['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='made')
            missed = np.sum(stardata[np.logical_and(stardata['CLOSE_DEF_DIST'] < defenderDist, 
                                                  stardata['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='missed')
            percentage = made/(made+missed)
            playerData.loc[stars.index(star), ('made_' + str(defenderDist) + 'ft')] = made
            playerData.loc[stars.index(star), ('missed_' + str(defenderDist) + 'ft')] = missed
            playerData.loc[stars.index(star), ('fg_percentage_' + str(defenderDist) + 'ft')] = percentage
        except:
            playerData.loc[stars.index(star), ('made_' + str(defenderDist) + 'ft')] = 'N/A'
            playerData.loc[stars.index(star), ('missed_' + str(defenderDist) + 'ft')] = 'N/A'
            playerData.loc[stars.index(star), ('fg_percentage_' + str(defenderDist) + 'ft')] = 'N/A'

    try:
        made = np.sum(stardata[stardata['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='made')
        missed = np.sum(stardata[stardata['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='missed')
        percentage = made/(made+missed)
        playerData.loc[stars.index(star), 'made_over_5ft'] = made
        playerData.loc[stars.index(star), 'missed_over_5ft'] = missed
        playerData.loc[stars.index(star), 'fg_percentage_over_5ft'] = percentage    
    except:
        playerData.loc[stars.index(star), 'made_over_5ft'] = 'N/A'
        playerData.loc[stars.index(star), 'missed_over_5ft'] = 'N/A'
        playerData.loc[stars.index(star), 'fg_percentage_over_5ft'] = 'N/A'

playerData


# In[ ]:


trace1 = go.Bar(
    x=list(playerData['player']),
    y=list(playerData['fg_percentage_1ft']),
    name='Defender 0-1ft away',
    text = 'Made: ' + playerData['made_1ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_1ft'].astype(str)
)
trace2 = go.Bar(
    x=list(playerData['player']),
    y=list(playerData['fg_percentage_2ft']),
    name='Defender 1-2ft away',
    text = 'Made: ' + playerData['made_2ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_2ft'].astype(str)
)
trace3 = go.Bar(
    x=list(playerData['player']),
    y=list(playerData['fg_percentage_3ft']),
    name='Defender 2-3ft away',
    text = 'Made: ' + playerData['made_3ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_3ft'].astype(str)
)
trace4 = go.Bar(
    x=list(playerData['player']),
    y=list(playerData['fg_percentage_4ft']),
    name='Defender 3-4ft away',
    text = 'Made: ' + playerData['made_4ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_4ft'].astype(str)
)
trace5 = go.Bar(
    x=list(playerData['player']),
    y=list(playerData['fg_percentage_5ft']),
    name='Defender 4-5ft away',
    text = 'Made: ' + playerData['made_5ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_5ft'].astype(str)
)
trace6 = go.Bar(
    x=list(playerData['player']),
    y=list(playerData['fg_percentage_over_5ft']),
    name='Defender >5ft away',
    text = 'Made: ' + playerData['made_over_5ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_over_5ft'].astype(str)
)

data = [trace1,trace2,trace3,trace4,trace5,trace6]
layout = go.Layout(
    barmode='group',
    title = "FG% Depending on Distance from Defender",
    annotations=Annotations([
        Annotation(
            x=-0.05944728761514841,
            y=0.4714285714285711,
            showarrow=False,
            text='Field Goal %',
            textangle=-90,
            xref='paper',
            yref='paper'
        )
    ]),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Although it'd be reasonable to expect that field goal percentages increase for each player when they are given more and more space away from the defender, that's actually rarely the case among the stars represented in the above plot. In fact, the only players that actually follow this expectation are Chris Paul and Andre Drummond, who do see increases in field goal percentages along every increasing interval of space given by the defender. Dirk Nowitzki comes close to fulfilling the pattern. Most of the players, on the other hand, have a peak in field goal percentage when shooting a unique distance away from the closest defender. For instance, Al Horford was incredible when shooting with a defender between 3-4 feet away, as he shot over 73% in those cases. However, given one more foot of space by the defender, Horford shot more than 20% lower. It's very possible that there are some confounding variables that explain the anomolies we're observing with this plot, such as certain, closer distances from the defender corresponding with higher percentage shots such as layups and dunks, while further distances from the defender corresponding with lower percentage shots such as jump-shots and three-pointers. Whatever it is, I still like to imagine a scout informing players to make sure they're 4-5 feet, but absolutely not 3-4 feet, from Al Horford when he's shooting.
