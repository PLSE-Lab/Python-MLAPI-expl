#!/usr/bin/env python
# coding: utf-8

# ## NFL Punt Analytics Proposal  ##
# 
# Thirty seven known punt-related concussions occurred during the 2016 and 2017 NFL seasons. By reviewing the evidence from all 6,681 punt plays that are recorded from these two seasons, I seek to show that the following three rule changes will improve player safety while being implementable and preserving the excitement and unpredictability of the great game of football.
# 
# * [Proposal #1](#proposal_1): Award a five yard bonus to the receiving team on a fair catch. 
# * [Proposal #2](#proposal_2): Require single coverage of gunners by the receiving team.
# * [Proposal #3](#proposal_3): Install helmet sensors to monitor deceleration.
# 
# In support of proposal #1, **I attempt to quantify the risk-reward tradeoff of reducing the number of punt returns and increasing the number of fair catches**. I find that 32% of punt returns conditional on a punt received yield fewer than five yards on the return. In the current framework, punt returners have an incentive to attempt a return whenever they think their expected value of returning is greater than 0 yards. In the new framework, punt returns will have an incentive to make a return only when they believe the expected value of returning is greater than 5 yards. The concussion rate is 10 injuries per 1000 plays for a fielded return versus 1.8 injuries per 1000 plays for a fair catch, so we expect a 80% decrease in concussions for each incremental fair catch that is called. On the other hand, punt returners will continue to be able to try to go for a return and make a play if they see an opportunity to go for it.  
# 
# In support of proposal #2, **I analyze the injury rate conditioning on both the choice of coverage and on the yards between the line of scrimmage and the end zone, as these factors are not independent of one another**. Teams are more likely to choose double coverage when there is more open field, i.e. greater yards to go between the kicking team and the receiving team's end zone. On the other hand, plays where the field is longer also have higher injury rates, perhaps because the possibility of a touchback or coffin corner is reduced and the coverage team has to run farther to reach the spot of punt reception, resulting in a more open field and higher speed play. I show that even when one conditions on yards-to-go, double coverage generates somewhat higher injury rates than single coverage in these long field situations. On the other hand, double coverage does not appear to help the receiving team make longer punt returns, i.e. the punt returner is able to generate just as much excitement whether or not the gunners were single covered or not. Enforcing single coverage would then appear to reduce injuries without limiting excitement, and so seems like a costless design proposal, although the statistical significance of these findings (recall there are only 37 concussion events in this entire sample) is limited by the small sample size.
# 
# In support of proposal #3, **I show that deceleration appears more important than velocity in determining injury, but that the NGS data is inherently limited in its ability to measure deceleration.**. The code below attempts to measure each player's velocity and deceleration at the time of a tackle event. I find that players who are not injured on a play have velocity as great as or greater than players involved in an injury generating tackle. On the other hand, the injured player experiences much greater deceleration than uninjured players. This suggests that deceleration is a more important factor in injury than velocity. The NGS data provides (x, y) coordinates at 100 millisecond resolution, so a player who decelerates from 9.8 m/s to 0.0 m/s would be recorded as having experienced at most 10 g's in deceleration, since 10g = (9.8m/s - 0.0m/s) / 0.1s. In fact the player could have experienced much greater deceleration, e.g. he could have decelerated to 0 in 20 milliseconds, but the resolution of the NGS system is not granular enough to show this. For the purpose of measuring head injury, the deceleration experienced at the head would seem to be a critical indicator of whether injury is likely to have occurred. This data would be collected not to take players out of the game or to limit their playing time, but rather to better study the conditions and plays that result in greater head deceleration and likely injury.
# 
# The notebook below provides the analysis and figures that support each of these three proposals. There is an extensive amount of formatting code and data preparation code, which I have attempted to hide, so that the focus can be on the interesting parts - the output and figures.
# 
# [Additional observations](#additional_observations) are found at the end.
# 
# [Final thoughts](#final_thoughts) concludes.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import re
import pandas as pd
import numpy as np
import glob
import os
import logging
import sys
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


ngs_files = ['../input/NGS-2016-pre.csv',
             '../input/NGS-2016-reg-wk1-6.csv',
             '../input/NGS-2016-reg-wk7-12.csv',
             '../input/NGS-2016-reg-wk13-17.csv',
             '../input/NGS-2016-post.csv',
             '../input/NGS-2017-pre.csv',
             '../input/NGS-2017-reg-wk1-6.csv',
             '../input/NGS-2017-reg-wk7-12.csv',
             '../input/NGS-2017-reg-wk13-17.csv',
             '../input/NGS-2017-post.csv']
PUNT_TEAM = set(['GL', 'PLW', 'PLT', 'PLG', 'PLS', 'PRG', 'PRT', 'PRW', 'PC',
                 'PPR', 'P', 'GR'])
RECV_TEAM = set(['VR', 'PDR', 'PDL', 'PLR', 'PLM', 'PLL', 'VL', 'PFB', 'PR'])


# ### **Proposal 1: **Award five yard bonus on fair catches ###
# 
# <a id="proposal_1"></a>
# 
# **Fair catches are 80% safer than punt received events.**
# The graph below shows that the concussion rate is 10.0 per 1000 returns on punt received events vs. 1.8 on fair catch events. While fair catches are not as exciting as punt returns, they are certainly much safer. Our goal should not be to ban all punt returns wholesale, but rather to reduce the number of dangerous but less exciting (i.e. short) punt returns while preserving the potential for long punt returns (ESPN highlight material). There is always risk in each play (even fair catches), and the analysis below helps us to find the best trade off between risk and reward.
# 

# In[ ]:


plays_df = pd.read_csv('../input/play_information.csv')

def get_return_yards(s):
    m = re.search('for ([0-9]+) yards', s)
    if m:
        return int(m.group(1))
    elif re.search('for no gain', s):
        return 0
    else:
        return np.nan

plays_df['Return'] = plays_df['PlayDescription'].map(
        lambda x: get_return_yards(x))

video_review = pd.read_csv('../input/video_review.csv')
video_review = video_review.rename(columns={'GSISID': 'InjuredGSISID'})

plays_df= plays_df.merge(video_review, how='left',
                         on=['Season_Year', 'GameKey', 'PlayID'])

plays_df['InjuryOnPlay'] = 0
plays_df.loc[plays_df['InjuredGSISID'].notnull(), 'InjuryOnPlay'] = 1

plays_df = plays_df[['Season_Year', 'GameKey', 'PlayID', 'Return', 'InjuryOnPlay']]

ngs_df = []
for filename in ngs_files:
    df = pd.read_csv(filename, parse_dates=['Time'])
    df = df.loc[df['Event'].isin(['fair_catch', 'punt_received'])]
    df = pd.concat([df, pd.get_dummies(df['Event'])], axis=1)
    df = df.groupby(['Season_Year', 'GameKey', 'PlayID'])[['fair_catch', 'punt_received']].max()
    ngs_df.append(df.reset_index())
ngs_df = pd.concat(ngs_df)

plays_df = plays_df.merge(ngs_df, on=['Season_Year', 'GameKey', 'PlayID'])


# In[ ]:


injury_per_1000_fair_catch = 1000 * plays_df.loc[plays_df['fair_catch']==1,
                                          'InjuryOnPlay'].mean()
injury_per_1000_punt_received = 1000 * plays_df.loc[plays_df['punt_received']==1,
                                           'InjuryOnPlay'].mean()
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar([0, 1], [injury_per_1000_fair_catch, injury_per_1000_punt_received])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fair Catch', 'Punt Received'])
plt.text(0, injury_per_1000_fair_catch+0.2, '{:.1f}'.format(injury_per_1000_fair_catch))
plt.text(1, injury_per_1000_punt_received+0.2, '{:.1f}'.format(injury_per_1000_punt_received))
plt.title("Concussion Rate")
plt.ylabel("Injuries per 1000 Events")
sns.despine(top=True, right=True)
plt.show()


# #### ** 32% of punt returns result in fewer than 5 yards ** ####
# The graph below shows that 32% of punt returns yield fewer than 5 yards. These seem like prime candidates for events that we could replace with an awarded bonus of 5 yards, with little effect on field position but with substantial decrease in injury. 
# 
# **Nudge Theory**
# 
# As an additional alternative, we could consider a proposal that makes fair catch the default, and require a returner to make a signal or gesture in order to indicate that he will not choose a fair catch. Evidence in support of setting an optimal default when giving people choices include research by Richard Thaler, Nobel prize-winning economist at the University of Chicago, who has written extensively on the notion of 'nudges.' If we nudge receivers into choosing a fair catch, then they will have to make the conscious decision to elect to return after assessing that the situation presents an opportunity to make a long return.

# In[ ]:


x_groups = ['0-3 yds', '3-5 yds', '5-7 yds', '7-9 yds',
            '9-12 yds', '12-15 yds', '15-20 yds', '20+ yds']
rec = plays_df.loc[(plays_df['punt_received']==1) 
                   &(plays_df['Return'].notnull())]

y_groups = [sum(rec['Return']<=3) / len(rec),
            sum((rec['Return']>3) & (rec['Return']<=5)) / len(rec),
            sum((rec['Return']>5) & (rec['Return']<=7)) / len(rec),
            sum((rec['Return']>7) & (rec['Return']<=9)) / len(rec),
            sum((rec['Return']>9) & (rec['Return']<=12)) / len(rec),
            sum((rec['Return']>12) & (rec['Return']<=15)) / len(rec),
            sum((rec['Return']>15) & (rec['Return']<=20))/ len(rec),
            sum(rec['Return']>20) / len(rec)]

y_bottoms = [0,
             sum(rec['Return']<=3) / len(rec),
             sum(rec['Return']<=5) / len(rec),
             sum(rec['Return']<=7) / len(rec),
             sum(rec['Return']<=9) / len(rec),
             sum(rec['Return']<=12) / len(rec),
             sum(rec['Return']<=15) / len(rec),
             sum(rec['Return']<=20) / len(rec)]

fig = plt.figure(figsize=(8.5,4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar(range(len(x_groups)), y_groups, bottom=y_bottoms)
ax.set_xticks(range(len(x_groups)))
ax.set_xticklabels(x_groups)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
for i in range(len(x_groups)):
    plt.text(i-0.2, y_bottoms[i]+y_groups[i]+0.02, '{:.0f}%'.format(100*y_groups[i]))
sns.despine(top=True, right=True)
plt.title("Distribution of Punt Returns by Length")
plt.show()


# #### ** Example 1 of a play where the punt returner should probably have called a fair catch ** ####
# A NFL punter can kick a punt with hang time of 4.4 seconds on average, which gives NFL level gunners time to advance 40 yards. In practice there is usually a 1-2 second gap between the time of ball snap and the time of the punt, allowing the gunner to advance 50 or 60 yards down the field before the ball is caught That's a lot of time, and sometimes I cringe while watching a punt returner (aka sitting duck) about to get nailed by a gunner - often the punt returner is too busy focusing on the ball to even notice that a player is about to hit him on his blind side. This appears to be one of those plays, although as often happens it's the gunner who suffers the injury and not the returner.

# In[ ]:


play_player_role_data = pd.read_csv('../input/play_player_role_data.csv')
gfp = video_review.merge(play_player_role_data, on=['GameKey', 'PlayID', 'Season_Year'])


# In[ ]:


df_29 = pd.read_csv('../input/NGS-2016-pre.csv', parse_dates=['Time'])
df_29 = df_29.loc[(df_29['GameKey']==29) & (df_29['PlayID']==538)]
df_29 = df_29.merge(gfp, on=['GameKey', 'PlayID', 'Season_Year', 'GSISID'])
df_29 = df_29.sort_values(['GameKey', 'PlayID', 'Season_Year', 'GSISID', 'Time'])

fig = plt.figure(figsize = (10, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
line_set = df_29.loc[df_29['Event'].isin(['ball_snap'])]
line_set_time = line_set['Time'].min()
line_of_scrimmage = line_set.loc[line_set['Role'].isin(['PLS', 'PLG', 'PRG']), 'x'].median()

recv_df = df_29.loc[df_29['Event']=='punt_received']
recv_time = recv_df['Time'].min()
event_df = df_29.loc[df_29['Time'] <= recv_time]
event_df = event_df.loc[event_df['Time'] >= recv_time + pd.Timedelta('-2s')]

injured = df_29['InjuredGSISID'].values[0]
partner = float(df_29['Primary_Partner_GSISID'].values[0])
            
players = event_df['GSISID'].unique()
for player in players:
    player_df = event_df.loc[event_df['GSISID'] == player]
    role = str(player_df['Role'].values[0])
    if re.sub('[io0-9]', '', str(role)) in PUNT_TEAM:
        color = '#fdc086'
        marker = 'x'
        linewidth = 6
    else:
        color = '#beaed4'
        marker = 'o'
        linewidth = 6
    if player == injured:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#f0027f'
    elif player == partner:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#ffff99'
    else:
        linestyle = '--'
    alphas = np.ones(len(player_df))
    alphas = alphas.cumsum() / alphas.sum()
    px = player_df['x'].values
    py = player_df['y'].values
    for k in range(len(px)):
        plt.plot(px[k:], py[k:], color=color,
                 linewidth=linewidth*(k+1+4)/(4+len(px)),
                 linestyle=linestyle,
                 alpha=(k+1)/len(px))
        player_df = player_df.reset_index(drop=True)
        x = player_df['x'].iloc[-1]
        y = player_df['y'].iloc[-1]

        marker = (3, 0, 90 + player_df['o'].iloc[-1])
        plt.scatter(player_df['x'].iloc[-1],
                    player_df['y'].iloc[-1],
                    marker=marker,
                    s=linewidth*60,
                    color=color)
        if (role == 'PR'):
            circ = plt.Circle((player_df['x'].iloc[-1],
                                player_df['y'].iloc[-1]),
                                5, color=color,
                                fill=False)
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
plt.axvline(x=0, color='w', linewidth=2)
plt.axvline(x=10, color='w', linewidth=2)
plt.axvline(x=15, color='w', linewidth=2)
plt.axvline(x=20, color='w', linewidth=2)
plt.axvline(x=25, color='w', linewidth=2)
plt.axvline(x=30, color='w', linewidth=2)
plt.axvline(x=35, color='w', linewidth=2)
plt.axvline(x=40, color='w', linewidth=2)
plt.axvline(x=45, color='w', linewidth=2)
plt.axvline(x=50, color='w', linewidth=2)
plt.axvline(x=55, color='w', linewidth=2)
plt.axvline(x=60, color='w', linewidth=2)
plt.axvline(x=65, color='w', linewidth=2)
plt.axvline(x=70, color='w', linewidth=2)
plt.axvline(x=75, color='w', linewidth=2)
plt.axvline(x=80, color='w', linewidth=2)
plt.axvline(x=85, color='w', linewidth=2)
plt.axvline(x=90, color='w', linewidth=2)
plt.axvline(x=95, color='w', linewidth=2)
plt.axvline(x=100, color='w', linewidth=2)
plt.axvline(x=105, color='w', linewidth=2)
plt.axvline(x=110, color='w', linewidth=2)
plt.axvline(x=120, color='w', linewidth=2)
plt.axvline(x=line_of_scrimmage, color='y', linewidth=3)
plt.axhline(y=0, color='w', linewidth=2)
plt.axhline(y=53.3, color='w', linewidth=2)
plt.text(x=18, y=2, s= '1', color='w')
plt.text(x=21, y=2, s= '0', color='w')
plt.text(x=28, y=2, s= '2', color='w')
plt.text(x=31, y=2, s= '0', color='w')
plt.text(x=38, y=2, s= '3', color='w')
plt.text(x=41, y=2, s= '0', color='w')
plt.text(x=48, y=2, s= '4', color='w')
plt.text(x=51, y=2, s= '0', color='w')
plt.text(x=58, y=2, s= '5', color='w')
plt.text(x=61, y=2, s= '0', color='w')
plt.text(x=68, y=2, s= '4', color='w')
plt.text(x=71, y=2, s= '0', color='w')
plt.text(x=78, y=2, s= '3', color='w')
plt.text(x=81, y=2, s= '0', color='w')
plt.text(x=88, y=2, s= '2', color='w')
plt.text(x=91, y=2, s= '0', color='w')
plt.text(x=98, y=2, s= '1', color='w')
plt.text(x=101, y=2, s= '0', color='w')

ax.set_xticks([0, 120])
ax.set_yticks([0, 53.3])
ax.set_xticklabels(['', ''])
ax.set_yticklabels(['', ''])
ax.tick_params(axis=u'both', which=u'both', length=0)
ax.add_artist(circ)
ax.set_facecolor("#2ca25f")
plt.title("GR is injured while tackling PR (helmet to body).\nGameKey 29, Play ID 538. NYJ Punting to WAS.")
plt.show() 


# #### Example 2 of a play where the punt returner should probably have called a fair catch ####
# Here  the punt returner isn't even directly involved in the injury, but rather two members of the coverage team collide with each other. When multiple players have a chance to converge at a high rate of speed to the same stationary target, there is increased risk. 

# In[ ]:


df_296 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv', parse_dates=['Time'])
df_296 = df_296.loc[(df_296['GameKey']==296) & (df_296['PlayID']==2667)]
df_296 = df_296.merge(gfp, on=['GameKey', 'PlayID', 'Season_Year', 'GSISID'])
df_296 = df_296.sort_values(['GameKey', 'PlayID', 'Season_Year', 'GSISID', 'Time'])
fig = plt.figure(figsize = (10, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
line_set = df_296.loc[df_296['Event'].isin(['ball_snap'])]
line_set_time = line_set['Time'].min()
line_of_scrimmage = line_set.loc[line_set['Role'].isin(['PLS', 'PLG', 'PRG']), 'x'].median()

recv_df = df_296.loc[df_296['Event']=='punt_received']
recv_time = recv_df['Time'].min()
event_df = df_296.loc[df_296['Time'] <= recv_time]
event_df = event_df.loc[event_df['Time'] >= recv_time + pd.Timedelta('-2s')]

injured = df_296['InjuredGSISID'].values[0]
partner = float(df_296['Primary_Partner_GSISID'].values[0])
            
players = event_df['GSISID'].unique()
for player in players:
    player_df = event_df.loc[event_df['GSISID'] == player]
    role = str(player_df['Role'].values[0])
    if re.sub('[io0-9]', '', str(role)) in PUNT_TEAM:
        color = '#fdc086'
        marker = 'x'
        linewidth = 6
    else:
        color = '#beaed4'
        marker = 'o'
        linewidth = 6
    if player == injured:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#f0027f'
    elif player == partner:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#ffff99'
    else:
        linestyle = '--'
    alphas = np.ones(len(player_df))
    alphas = alphas.cumsum() / alphas.sum()
    px = player_df['x'].values
    py = player_df['y'].values
    for k in range(len(px)):
        plt.plot(px[k:], py[k:], color=color,
                 linewidth=linewidth*(k+1+4)/(4+len(px)),
                 linestyle=linestyle,
                 alpha=(k+1)/len(px))
        player_df = player_df.reset_index(drop=True)
        x = player_df['x'].iloc[-1]
        y = player_df['y'].iloc[-1]

        marker = (3, 0, 90 + player_df['o'].iloc[-1])
        plt.scatter(player_df['x'].iloc[-1],
                    player_df['y'].iloc[-1],
                    marker=marker,
                    s=linewidth*60,
                    color=color)
        if (role == 'PR'):
            circ = plt.Circle((player_df['x'].iloc[-1],
                                player_df['y'].iloc[-1]),
                                5, color=color,
                                fill=False)
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
plt.axvline(x=0, color='w', linewidth=2)
plt.axvline(x=10, color='w', linewidth=2)
plt.axvline(x=15, color='w', linewidth=2)
plt.axvline(x=20, color='w', linewidth=2)
plt.axvline(x=25, color='w', linewidth=2)
plt.axvline(x=30, color='w', linewidth=2)
plt.axvline(x=35, color='w', linewidth=2)
plt.axvline(x=40, color='w', linewidth=2)
plt.axvline(x=45, color='w', linewidth=2)
plt.axvline(x=50, color='w', linewidth=2)
plt.axvline(x=55, color='w', linewidth=2)
plt.axvline(x=60, color='w', linewidth=2)
plt.axvline(x=65, color='w', linewidth=2)
plt.axvline(x=70, color='w', linewidth=2)
plt.axvline(x=75, color='w', linewidth=2)
plt.axvline(x=80, color='w', linewidth=2)
plt.axvline(x=85, color='w', linewidth=2)
plt.axvline(x=90, color='w', linewidth=2)
plt.axvline(x=95, color='w', linewidth=2)
plt.axvline(x=100, color='w', linewidth=2)
plt.axvline(x=105, color='w', linewidth=2)
plt.axvline(x=110, color='w', linewidth=2)
plt.axvline(x=120, color='w', linewidth=2)
plt.axvline(x=line_of_scrimmage, color='y', linewidth=3)
plt.axhline(y=0, color='w', linewidth=2)
plt.axhline(y=53.3, color='w', linewidth=2)
plt.text(x=18, y=2, s= '1', color='w')
plt.text(x=21, y=2, s= '0', color='w')
plt.text(x=28, y=2, s= '2', color='w')
plt.text(x=31, y=2, s= '0', color='w')
plt.text(x=38, y=2, s= '3', color='w')
plt.text(x=41, y=2, s= '0', color='w')
plt.text(x=48, y=2, s= '4', color='w')
plt.text(x=51, y=2, s= '0', color='w')
plt.text(x=58, y=2, s= '5', color='w')
plt.text(x=61, y=2, s= '0', color='w')
plt.text(x=68, y=2, s= '4', color='w')
plt.text(x=71, y=2, s= '0', color='w')
plt.text(x=78, y=2, s= '3', color='w')
plt.text(x=81, y=2, s= '0', color='w')
plt.text(x=88, y=2, s= '2', color='w')
plt.text(x=91, y=2, s= '0', color='w')
plt.text(x=98, y=2, s= '1', color='w')
plt.text(x=101, y=2, s= '0', color='w')

ax.set_xticks([0, 120])
ax.set_yticks([0, 53.3])
ax.set_xticklabels(['', ''])
ax.set_yticklabels(['', ''])
ax.tick_params(axis=u'both', which=u'both', length=0)
ax.add_artist(circ)
ax.set_facecolor("#2ca25f")
plt.title("GL and GR collide, injuring GL (helmet to helmet friendly fire).\nGameKey 296, Play ID 2667. TEN punting to JAX.")
plt.show()


# #### **Going for it on fourth down**
# While somewhat out of the scope of this study, there is considerable outside evidence that teams are overly conservative on fourth down, choosing to punt when they would be better off attempting to move the chains. If the fair catch bonus of 5 yards were to encourage more teams to go for it rather than punt, that would seem to a positive side effect.

# ### **Proposal 2** : require single coverage on gunners.
#  <a id="proposal_2"></a>
# Does single coverage result in fewer injuries? To start to analyze this, we classify plays as having single, hybrid, or double coverage, and see whether or not an injury occurred. Because teams are more likely to opt for double coverage in plays where the field is long (i.e. the punt is less likely to be coffin cornered or result in touchback), we need to bucket our data in two dimensions: (a) choice of coverage and (b) distance from line of scrimmage to the end zone.
#  

# In[ ]:


ppr = pd.read_csv('../input/play_player_role_data.csv')
ppr['Role'] = ppr['Role'].map(lambda x: re.sub('[oi0-9]', '', x))
roles = ppr['Role'].unique()

ppr = pd.concat([ppr, pd.get_dummies(ppr['Role'])], axis=1)
ppr = ppr.groupby(['Season_Year', 'GameKey', 'PlayID'])[roles].sum()
ppr = ppr.reset_index()

vi = pd.read_csv('../input/video_review.csv')
vi = vi[['Season_Year', 'GameKey', 'PlayID', 'GSISID']]

ppr = ppr.merge(vi, on=['Season_Year', 'GameKey', 'PlayID'], how='left')
ppr['Injury'] = 0
ppr['const'] = 1

ppr.loc[ppr['GSISID'].notnull(), 'Injury'] = 1

play_information = pd.read_csv('../input/play_information.csv')
def extract_recv_yards(s):
    m = re.search('for ([0-9]+) yards', s)
    if m:
        return int(m.group(1))
    elif re.search('for no gain', s):
        return 0
    else:
        return np.nan
play_information['recv_length'] = play_information['PlayDescription'].map(
        lambda x: extract_recv_yards(x))
play_information = play_information[['Season_Year', 'GameKey', 'PlayID', 'YardLine', 'Poss_Team', 'recv_length']]
play_information['yards_to_go'] = play_information['YardLine'].map(lambda x: int(x[-2:]))
play_information['back_half'] = play_information.apply(lambda x:
    x['YardLine'].startswith(x['Poss_Team']), axis = 1)
play_information.loc[play_information['back_half']==1, 'yards_to_go'] = 100 - (
    play_information.loc[play_information['back_half']==1, 'yards_to_go'])

play_information = play_information[['Season_Year', 'GameKey', 'PlayID', 'yards_to_go', 'recv_length']]
ppr = ppr.merge(play_information, on=['Season_Year', 'GameKey', 'PlayID'], how='inner')

col = 'VR_VL'
ppr[col] = ppr.loc[:, ['VR', 'VL']].sum(axis=1)
ptiles = np.percentile(ppr[col], [0, 25, 50, 75, 100])

ppr['yards'] = ''
ppr['coverage'] = ''
ppr.loc[ppr[col]==2, 'coverage'] = 'single'
ppr.loc[ppr[col]==3, 'coverage'] = 'hybrid'
ppr.loc[ppr[col]==4, 'coverage'] = 'double'


# This graph below shows that receiving teams choose single coverage 84% of the time when the field of play is effectively short, e.g. there are fewer than 50 yards from the line of scrimmage to the end zone. On the other hand, receiving teams choose double coverage, single coverage, and hybrid coverage with more or less even probability when there are more than 70 yards between the line of scrimmage and the goal line. 

# In[ ]:


### For graphing, keep track of counts of plays by single, hybrid, or double coverage
### Sort by yards-to-go.
x_single = [0.00, 1.00, 2.00]
x_hybrid = [0.25, 1.25, 2.25]
x_double = [0.50, 1.50, 2.50]
y_single = []
y_hybrid = []
y_double = []
injury_rate = {'Yards to Go': [], '30-50': [], '50-70': [], '70-100': []}
for i in range(2, 5):
    if i == 2:
        mode = 'Single'
    elif i == 3:
        mode = 'Hybrid'
    elif i == 4:
        mode = 'Double'
    injury_rate['Yards to Go'].append(mode)
    for r in ([30, 50], [50, 70], [70, 100]):
        ii = (ppr['yards_to_go']>=r[0])&(ppr['yards_to_go']<r[1])
        if (r[0]==30) & (r[1]==50):
            ppr.loc[ii, 'yards'] = '30 to 50'
        elif (r[0]==50) & (r[1]==70):
            ppr.loc[ii, 'yards'] = '50 to 70'
        elif (r[0]==70) & (r[1]==100):
            ppr.loc[ii, 'yards'] = '70 to 100'

        pprt = ppr.loc[ii]
        if len(pprt) == 0:
            pass
        iii = (pprt[col] == i)
        if sum(iii) == 0:
            continue
        # Keep track of coverage choice by yards to go
        if mode == 'Single':
            y_single.append(sum(iii) / sum(ii)) # Ratio of times single coverage is elected
        elif mode == 'Hybrid':
            y_hybrid.append(sum(iii) / sum(ii)) # Ratio of times hybrid coverage is elected
        elif mode == 'Double':
            y_double.append(sum(iii) / sum(ii)) # Ratio of times double coverage is elected
        if r[0] == 30:
            injury_rate['30-50'].append(1000 * pprt.loc[iii, 'Injury'].mean())
        elif r[0] == 50:
            injury_rate['50-70'].append(1000 * pprt.loc[iii, 'Injury'].mean())
        elif r[0] == 70:
            injury_rate['70-100'].append(1000 * pprt.loc[iii, 'Injury'].mean())
            
injury_rate['Yards to Go'].append("All")
injury_rate['30-50'].append(1000 * ppr.loc[
    (ppr['yards_to_go']>=30) & (ppr['yards_to_go']<50), 'Injury'].mean())
injury_rate['50-70'].append(1000 * ppr.loc[
    (ppr['yards_to_go']>=50) & (ppr['yards_to_go']<70), 'Injury'].mean())
injury_rate['70-100'].append(1000 * ppr.loc[
    (ppr['yards_to_go']>=70) & (ppr['yards_to_go']<100), 'Injury'].mean())
injury_rate = pd.DataFrame(injury_rate)
injury_rate = injury_rate[['Yards to Go', '30-50', '50-70', '70-100']]

fig = plt.figure(figsize = (6.5, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar(x_single, y_single, color='#7fc97f', width=.25, label='Single Coverage')
plt.bar(x_hybrid, y_hybrid, color='#beaed4', width=.25, label='Hybrid Coverage')
plt.bar(x_double, y_double, color='#fdc086', width=.25, label='Double Coverage')
for i in range(3):
    plt.text(x_single[i]-0.06, y_single[i]+0.02, '{:.0f}%'.format(y_single[i] * 100))
    plt.text(x_hybrid[i]-0.06, y_hybrid[i]+0.02, '{:.0f}%'.format(y_hybrid[i] * 100))
    plt.text(x_double[i]-0.06, y_double[i]+0.02, '{:.0f}%'.format(y_double[i] * 100))
    
plt.legend()
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.00])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xticks([0.25, 1.25, 2.25])
ax.set_xticklabels(['30-50 Yards', '50-70 Yards', '70-100 Yards'])
ax.set_xlabel('Yards from Line of Scrimmage to End Zone')
ax.set_ylabel('Coverage Choice (%)')
sns.despine(top=True, right=True)

plt.title("Coverage Choice vs. Yards between Line of Scrimmage and End Zone\n")
plt.show()


# The injury rate is higher on double coverage and hybrid coverage vs. single coverage, at least in plays with longer fields. E.g. looking at all coverage types (the last row of the table below), there are 3 injuries per 1000 plays with 30-50 yards-to-go versus 9 injuries per 1000 plays when there are 70-100 yards-to-go. Because choice of coverage is correlated with yards-to-go, we need to control for both effects when trying to evaluate whether coverage is related to injury rate.
# 
# By subsetting plays on both dimensions, we can see that even when controlling for yards to go, double coverage appears to result in substantially higher injury rates than single and hybrid coverage. That said, the statistical significance of this finding is limited by the small sample size, as there are too few observed injuries in the provided data to draw sharp inferences. Given that we are not strongly confident that double coverage increases the injury rate, we need at least to be sure that prohibiting double coverage does not reduce the quality of game play.
# 
# ** Table of Injury Rate vs. Yards-to-Go and Coverage Choice **

# In[ ]:


injury_rate.style.set_precision(2).hide_index()


# One measure of quality of play is the length of punt return, with longer punt returns being (in my mind) associated with more exciting games. Let us look at the distribution of punt return length for punts received with single, hybrid, and double coverage. Again, we group by yards-to-go, since coverage decisions appear partly conditioned on the length of the playable field. The "violin plots" below show the density and distribution of punt returns, condtioned on coverage choice and yards to go. At first glance, return length appears to be indistinguishable between single, hybrid, and double coverage, suggesting that the return team would do just as well electing single coverage.
# 
# ** Distribution of Punt Return Length Conditioned on Yards-to-Go and Coveage Choice **

# In[ ]:


fig = plt.figure(figsize = (8.5, 5.5))
ax = plt.subplot2grid((1, 1), (0, 0))
pal = {'single': '#7fc97f', 'hybrid': '#beaed4', 'double': '#fdc086'}
ax = sns.violinplot(x="yards", y="recv_length", hue="coverage",
                    data=ppr, palette=pal,
                    order=['30 to 50', '50 to 70', '70 to 100'],
                    hue_order=['single', 'hybrid', 'double'],
                    cut=0)
ax.legend().remove()
sns.despine(top=True, right=True)
plt.legend()
plt.show()


# ### **Proposal 3**: Require helmet monitors to better measure deceleration at time of tackle.
# <a id="proposal_3"></a>
# As the plots below show, deceleration is greater for tackles generating injuries than for tackles without injuries. On the other hand, velocities are somewhat higher on non-injury players and plays. Intuitively, there is nothing dangerous about running fast, unless you have to stop suddenly. 
# 
# Because NGS data is provided only at 100 millisecond resolution, the maximum observable deceleration is approximately 100 m/s/s (assuming a person running flat out at 10 m/s at time t is moving at 0 m/s  in the next NGS data point, 100 milli seconds later) , or less than 10g's. On the other hand, it is possible for players to have experienced much greater g-forces, and for those forces to be stronger at the helmet than in the rest of the body. Without helmet sensor data, there is a finite upper bound on the measurable deceleration forces, one that is too coarse to be really useful at distinguishing injury from safe and routine play.
# 

# In[ ]:


# Collate data
game_data = pd.read_csv('../input/game_data.csv')
play_information = pd.read_csv('../input/play_information.csv')
player_punt_data = pd.read_csv('../input/player_punt_data.csv')
player_punt_data = player_punt_data.groupby('GSISID').head(1).reset_index()
play_player_role_data = pd.read_csv('../input/play_player_role_data.csv')
video_review = pd.read_csv('../input/video_review.csv')

combined = game_data.merge(play_information.drop(['Game_Date'], axis=1),
                        on=['GameKey', 'Season_Year', 'Season_Type', 'Week'])
combined = combined.merge(play_player_role_data,
                        on=['GameKey', 'Season_Year', 'PlayID'])
combined = combined.merge(player_punt_data, on=['GSISID'])

combined = combined.merge(video_review, how='left',
                          on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'])
combined['injury'] = 0
combined.loc[combined['Player_Activity_Derived'].notnull(), 'injury'] = 1

ngs_files = ['../input/NGS-2016-pre.csv',
             '../input/NGS-2016-reg-wk1-6.csv',
             '../input/NGS-2016-reg-wk7-12.csv',
             '../input/NGS-2016-reg-wk13-17.csv',
             '../input/NGS-2016-post.csv',
             '../input/NGS-2017-pre.csv',
             '../input/NGS-2017-reg-wk1-6.csv',
             '../input/NGS-2017-reg-wk7-12.csv',
             '../input/NGS-2017-reg-wk13-17.csv',
             '../input/NGS-2017-post.csv']

max_decel_df = []
for filename in ngs_files:
    logging.info("Loading file " + filename)
    group_keys = ['Season_Year', 'GameKey', 'PlayID', 'GSISID']
    df = pd.read_csv(filename, parse_dates=['Time'])
    logging.info("Read file " + filename)

    df = df.sort_values(group_keys + ['Time'])
    df['dx'] = df.groupby(group_keys)['x'].diff(1)
    df['dy'] = df.groupby(group_keys)['y'].diff(1)
    df['dis'] = (df['dx']**2 + df['dy']**2)**0.5
    df['dt'] = df.groupby(group_keys)['Time'].diff(1).dt.total_seconds()
    df['velocity'] = 0
    ii = (df['dis'].notnull() & df['dt'].notnull() & (df['dt']>0))
    df.loc[ii, 'velocity'] = df.loc[ii, 'dis'] / df.loc[ii, 'dt']
    df['velocity'] *= 0.9144 # Convert yards to meters
    df['deceleration'] = -1 * df.groupby(group_keys)['velocity'].diff(1)
    df['velocity'] = df.groupby(group_keys)['velocity'].shift(1)

    # Only look at the one second window around each tackle
    df['Event'] = df.groupby(group_keys)['Event'].ffill(limit=5)
    df['Event'] = df.groupby(group_keys)['Event'].bfill(limit=5)

    t_df = df.loc[df['Event']=='tackle']

    t_max_decel = t_df.loc[t_df.groupby(['Season_Year', 'GameKey', 'PlayID', 'GSISID'])['deceleration'].idxmax()]
    t_max_decel = t_max_decel[['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'deceleration']].rename(columns={'deceleration': 'deceleration_at_tackle'})
    
    t_max_velocity = t_df.loc[t_df.groupby(['Season_Year', 'GameKey', 'PlayID', 'GSISID'])['velocity'].idxmax()]
    t_max_velocity = t_max_velocity[['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'velocity']].rename(columns={'velocity': 'velocity_at_tackle'})

    max_decel = t_max_velocity.merge(t_max_decel, on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'],
                                      how='outer')
    max_decel_df.append(max_decel)

max_decel_df = pd.concat(max_decel_df)
combined = combined.merge(max_decel_df, on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'], how='left')


combined['tackle_injury'] = combined['Player_Activity_Derived'].isin(['Tackled', 'Tackling'])
### Original work by Halla Yang


# #### Distribution of Player Velocities and Deceleration at time of Event Tackle, Injury Plays vs. non-Injury Plays 

# In[ ]:


fig = plt.figure(figsize = (6.5, 5.0))
ax = plt.subplot2grid((1, 1), (0, 0))
inj = combined.loc[(combined['injury']==1)&(combined['velocity_at_tackle'].notnull())
                  &(combined['deceleration_at_tackle'].notnull())]
ax = sns.kdeplot(inj.velocity_at_tackle, inj.deceleration_at_tackle,
                 cmap="Reds")
notinj = combined.loc[(combined['injury']==0)&(combined['velocity_at_tackle'].notnull())
                  &(combined['deceleration_at_tackle'].notnull())]
ax = sns.kdeplot(notinj.velocity_at_tackle, notinj.deceleration_at_tackle,
                 cmap="Blues")
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
plt.xlabel("Velocity at Tackle")
plt.ylabel("Deceleration at Tackle")
sns.despine(top=True, right=True)
plt.title("Velocity/Deceleration at Time of Tackle for Injuries shown in Red, Non-Injuries in Blue")
plt.subplots_adjust(top=0.9)
plt.show()


# ### **Additional Observations and Opinions**
# <a id="additional_observations"></a>
# #### The Helmet Rule of 2018 
# * Many of the concussions on punt returns could possibly have been avoided if the new Helmet Rule introduced in 2018 had been in effect. If data from the 2018 season were available, it could be used to quantify the effect of the Helmet Rule in reducing the number of concussions. 
# 
# #### Making gunners wait at the line of scrimmage
# * In my opinion, the possibility of a fake punt and a throw by the punter adds excitement to a punt play. Requiring gunners to wait at the line of scrimmage until the ball is kicked would make these fake punt plays impractical and overly reduce the unpredictability and excitement of punt plays. 
# 
# #### The CFL Rule
# * The CFL rule giving the returner a 5 yard buffer of protection seems to pre-assume that a punt is to be made. What if instead the punter rolls out of the pocket and throws a long pass on the run to a 'gunner'? Unpredictability, e.g. not knowing whether the punting team will really punt, is for me a key factor in enjoying the game, and I love watching the games of great coaches who are not afraid to gamble and take risks in 4th and short situations. 
# 
# ### ** Final Thoughts **
# <a id="final_thoughts"></a>
# In summary, I have proposed three rule changes that all strive to increase safety while importantly preserving unpredictability, excitement, and the possibility for players to make great plays. The NFL is about great individual and team performance, and we need to protect the players' ability to showcase their talent and to be rewarded for their hard work, both in the near term when they are on the field and in the long term when they serve as ambassadors for the game, long after they have hung up their cleats. Thank you for the opportunity to work on this important and interesting challenge.
# 
# 
