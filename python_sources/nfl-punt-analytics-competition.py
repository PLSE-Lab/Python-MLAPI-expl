#!/usr/bin/env python
# coding: utf-8

# # NFL Punt Analytics Competition
# ### Andrew Rall

# In[ ]:


import pandas as pd
import re
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt


# ## Introduction and Motivation
# 
# Punt returns are often perceived as one of the most dangerous yet also exciting plays in football. First, I explore the data and verify that the first part of this claim has validity. From this exploration, I attempt to come up with a minimally invasive yet highly effective rule change to reduce the occurrence of concussions on punt plays.

# ## Loading the Data

# #### Loading the Next Gen Stats for Punts on Which a Concussion was Sustained

# In[ ]:


get_ipython().run_cell_magic('capture', '', "ngs_2016_pre_conc = pd.read_csv('../input/punt-data/con_ngs_2016_pre.csv')\nngs_2016_reg_wk1_6 = pd.read_csv('../input/punt-data/con_ngs_2016_wk1_6.csv')\nngs_2016_reg_wk7_12 = pd.read_csv('../input/punt-data/con_ngs_2016_wk7_12.csv')\nngs_2016_reg_wk13_17 = pd.read_csv('../input/punt-data/con_ngs_2016_wk13_17.csv')\nngs_2017_pre_conc = pd.read_csv('../input/punt-data/con_ngs_2017_pre.csv')\nngs_2017_reg_wk1_6 = pd.read_csv('../input/punt-data/con_ngs_2017_wk1_6.csv')\nngs_2017_reg_wk7_12 = pd.read_csv('../input/punt-data/con_ngs_2017_wk7_12.csv')\nngs_2017_reg_wk13_17 = pd.read_csv('../input/punt-data/con_ngs_2017_wk13_17.csv')\n\nngs_list = [ngs_2016_pre_conc, ngs_2016_reg_wk1_6, ngs_2016_reg_wk7_12, ngs_2016_reg_wk13_17, ngs_2017_pre_conc, ngs_2017_reg_wk1_6, ngs_2017_reg_wk7_12, ngs_2017_reg_wk13_17]\nngs_conc = pd.concat(ngs_list)\nngs_conc['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in ngs_conc.GameKey], [str(x) for x in ngs_conc.PlayID], [str(x) for x in ngs_conc.Season_Year])]\nngs_conc = ngs_conc.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)")


# In[ ]:


ngs_conc.head()


# #### Loading and Combining the Play Information and Injury Data

# In[ ]:


video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_footage_control = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')
video_footage_injury = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
punt_data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')

regexp_no_punt = re.compile('(No)\s(Play)|(Delay)\s(of)\s(Game)|(Aborted)|(pass)|(False)\s(Start)')
no_punt = [regexp_no_punt.search(x) == None for x in punt_data.PlayDescription]

### There was a penalty on the play so the punt didn't count but there was still a concussion
no_punt[4018] = True
punt_data = punt_data[no_punt]

video_review['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in video_review.GameKey], [str(x) for x in video_review.PlayID], [str(x) for x in video_review.Season_Year])]
video_footage_injury['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in video_footage_injury.gamekey], [str(x) for x in video_footage_injury.playid], [str(x) for x in video_footage_injury.season])]
video_footage_control['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in video_footage_control.gamekey], [str(x) for x in video_footage_control.playid], [str(x) for x in video_footage_control.season])]
punt_data['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in punt_data.GameKey], [str(x) for x in punt_data.PlayID], [str(x) for x in punt_data.Season_Year])]

injury_data = video_review.merge(video_footage_injury, on='unique_id')


# In[ ]:


data = punt_data.merge(injury_data, on='unique_id', how='outer', suffixes=('', '_drop'))
data = data.iloc[:, np.r_[0:15, 19:25]]
data.info()


# ## Exploring the Data

# First, I categorize the most common outcomes of a punt to gain an understanding of what happens when a punt is executed. A return is by far the most common outcome of a punt, occuring on over 40 percent of punt plays. This is a simple yet very important statistic when considering potential rule changes as it illustrates how important punt returns are to the game. In the plot below, note that the "muffed" category includes both punts that were dropped immediately by the returner and punts that were returned but then fumbled on the return. Thus, the "returned" category only includes punts that were returned without being muffed or fumbled.

# In[ ]:


regexp_fair_catch = re.compile('(fair)\s(catch)')
fair_catch = [regexp_fair_catch.search(x) != None for x in punt_data.PlayDescription]
data["fair_catch"] = fair_catch

regexp_bounds = re.compile('(out)\s(of)\s(bounds)\.')
out_of_bounds = [regexp_bounds.search(x) != None for x in punt_data.PlayDescription]
data["out_of_bounds"] = out_of_bounds

regexp_touchback = re.compile('(Touchback)\.')
touchback = [regexp_touchback.search(x) != None for x in punt_data.PlayDescription]
data["touchback"] = touchback

regexp_muffs = re.compile('(MUFFS)|(FUMBLE)|(Fumble)|(fumble)')
muffed = [regexp_muffs.search(x) != None for x in punt_data.PlayDescription]
data["muffed"] = muffed

regexp_downed = re.compile('(downed)')
downed = [regexp_downed.search(x) != None for x in punt_data.PlayDescription]
data["downed"] = downed

regexp_blocked = re.compile('(BLOCKED)')
blocked = [regexp_blocked.search(x) != None for x in punt_data.PlayDescription]
data["blocked"] = blocked

regexp_returned = re.compile("[a-zA-Z\.]*\sto\s[A-Z]*\s[0-9]*\sfor\s[-0-9]*\s|[a-zA-Z\.]*\s(pushed)\sob\sat\s[A-Z]*\s[0-9]*\sfor\s[-0-9]*\s|[a-zA-Z\.]*\sto\s[0-9]*\sfor\s[-0-9]*\s|[a-zA-Z\.]*\sto\s[A-Z]*\s[0-9]*\sfor\sno\sgain|[a-zA-Z\.]*\sran\sob\sat\s[A-Z]*\s[0-9]*\sfor|[a-zA-Z\.]*\spushed\sob\sat\s[A-Z]*\s[0-9]*\sfor|[a-zA-Z\.]*\sfor\s[0-9]*\syards, TOUCHDOWN|[a-zA-Z\.]*\spushed\sob\sat\s[0-9]*\sfor\s|[a-zA-Z\.]*\sran\sob\sat\s[0-9]*\sfor\s")
returned = [regexp_returned.search(x) != None for x in punt_data.PlayDescription]
data["returned"] = returned & ~np.array(muffed) & ~np.array(fair_catch) & ~np.array(out_of_bounds) & ~np.array(touchback) & ~np.array(downed) & ~np.array(blocked)

### Manually Change Punts that were challenged

data.loc[0, 'downed'] = False
data.loc[155, 'downed'] = False
data.loc[161, 'downed'] = False
data.loc[661, 'downed'] = False
data.loc[1254, 'downed'] = False
data.loc[1415, 'downed'] = False
data.loc[1649, 'downed'] = False
data.loc[1745, 'downed'] = False
data.loc[3796, 'downed'] = False


# In[ ]:


count_data = pd.DataFrame({'Type of Outcome': ["returned", "fair_catch", "downed", "out_of_bounds", "touchback", "muffed", "blocked"],
                         'Proportion of Executed Punts' : np.array([sum(data.returned), sum(data.fair_catch), sum(data.downed), sum(data.out_of_bounds),
                                    sum(data.touchback), sum(data.muffed), sum(data.blocked)])/len(data)})

fig, ax = plt.subplots(figsize=(15, 10))
plt.title('Common Outcomes of a Punt')
sns.barplot(x='Type of Outcome', y='Proportion of Executed Punts', data=count_data);


# Next, one simple notion that I thought was important to address was the type of contact that resulted in a concussion. Interestingly, there is an even mix of helmet-to-helmet and helmet-to-body contact with a very small amount of hemlet-to-ground contact. This is in line with the new rule instituted in Spring 2018 that prevents players from "lowering the head to initiate contact". The effects of this rule change are still to be observed, but the initiation of this rule indicates that a new and separate tackling technique rule is unlikely to practical or effective.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Type of Contact that Resulted in a Concussion")
data.Primary_Impact_Type.value_counts().plot(kind='bar');


# It is also very important when considering rule changes to reduce the occurence of concussions during punt plays to look at the action the concussed players were taking at the time of the injury. Somewhat counterintuitively, most of the concussed players on punts from 2016-2017 were in the process of tackling another player, while the action with the fewest concussions is a player being tackled. However, it is important to remember that there are 11 players on the field that are attempting to tackle the returner (or get blocked in the process), but only one player at a time that is returning the ball and has the potential to be tackled. Still, this plot raises a new question of whether punt returns are really more dangerous than other outcomes of executed punts. This question will be resolved shortly.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Action Taken by Concussed Player")
data.Player_Activity_Derived.value_counts().plot(kind='bar');


# As expected, the vast majority of concussions are not sustained from friendly fire. This is still important as it eliminates the need to created a rule that addresses friendly fire.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Occurence of Friendly Fire Resulting in a Concussion")
data.Friendly_Fire.value_counts().plot(kind='bar');


# Most importantly, and to answer the question posed above, the vast majority of concussions were sustained on punt plays where the ball was returned. So to reiterate, **even though most of the players who sustained a concussion were not returning the ball, the majority of concussions were sustained on returned punts**. This motivates the thought, which was to be expected, that punt returns are dangerous. But as noted earlier, removing them entirely would be too invasive to the game, so how could punt returners be incentivized to call fair catch (on which concussions are very unlikely) instead? 

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Outcomes of Punts Plays that Resulted in a Concussion  ")
data[data.Player_Activity_Derived.notna()].iloc[:, 21:].sum().plot(kind='bar');


# To build off of this thought of incentivizing punt returners to call fair catch, it is first important to get an idea of how long the average punt return is. Obviously the distribution has a strong right skew, resulting in a mean larger than the median.

# In[ ]:


regexp_ret_dist = re.compile('(for\sno\sgain)|(for\s[-0-9]*\s(yards|yard))')
ret_dist = np.array([regexp_ret_dist.search(x).group(0) for x in data[data.returned].PlayDescription])

ret_dist = np.array([re.search('[-0-9]+|no', x).group(0) for x in ret_dist])
ret_dist[ret_dist == 'no'] = 0
ret_dist = [int(x) for x in ret_dist]
print("Mean Punt Return Distance (yards): " + str(np.mean(ret_dist)))
print("Median Punt Return Distance (yards): " + str(np.median(ret_dist)))


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Distribution of Punt Return Distance (yards)")
sns.distplot(ret_dist);


# ## Rule Change Proposal: Provide x Additional Yards from the Spot of a Fair Catch

# Again, fewer returns and more fair catches will reduce the occurence of concussions during punt plays. The thought process should thus be to incentivize punt returners to call fair catch without eliminating punt returns entirely, which as supported above is a significant part of the game. Removing punt returns entirely would be a far too invasive change to the game.
# 
# Akin to moving touchbacks from the 20 to the 25, the first rule change that I am proposing is to give either 5 or 10 free yards to the returning team from the spot that a fair catch is called. Providing either 5 or 10 additional yards could be decided by surveying punt returners and/or special teams coaches to determine how much more "likely" they would be to call fair catch at any given point on the field if 5 or 10 additional yards would be given. Five free yards would obviously have a less invasive impact on the game, but might not provide enough of an incentive for punt returners. Obviously one of the most direct potential side effects of this rule change would be an increase in scoring. However, even if 10 free yards were given, the additional points scored per game would be [minimal](https://www.cmusportsanalytics.com/nfl-expected-points-nflscrapr-part-1-introduction-expected-points/) and similar for all teams. Even if there was an unexpected increase in points per game, this would not necessarily be a bad change to the game since higher scoring games are often deemed more entertaining.
# 
# Some other potential side effects of this rule change include:
# * fewer penalties (fewer returns should result in fewer holding/illegal block in the back penalites, a postive impact on the quality of game)
# * encouraging punters to kick the ball away from punt returners (also a much safer punt outcome than a return)
# * encouraging punt returners to chase after a ball to call fair catch (maintaining the excitement of decision making in punt plays)
# * a decrease in Net Punting Average (but no change in Gross Punting Average)
# 
#     
# This is a very simple and intuitive change that would require further research to determine it's impact. Next I wil explore the Next Gen Stats to see if either blocking schemes or punt formations have a significant impact on concussion rates.

# ## Exploring the Next Gen Stats

# The first image that I produce below is the line set formation paired with the path taken by the concussed player for each of the 37 punt plays on which a concussion was sustained. The second image that I generate is the line set formation for each of the 37 video footage control punt plays. 
# 
# There are two different types of formational, tackling, or blocking rule changes that can be made: prior to the snap or during the play. In regard to prior to the snap rule changes, the fundamental question is whether there is an underlying difference between the line set formations of the 37 concussion punts and the 37 control punts. The short answer is no. The more precise answer is that there is not enough data to tell. With there being only 37 punts plays that resulted in a concussion out of over 6000, it is not reasonable or practical to assume that the line set formations for those 37 punts is different than the other 6000+ punts.
# 
# In regard to the during the play rule changes, the fundamental question is whether there is an underlying difference between the paths (including contact) taken by the 37 concussed players and the more than 132,000 (22\*6000) paths taken by other players. Again we run into the issue of not having enough concussion data. There are nonparametric statistical tests that can be used to address this issue, but other than the obvious helmet-to-helmet or helmet-to-body contact, there is unlikely to be any practical blocking or tackling rule changes that can be made. 
# 
# That being said, there is an intuitive benefit to reducing the occurence of high speed collisions. These collisions are more likely to cause concussions in addition to other injuries. However, when looking at the rule changes made to reduce high speed collisions on kickoff returns, these are already characteristics present in punting plays. There is no running start, no wedge blocking, and the returning team blockers are required to run down field "with" the coverage team. 
# 
# Because of this, I find it unlikely for there to exist an additional and still practical rule change in regard to starting formation, tackling, or blocking. Thus, my final proposal to decrease the occurence of concussions sustained on punting plays is the rule change outlined above. 
# 

# In[ ]:


ngs_conc['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in ngs_conc.GameKey], [str(x) for x in ngs_conc.PlayID], [str(x) for x in ngs_conc.Season_Year])]
ids = ngs_conc.unique_id.unique()
line_set = ngs_conc[ngs_conc.Event == "line_set"]

### There is no "line_set" event for this punt so I use ball_snap instead to get starting formation
line_set = line_set.append(ngs_conc[(ngs_conc.Event == 'ball_snap') & (ngs_conc.unique_id == '56714072017')])

x_vals = []
y_vals = []

for i in np.arange(0, len(ids)):
    temp = line_set[line_set.unique_id == ids[i]]
    x_vals.append(temp.x.values)
    y_vals.append(temp.y.values)
    
    

ngs_control = pd.read_csv('../input/punt-data/ngs_control.csv')
control_line_set = ngs_control[ngs_control.Event == 'line_set']

### There is no "line_set" event for these punts so I use ball_snap instead to get starting formation
control_line_set = control_line_set.append(ngs_control[(ngs_control.Event == 'ball_snap') & ((ngs_control.unique_id == 42333332017) | (ngs_control.unique_id == 4276892017))])

control_ids = ngs_control.unique_id.unique()

control_x = []
control_y = []

for j in np.arange(0, len(control_ids)):
    temp = control_line_set[control_line_set.unique_id == control_ids[j]]
    control_x.append(temp.x.values)
    control_y.append(temp.y.values)


# In[ ]:


conc_paths = []

fig, ax = plt.subplots(37, 2, figsize=(15, 150))

for i in range(len(ids)):
    ax[i, 0].scatter(x_vals[i],y_vals[i])
    ax[i, 0].set_xlim(0, 120)
    ax[i, 0].set_ylim(0, 54)
    
for index, rows in injury_data.iterrows():
    temp = ngs_conc[(ngs_conc.GSISID == rows.GSISID) & (ngs_conc.unique_id == rows.unique_id)]
    conc_paths.append(temp)
    ax[index, 1].scatter(temp.x, temp.y)
    ax[index, 1].set_xlim(0, 120)
    ax[index, 1].set_ylim(0, 54)


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 150))

for i in range(len(control_ids)):
    plt.subplot(37, 1, i+1)
    plt.scatter(control_x[i], control_y[i])
    plt.xlim(0, 120)
    plt.ylim(0, 54)

