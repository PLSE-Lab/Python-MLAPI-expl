#!/usr/bin/env python
# coding: utf-8

# **NBA TEAM SUCCESS ANALYSIS**  
# The Goal: Analyze NBA data to find what makes teams successful. 
# 
# This notebook covers some analysis from shot chart data. The data was scraped from stats.nba.com and was aggregated into 4 different datasets that I will use to analyze the differences between "good", "average", and "bad" teams. The data includes every shot from every player in the 2017-18 season, including shot type, location of the shot, the player, the team, etc. To make the analysis more concise **four** separate "potential indicators" were specifically analyzed:
# 1. **The Percentage of Field Goals Made that were 3 Pointers (3 Point Field Goals)**
# 2. **Diversity of Shot Zones**
# 3. **Shot Distance**
# 4. **Period (Quarter) and Minutes** 
# 
# These potential indicators will be used to see if any of these have an affect on performance and potentially on how to advise the staff of team on the best course of action e.g. If all the best teams shot from the Left Corner 3, coaches would use that information to plan defensive schemes with a focus on that area in mind. In addition to being a potential defensive tool, coaches can potentially see and copy the trends seen in the teams that perform better.
# 

# For the data, I split the teams up into three different categories: "Good", "Avg", and "Bad". These categories represent the different "classes" present in the NBA for my purposes. Using the 2017-18 season records, I selected the top 6 teams in the NBA for the "Good" category, the middle 18 teams for the "Avg" category, and the bottom 6 teams for the "Bad" category (sorted by season record). 
# 
# Formulas:
# FG% = Shots Made / Shots Attempted  
# eFG% = (Shots Made + 0.5 * 3 Point Shots Made) / Shots Attempted
# 
# eFG% helps give a more accurate view of true story because weighs the 3 pointer as 1.5 times the value of a 2 pointer (which it is)

# tl;dr: I use data from stats.nba.com to analyze 4 different potential indicators to evaluate what makes a team successful to inform whomever (coaches). Indicator 1 showed that the volume of 3s taken isn't too impactful, indicator 2 showed that every team utilizes different Locations on the court the same, indicator 3 showed that shot distance isn't indicative of team success, and indicator 4 showed that coming out strong in the first quarter seems to be helpful and limiting the drop in FG% each quarter is beneficial, but showed  an inconclusive indication between the minutes of the quarter and success.

# **The Graph Generator Code**

# Below is the code used to build the following charts. All I have to do is call the Chart method with the appropriate params to create the graphs necesary for the analysis.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Params for method Chart
Indicator1 = {'%3sMade': {'y': '%3sMade', 'hue': None, 'hue_order': None, 'yaxis': 'The Percent of field goals that was a 3PT Field Goal', 'title': 'The Percentage a team Shoots and makes a 3 Pointer', 'legend': False}, '%3sAttempted': {'y': '%3sAttempted', 'hue': None, 'hue_order': None, 'yaxis': 'The Percent of field goals that was a 3PT Field Goal', 'title': 'The Percentage a team Shoots a 3 Pointer', 'legend': False}}
Indicator2 = {'SHOTS_ATTEMPTED by LOCATION': {'y': 'SHOTS_ATTEMPTED', 'hue': 'LOCATION', 'hue_order': ['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'], 'yaxis': 'Amount of Shots Attempted', 'title': 'The amount of shots attempted for different locations', 'legend': True}, 'FG% by LOCATION': {'y': 'FG%', 'hue': 'LOCATION', 'hue_order': ['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'], 'yaxis': 'Field Goal Percentage', 'title': 'The FG% for different locations', 'legend': True}, 'eFG% by LOCATION': {'y': 'eFG%', 'hue': 'LOCATION', 'hue_order': ['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'], 'yaxis': 'Effective Field Goal Percentage', 'title': 'The eFG% for different locations', 'legend': True}}
Indicator3 = {'Bucket List': {'y': 'SHOTS_ATTEMPTED', 'hue': 'SHOT_DISTANCE', 'hue_order': range(40), 'yaxis': 'Amount of Shots Attempted', 'title': 'The amount of shots attempted for every distance', 'legend': True}, 'SHOTS_ATTEMPTED by SHOT_DISTANCE': {'y': 'SHOTS_ATTEMPTED', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Amount of Shots Attempted', 'title': 'The amount of shots attempted for different distances', 'legend': True}, 'SHOTS_MADE by SHOT_DISTANCE': {'y': 'SHOTS_MADE', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Amount of Shots Made', 'title': 'The amount of shots made for different distances', 'legend': True}, 'FG% by SHOT_DISTANCE': {'y': 'FG%', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Field Goal Percentage', 'title': 'The FG% for different distances', 'legend': True}, 'eFG% by SHOT_DISTANCE': {'y': 'eFG%', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Effective Field Goal Percentage', 'title': 'The Effective FG% for different distances', 'legend': True}}
Indicator4 = {'SHOTS_MADE by PERIOD': {'y': 'eFG%', 'hue': 'PERIOD', 'hue_order': [1,2,3,4], 'yaxis': 'Effective Field Goal Percentage', 'title': 'The Effective FG% for different Periods/Quarters', 'legend': True}, 'eFG% by MINUTES_REMAINING': {'y': 'eFG%', 'hue': 'MINUTES_REMAINING', 'hue_order': [11,10,9,8,7,6,5,4,3,2,1,0], 'yaxis': 'Effective Field Goal Percentage', 'title': 'The Effective FG% for every Minute remaining in the Quarter', 'legend': True}}
Indicator = [None, Indicator1, Indicator2, Indicator3, Indicator4]

sns.set(style="darkgrid")

#Method that build chart based on parameters 
def Chart(indicator, chart, ncol=1, x=0.8):
	data = pd.read_csv('../input/Indicator' + str(indicator) + '.csv')
	ax = sns.barplot(x='TEAM_RATING', y=Indicator[indicator][chart]['y'], hue=Indicator[indicator][chart]['hue'], data=data, order=['Good','Avg','Bad'], hue_order=Indicator[indicator][chart]['hue_order'], ci=None)
	ax.set(xlabel='Team Rating', ylabel=Indicator[indicator][chart]['yaxis'])
	ax.set_title(Indicator[indicator][chart]['title'])
	if Indicator[indicator][chart]['legend']:
		plt.legend(loc=(1,x), ncol=ncol)
	plt.show()

    


# **Indicator 1:** Percentage of Field Goals that were 3 PT Field Goals

# In[ ]:


plt.figure(figsize=(10,5))
Chart(1, '%3sAttempted')
plt.figure(figsize=(10,5))
Chart(1, "%3sMade")


# The graphs show that the better teams shoot more 3s and make more 3s, but this doesn't necesarily mean that their FG% is higher. However, in this case their FG% is slightly higher. There isnt't anything of significance here, besides that the better teams shoot slightly more 3s.

# **Indicator 2:** Diversity of Shot Zones

# In[ ]:


plt.figure(figsize=(15,8))
Chart(2,'SHOTS_ATTEMPTED by LOCATION')
plt.figure(figsize=(15,8))
Chart(2,'FG% by LOCATION')
plt.figure(figsize=(15,8))
Chart(2,'eFG% by LOCATION')


# These graphs show that the "Good" teams take slightly more Above the Break 3s than the other teams, but their Left and Right Corner threes have a higher FG% than the Above the Break 3. (I think this is because Above the Break 3s are often used for iso/pullup jumpers, which have a lower FG% than Corner 3s because Corner 3s are normally catch and shoot). Another thing is that the good teams have a slightly higher FG% in the Restricted Area (about 4% better than bad teams). An interesting thing to note is that the good teams have the best Mid Range FG% but take the least Mid Range shots. But let's investigate this further by filtering out the Rockets and Warriors out of the data because I suspect they are outliers causing this.

# In[ ]:


#Remove the warriors and rockets
data = pd.read_csv('../input/Indicator2.csv')
data = data[data['TEAM_NAME'] != 'Houston Rockets'] 
data = data[data['TEAM_NAME'] != 'Golden State Warriors']

plt.figure(figsize=(15,8))
ax = sns.barplot(x='TEAM_RATING', y='SHOTS_ATTEMPTED', hue='LOCATION',data=data, order=["Good", "Avg", "Bad"], hue_order=['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'])
ax.set(xlabel='Team Rating', ylabel='Amount of Shots Attempted')
ax.set_title('The amount of shots attempted for different locations w/out the Rockets and Warriors')	
plt.legend(loc=(1,0.8))
plt.show()
    
plt.figure(figsize=(15,8))
ax = sns.barplot(x='TEAM_RATING', y='FG%', hue='LOCATION',data=data, order=["Good", "Avg", "Bad"], hue_order=['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'])
ax.set(xlabel='Team Rating', ylabel='Field Goal Percentage')
ax.set_title('The FG% for different locations w/out the Rockets and Warriors')
plt.legend(loc=(1,0.8))
plt.show()

plt.figure(figsize=(15,8))  
ax = sns.barplot(x='TEAM_RATING', y='eFG%', hue='LOCATION',data=data, order=["Good", "Avg", "Bad"], hue_order=['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'])
ax.set(xlabel='Team Rating', ylabel='Effective Field Goal Percentage')
ax.set_title('The eFG% for different locations w/out the Rockets and Warriors')
plt.legend(loc=(1,0.8))
plt.show()


# After taking out the Golden State Warriors and the Houston Rockets, the trend for the groups is the same. With the exception of slight variances in FG% for some Locations, every group has the same shape, showing that every group utilizes the Locations the same.

# **Indicator 3:** Shot Distance

# In[ ]:


plt.figure(figsize=(16,8))
Chart(3,'Bucket List',3, 0.52)


# The above graph shows the reasoning for the different buckets that I used to help in my analysis. Instead of analyzing shot distance as a continuous measure, I grouped them into buckets: 0-3, 3-23, 23-27, and 27+. The next graphs use these buckets instead.

# In[ ]:


plt.figure(figsize=(14,6))
Chart(3,'SHOTS_ATTEMPTED by SHOT_DISTANCE')
plt.figure(figsize=(14,6))
Chart(3, 'SHOTS_MADE by SHOT_DISTANCE')
plt.figure(figsize=(14,6))
Chart(3, 'FG% by SHOT_DISTANCE')
plt.figure(figsize=(14,6))
Chart(3, 'eFG% by SHOT_DISTANCE')


# The first trend to notice is that the "Good" teams shoot more 3 pointers and make more 3 pointers. The third graph shows that the Field Goal ercentages between the three groups are extremely similar in shape, the only difference being that the percentage for each category goes down as the group gets worse. The trend is very consistent. The Effective Field Goal percentage in the fourth chart shows the same story. The take away from these graphs is that all the teams have almost exactly the same relationship between shot distance and FG%/Shot Attempts, the only difference being that the "Good" teams have a slightly better FG%.

# **Indicator 4:** Period (Quarter) and Minutes

# In[ ]:


plt.figure(figsize=(16,8))
Chart(4, 'SHOTS_MADE by PERIOD')
plt.figure(figsize=(16,8))
Chart(4, 'eFG% by MINUTES_REMAINING')


# The second graph shows two interesting spikes in 4 minutes remaining for the "Good" teams and 3 minutes remaining for the "Bad" teams. Besides this, they all follow a similar trend with nothing of importance showing. These spikes will be explored in the section "Interesting, but Irrelevant Data". The interesting thing about the first graph is that they all follow a very similar negative trend. What separates the good teams is that they come out the strongest in the first period and they have a big jump in FG% from 2nd to 3rd.  The take away from this is that the good teams comes out stronger and maintain it better for the first 3 quarters.

# **Final Analysis**

# Recap of the analysis:
# * Indicator 1 showed no significance between volume/FG% of 3 pointers and winning 
# * Indicator 2 showed that all teams use all areas of the court equally 
# * Indicator 3 showed no significance between the different buckets I chose and volume/FG%
# * Indicator 4 showed that successful teams tend to come out strong in the first quarter and they also tend to limit the the drop in FG% each quarter. However, there was an an inconclusive association between performance based on the minutes left in the quarter and team success.
# 
# These indicators might suggest that you don't have to be like the Rockets and chuck up 3s constantly to be successful, as long as you maintain the average attempts and FG% of 3 Point Field Goals. Surprisingly, there wasn't much difference between how the different shot distances affected the volume/FG% of the shots taken for the different ranks of teams. Coming out strong in the first quarter seemed to be a meaningful factor and maintaining that momentum was a good key to success, which "Good" teams did by "revitalizing" themselves in the 3rd quarter somehow.  
# All in all, my analysis indicates that teams should focus less on attempting more 3s and on coming out as strong as possible in the first quarter, while trying to maintain that strength throughout the quarters. 

# **Interesting, but Irrelevant Data**

# Let's take a look at those weird spikes from Indicator 4 and split up the groups into their actual teams.

# In[ ]:


#Good Teams 
data = pd.read_csv('../input/Indicator4.csv')
data = data[data['TEAM_RATING'] != 'Bad']
data = data[data['TEAM_RATING'] != 'Avg'] 
plt.figure(figsize=(16,8))
ax = sns.barplot(x='TEAM_NAME', y='eFG%', hue='MINUTES_REMAINING',data=data, hue_order=[11,10,9,8,7,6,5,4,3,2,1,0], ci=None)
ax.set(xlabel='Team Name', ylabel='Effective Field Goal Percentage')
ax.set_title('The Effective FG% for every Minute remaining in the quarter for the good teams')
plt.legend(loc=(1,0))
plt.show()
 
#Bad Teams
data = pd.read_csv('../input/Indicator4.csv')
data = data[data['TEAM_RATING'] != 'Good']
data = data[data['TEAM_RATING'] != 'Avg'] 
plt.figure(figsize=(16,8))
ax = sns.barplot(x='TEAM_NAME', y='eFG%', hue='MINUTES_REMAINING',data=data, hue_order=[11,10,9,8,7,6,5,4,3,2,1,0], ci=None)
ax.set(xlabel='Team Name', ylabel='Effective Field Goal Percentage')
ax.set_title('The Effective FG% for every Minute remaining in the quarter for the bad teams')
plt.legend(loc=(1,0))
plt.show()


# These spikes are pretty consistent with most teams. The reason is not known to me. 

# 
