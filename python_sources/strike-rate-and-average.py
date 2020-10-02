#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


matches = pd.read_csv("../input/matches.csv")
matches.head()


# In[ ]:


deliveries = pd.read_csv("../input/deliveries.csv")
deliveries.head()


# We will look at how the batsmen have performed in all the seasons of IPL and try to come up with an in depth analysis of how things have been and how they have changed over the years.

# # Frequency of Scores

# We will be looking at the general distribution of scores among the batsmen.

# In[ ]:


batsmen_score = pd.DataFrame(deliveries.groupby(['match_id','batsman']).sum()['batsman_runs'])
batsmen_score.head()


# In[ ]:


plt.rcParams['figure.figsize'] = 10,5
batsmen_score.plot(kind = 'hist',fontsize = 20)
plt.xlabel('Runs Scored',fontsize = 20)
plt.ylabel('Number of Times',fontsize = 20)
plt.title('Histogram for Runs Scored',fontsize = 20)
plt.show()


# This includes time when batsmen might not have played much balls coming towards the end of the innings . Let see for those batsmen who have played atleast 20+ balls

# In[ ]:


batsmen_score_20 = pd.DataFrame(deliveries.groupby(['match_id','batsman']).agg({'batsman_runs' : 'sum', 'ball' :'count'}))
batsmen_score_20[batsmen_score_20['ball'] > 20].plot(kind = 'scatter', x = 'ball',y = 'batsman_runs')
plt.xlabel('Ball Faced',fontsize = 20)
plt.ylabel('Runs Scored',fontsize = 20)
plt.title('Runs Scored vs Balls Faced',fontsize = 20)
plt.show()


# This gives us aan indication that runs scored increases with increase in the ball faced.

# Now lets look at how the batsmen have scored runs with respect to their strike rate when they have faced more than 15 balls. Strike rate is defined as the number of runs scored divided by the number of balls faced.

# # Strike Rate

# In[ ]:


batsmen_strikerate = batsmen_score_20[batsmen_score_20['ball'] >= 15]
batsmen_strikerate['Strike Rate'] = batsmen_strikerate['batsman_runs']/batsmen_strikerate['ball']*100
batsmen_strikerate.head()


# In[ ]:


ax = batsmen_strikerate.plot(kind = 'scatter',x = 'batsman_runs', y = 'Strike Rate')
plt.xlabel('Runs Scored',fontsize = 20)
plt.ylabel('Strike Rate',fontsize = 20)
plt.title('Innings Progression with Runs',fontsize = 20)
plt.show()


# This appears to be a strange plot, something that i did not expect. It tells us that with increase in the number of runs the strike rate tends to improve. This can be linked to the fact that with increase in the number of balls faced, the scoring increases at a faster rate and hence their ratio tends to increase and so we get such a graph. We all see their are lots of different strike rate particular to a specific score, this can be because of the time the batsmen has come out to play. In the later stage of the innings, they generally come out all guns blazing, their by scoring a quick fire 25~30 runs with a higher strike rate then normal. But we can see that generally when the score is high the lowest limit of the strike rate is increasing and the maximum limit also tend to increase. Lets look at a graph of balls faced vs strike rate.

# In[ ]:


ax = batsmen_strikerate.plot(kind = 'scatter', x ='ball',y = 'Strike Rate',color = 'y')
batsmen_strikerate.groupby(['ball']).max().plot(kind = 'line',y = 'Strike Rate',ax = ax,color = 'green',label = 'Max Strike Rate')
batsmen_strikerate.groupby(['ball']).min().plot(kind = 'line',y = 'Strike Rate',ax = ax,color = 'red',label = 'Min Strike Rate')
plt.xlabel('Ball Faced',fontsize = 20)
plt.ylabel('Strike Rate',fontsize = 20)
plt.title('Strike Rate Progression with Balls',fontsize = 20)
plt.show()


# This line graphs gives us a kind of bucket in which we can expect the strikerate to lie when we know the number of balls faced and we can see that we increase in the number of ball faced , the bucket tends to become shorter and shorter.

# Since we have looked on the strike rates so much, lets have a look on how the general distribution of strike rate is.

# In[ ]:


batsmen_strikerate.boxplot(column = ['Strike Rate'])


# Lets look at the players who have the best Strike Rate in a game when they faced more than 30 balls

# In[ ]:


batsmen_strikerate[batsmen_strikerate['ball'] > 30].sort_values(by = 'Strike Rate',ascending = False).head()
#Balls Faced greated than 30


# As we can see most players have balls between 30 and 40. As we know strike rate tends to change with balls faced. Let us increase the balls and see what happens.

# In[ ]:


batsmen_strikerate[batsmen_strikerate['ball'] > 40].sort_values(by = 'Strike Rate',ascending = False).head()
#Balls Faced greated than 40


# As expected it has changed and we can also see the Strike Rate decreasing as we have increase the number of balls.Lets increase it further.

# In[ ]:


batsmen_strikerate[batsmen_strikerate['ball'] > 50].sort_values(by = 'Strike Rate',ascending = False).head()
#Balls Faced greated than 50


# In[ ]:


batsmen_strikerate[batsmen_strikerate['ball'] > 60].sort_values(by = 'Strike Rate',ascending = False).head()
#Balls Faced greated than 60


# In[ ]:


batsmen_strikerate[batsmen_strikerate['ball'] > 70].sort_values(by = 'Strike Rate',ascending = False).head()
#Balls Faced greated than 70


# Now we can see how the strike rate has changed with change in number of balls and also the players name have changed. Infact is not strange to see that the players who were there is the first list are not there in the last lists.

# Lets look the the strike rate of all the players overall in all the seasons.

# In[ ]:


aggregatedata = pd.merge(matches,deliveries, left_on = 'id',right_on = 'match_id')
aggregatedata.columns


# In[ ]:


batsmen_strikerate_season = pd.DataFrame(deliveries.groupby(['batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))
batsmen_strikerate_season['Strike Rate'] = batsmen_strikerate_season['batsman_runs']/batsmen_strikerate_season['ball']*100
batsmen_strikerate_season = batsmen_strikerate_season.sort_values(by ='Strike Rate' , ascending = False)
batsmen_strikerate_season[batsmen_strikerate_season['batsman_runs'] > 2500] 
# We have taken runs greater then 2500 So that we take a significant amount of runs


# In[ ]:


colors = cm.rainbow(np.linspace(0,1,len(batsmen_strikerate_season[batsmen_strikerate_season['batsman_runs'] > 2500])))
batsmen_strikerate_season[batsmen_strikerate_season['batsman_runs'] > 2500].plot(kind = 'bar',y = 'Strike Rate',
                                                                                color = colors,legend = '',fontsize = 10)
plt.xlabel('Batsman Name',fontsize = 20)
plt.ylabel('Strike Rate',fontsize = 20)
plt.show()


# We see that players who have scored a significant amount of runs tends to have almost similar strike rates and it does not vary much for batsmen who have played for a long time and have scored many runs. Lets look at the chart of who tops this list every season.

# In[ ]:


batsmen_strikerate_eseason = pd.DataFrame(aggregatedata.groupby(['season','batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))
batsmen_strikerate_eseason['Strike Rate'] = batsmen_strikerate_eseason['batsman_runs']/batsmen_strikerate_eseason['ball']*100
batsmen_strikerate_eseason = batsmen_strikerate_eseason.sort_values(by =['season','Strike Rate'] , ascending = False)
batsmen_strikerate_eseason.reset_index(inplace = True)
batsmen_strikerate_eseason[batsmen_strikerate_eseason['batsman_runs'] > 300].head()

# We have taken runs greater then 300 So that we take a significant amount of runs


# In[ ]:


colors = cm.rainbow(np.linspace(0,1,10))
plt.rcParams['figure.figsize'] = 10,5
for title,group in batsmen_strikerate_eseason.groupby('season'):
    group[group['batsman_runs'] > 300].head(10).plot(x = 'batsman',y = 'Strike Rate',kind = 'bar',legend = '',
                                                     color = colors,fontsize = 10)
    plt.xlabel('Batsman Name',fontsize = 20)
    plt.ylabel('Strike Rate',fontsize = 20)
    plt.title('Top 10 Strike Rate in Season %s '%title,fontsize = 20)
plt.show()


# Now lets look at best Strike rate during each innings for players who have scored more than 20 runs in the match

# In[ ]:


strikerate_inning = pd.DataFrame(deliveries.groupby(['match_id','inning','batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))
strikerate_inning['Strike Rate'] = strikerate_inning['batsman_runs']/strikerate_inning['ball']*100
strikerate_inning.reset_index(inplace = True)
strikerate_inning = strikerate_inning[strikerate_inning['inning'] <= 2]
strikerate_inning.reset_index(inplace = True)
plt.rcParams['figure.figsize'] = 20,15
sns.scatterplot(x = 'index',y = 'Strike Rate',hue = 'inning',data = strikerate_inning[strikerate_inning['batsman_runs'] > 20],hue_norm=(1, 2))
#strikerate_inning[strikerate_inning['batsman_runs'] > 20].plot(kind = 'scatter',x = 'index',y = 'Strike Rate')
plt.xlabel('Index',fontsize = 20)
plt.ylabel('Strike Rate',fontsize = 20)
plt.title('Strike rate in each Inning',fontsize = 20)
plt.show()
#Provide color using innings seaborn
strikerate_inning['inning'].unique()


# This graph tells us about thow strike rate vary across innings for all the players. Now lets look at player with best strike rates in each inning . The top 10 players for both over all the seasons.

# In[ ]:


player_strikerate_inning = pd.DataFrame(deliveries.groupby(['inning','batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))
player_strikerate_inning['Strike Rate'] = player_strikerate_inning['batsman_runs']/player_strikerate_inning['ball']*100
player_strikerate_inning.reset_index(inplace = True)
player_strikerate_inning = player_strikerate_inning[player_strikerate_inning['inning'] <= 2]
player_strikerate_inning = player_strikerate_inning.sort_values(by = ['Strike Rate'],ascending = False)


# In[ ]:


# Taking only players who have scored more than 1500 runs in either innings
plt.rcParams['figure.figsize'] = 35,35
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
player_strikerate_inning_one = player_strikerate_inning[player_strikerate_inning['inning'] == 1]
colors = cm.rainbow(np.linspace(0,1,len(player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500])))
player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500].plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax1)
ax1.set_ylabel('Strike Rate',fontsize = 20)
ax1.set_xlabel('')
ax1.set_title('Strike Rate 1st innings',fontsize = 20)

player_strikerate_inning_one = player_strikerate_inning[player_strikerate_inning['inning'] == 2]
colors = cm.rainbow(np.linspace(0,1,len(player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500])))
player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500].plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax2)
ax2.set_ylabel('Strike Rate',fontsize = 20)
ax2.set_title('Strike Rate 2nd innings',fontsize = 20)
plt.xlabel('Batsman Name',fontsize = 20)
plt.show()


# There are alot of drill downs that can be done for strike rate like best strike rate in the initial overs, middle overs and death overs. Each of this has its significance on the impact of the game and are worth considering for. I will look at all this graphs in some other section. Right now lets look the top strike rates for players against the result of the match.

# In[ ]:


strikerate_result = pd.DataFrame(aggregatedata.groupby(['match_id','winner','batsman','batting_team']).agg({'batsman_runs' : 'sum','ball' : 'count'}))
strikerate_result.reset_index(inplace = True)
def win(x):
    if x['winner'] == x['batting_team'] :
        return 'Yes'
    else:
        return 'No'
strikerate_result['Win'] = strikerate_result.apply(win, axis=1)

strikerate_result['Strike Rate'] = strikerate_result['batsman_runs']/strikerate_result['ball']*100
strikerate_result.reset_index(inplace = True)
strikerate_result.reset_index(inplace = True)
plt.rcParams['figure.figsize'] = 20,15
sns.scatterplot(x = 'index',y = 'Strike Rate',hue = 'Win',data = strikerate_result[strikerate_result['batsman_runs'] > 20])
#strikerate_result[strikerate_result['batsman_runs'] > 20].plot(kind = 'scatter',x = 'index',y = 'Strike Rate')
plt.xlabel('Index',fontsize = 20)
plt.ylabel('Strike Rate',fontsize = 20)
plt.title('Strike rate and match result',fontsize = 20)
plt.show()
#Provide color using Win seaborn


# In[ ]:


# Taking only players who have scored more than 1500 runs in either innings
strikerate_result_win = pd.DataFrame(aggregatedata.groupby(['winner','batsman','batting_team']).agg({'batsman_runs' : 'sum','ball' : 'count'}))
strikerate_result_win.reset_index(inplace = True)
def win(x):
    if x['winner'] == x['batting_team'] :
        return 'Yes'
    else:
        return 'No'
strikerate_result_win['Win'] = strikerate_result_win.apply(win, axis=1)

strikerate_result_win.reset_index(inplace = True)
strikerate_result_win.head()
strikerate_result_win = pd.DataFrame(strikerate_result_win.groupby(['batsman','Win']).agg({'batsman_runs' : 'sum','ball' : 'sum'}))
strikerate_result_win['Strike Rate'] = strikerate_result_win['batsman_runs']/strikerate_result_win['ball']*100
strikerate_result_win.reset_index(inplace = True)
strikerate_result_win = strikerate_result_win.sort_values(by = ['Strike Rate'],ascending = False)
strikerate_result_win.head()


# In[ ]:


plt.rcParams['figure.figsize'] = 25,25

f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
strikerate_result_winn = strikerate_result_win[strikerate_result_win['Win'] == 'Yes']
colors = cm.rainbow(np.linspace(0,1,10))
strikerate_result_winn[strikerate_result_winn['batsman_runs'] > 1500].head(10).plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax1)
ax1.set_ylabel('Strike Rate',fontsize = 20)
ax1.set_xlabel('')
ax1.set_title('Strike Rate during Wins',fontsize = 20)

strikerate_result_lose = strikerate_result_win[strikerate_result_win['Win'] == 'No']
strikerate_result_lose[strikerate_result_lose['batsman_runs'] > 1500].head(10).plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax2)
ax2.set_ylabel('Strike Rate',fontsize = 20)
ax2.set_title('Strike Rate during Losses',fontsize = 20)
plt.xlabel('Batsman Name',fontsize = 20)
plt.show()


# This tells us about all the players and how their strike rate has been when they were winning or when they were losing.

# # Average 

# This is the average number of runs scored by the batsman in each match.

# In[ ]:


batsmen_average = pd.DataFrame(deliveries.groupby(['batsman']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))
batsmen_average['Average'] = batsmen_average['batsman_runs']/batsmen_average['player_dismissed']
batsmen_average = batsmen_average.sort_values(by = 'Average',ascending = False)
batsmen_average[batsmen_average['batsman_runs'] > 2500]


# In[ ]:


plt.rcParams['figure.figsize'] = 15,10
colors = cm.rainbow(np.linspace(0,1,len(batsmen_average[batsmen_average['batsman_runs'] > 2500])))
batsmen_average[batsmen_average['batsman_runs'] > 2500].plot(kind = 'bar',y = 'Average',
                                                                                color = colors,legend = '',fontsize = 10)
plt.xlabel('Batsman Name',fontsize = 20)
plt.ylabel('Average',fontsize = 20)
plt.show()


# We can see that average tend have a higher decrease from the first to the end then what we saw for strike rate.Now a interesting thing to see will be the relation between Average and strike rate for a player.

# In[ ]:


batsmen_averagesr = pd.merge(batsmen_strikerate_season,batsmen_average,left_on = 'batsman',right_on = 'batsman')
batsmen_averagesr.reset_index(inplace = True)
batsmen_averagesr['Category'] = batsmen_averagesr['batsman_runs_x'].apply(lambda x: 1 if x <= 250 
                                                                         else( 2 if x<=500 
                                                                         else( 3 if x<=1000  
                                                                         else( 4 if x<=1500
                                                                         else( 5 if x<=2000
                                                                         else( 6 if x<=2500 
                                                                         else 7))))))
batsmen_averagesr['Category'].unique()


# In[ ]:


fig, ax = plt.subplots()
categories = np.unique(batsmen_averagesr['Category'])
colors = np.linspace(0, 1, len(categories))
colordict = dict(zip(categories, colors))  

batsmen_averagesr["Color"] = batsmen_averagesr['Category'].apply(lambda x: colordict[x])
#ax.scatter(batsmen_averagesr['Strike Rate'],batsmen_averagesr['Average'],c =batsmen_averagesr['Color'])
sns.scatterplot(x = 'Strike Rate',y = 'Average',hue = 'Category',data = batsmen_averagesr)
plt.xlabel('Strike Rate')
plt.ylabel('Average')

plt.show()


# There are few players shows by the dark plot that has both high average and high strike rate. Most of them tend to have high strike rate and low average.

# Let us now look at the best averages every season for player who have score more than 250 runs

# In[ ]:


batsmen_average_season = pd.DataFrame(aggregatedata.groupby(['season','batsman']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))
batsmen_average_season['Average'] = batsmen_average_season['batsman_runs']/batsmen_average_season['player_dismissed']
batsmen_average_season = batsmen_average_season.sort_values(by = ['season','Average'],ascending = False)
batsmen_average_season.reset_index(inplace = True)
batsmen_average_season[batsmen_average_season['batsman_runs'] > 300].head()


# In[ ]:


colors = cm.rainbow(np.linspace(0,1,10))
plt.rcParams['figure.figsize'] = 10,5
for title,group in batsmen_average_season.groupby('season'):
    group[group['batsman_runs'] > 300].head(10).plot(x = 'batsman',y = 'Average',kind = 'bar',legend = '',
                                                     color = colors,fontsize = 10)
    plt.xlabel('Batsman Name',fontsize = 20)
    plt.ylabel('Average',fontsize = 20)
    plt.title('Top 10 Average in Season %s '%title,fontsize = 20)
plt.show()


# Now lets look at the average for players in each of the innings and which are the top players to look out for.

# In[ ]:


average_inning = pd.DataFrame(deliveries.groupby(['inning','batsman']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))
average_inning['Average'] = average_inning['batsman_runs']/average_inning['player_dismissed']
average_inning.reset_index(inplace = True)
average_inning = average_inning[average_inning['inning'] <= 2]
average_inning.reset_index(inplace = True)
plt.rcParams['figure.figsize'] = 15,5
sns.scatterplot(x = 'index',y = 'Average',hue = 'inning',data = average_inning[average_inning['batsman_runs'] > 500],hue_norm=(1, 2))
#average_inning[average_inning['batsman_runs'] > 500].plot(kind = 'scatter',x = 'index',y = 'Average')
#Taking more than 500 so that we make usre we have included only the batsmen
plt.xlabel('Index',fontsize = 20)
plt.ylabel('Average',fontsize = 20)
plt.title('Average in each Inning',fontsize = 20)

plt.show()
#Provide color using innings seaborn


# In[ ]:


average_inning = average_inning.sort_values(by = ['Average'],ascending = False)

plt.rcParams['figure.figsize'] = 35,35

f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
average_inning_one = average_inning[average_inning['inning'] == 1]
average_inning_one = average_inning_one[average_inning_one['player_dismissed'] != 0 ]
colors = cm.rainbow(np.linspace(0,1,15))
average_inning_one[average_inning_one['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax1)
ax1.set_ylabel('Average',fontsize = 20)
ax1.set_xlabel('')
ax1.set_title('Average 1st innings',fontsize = 20)

average_inning_two = average_inning[average_inning['inning'] == 2]
average_inning_two = average_inning_two[average_inning_two['player_dismissed'] != 0 ]
average_inning_two[average_inning_two['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax2)
ax2.set_ylabel('Average',fontsize = 20)
ax2.set_title('Average 2nd innings',fontsize = 20)
plt.xlabel('Batsman Name',fontsize = 20)
plt.show()


# As we can see AB de Villiers who has the maximum average in the first innings but is 10th in terms of average in the second innings tells us that he is not a good chaser or that he does not perform well in the second innings. David Warner who is not even there on the list in the first graph suddenly tops the second one showcasing his chasing abilities.  

# Now hows the player's performance is when their team win or lose.

# In[ ]:


average_result = pd.DataFrame(aggregatedata.groupby(['winner','batsman','batting_team']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))
average_result.reset_index(inplace = True)
def win(x):
    if x['winner'] == x['batting_team'] :
        return 'Yes'
    else:
        return 'No'
average_result['Win'] = average_result.apply(win, axis=1)

average_result = pd.DataFrame(average_result.groupby(['batsman','Win']).agg({'batsman_runs' : 'sum', 'player_dismissed' : 'sum' }))

average_result['Average'] = average_result['batsman_runs']/average_result['player_dismissed']
average_result.reset_index(inplace = True)
average_result.reset_index(inplace = True)
average_result = average_result[average_result['player_dismissed'] != 0]
plt.rcParams['figure.figsize'] = 15,5

sns.scatterplot(x = 'index',y = 'Average',hue = 'Win',data = average_result[average_result['batsman_runs'] > 500],hue_norm=(1, 2))
#average_result[average_result['batsman_runs'] > 500].plot(kind = 'scatter',x = 'index',y = 'Average')
plt.xlabel('Index',fontsize = 20)
plt.ylabel('Average',fontsize = 20)
plt.title('Strike rate and match result',fontsize = 20)
plt.show()


# This tells us that players( win runs > 500) tend to have a high average when their team wins, showing that they usually perform well and contribute to their team's win and when the same player do not perform well or score less their team loses . As shown by the low average for the orange points

# In[ ]:


average_result = average_result.sort_values(by = ['Average'],ascending = False)

plt.rcParams['figure.figsize'] = 35,35

f, (ax1, ax2) = plt.subplots(2, 1,sharey = True)
average_result_win = average_result[average_result['Win'] == 'Yes']
colors = cm.rainbow(np.linspace(0,1,15))
average_result_win[average_result_win['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax1)
ax1.set_ylabel('Average',fontsize = 20)
ax1.set_xlabel('')
ax1.set_title('Average during wins ',fontsize = 20)

average_result_lose = average_result[average_result['Win'] == 'No']
average_result_lose[average_result_lose['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax2)
ax2.set_ylabel('Average',fontsize = 20)
ax2.set_title('Average during losses',fontsize = 20)
plt.xlabel('Batsman Name',fontsize = 20)
plt.show()


# The first graph tells us how important some players are for their team. As you can see the top four players have an average of 50+ when their team wins, showing that when they perform their team generally wins. That is they make sure they take their team over the line. The second graph shows us the details about the players who efforts goes in vain. This are the players who perform well even when the rest of the teams are not performing well.

# In[ ]:




