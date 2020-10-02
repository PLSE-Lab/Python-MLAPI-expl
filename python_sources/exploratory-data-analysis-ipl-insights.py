#!/usr/bin/env python
# coding: utf-8

# # IPL Analysis
# ## Introduction
# The Indian Premier League (IPL),is a professional Twenty20 cricket league in India contested during April and May of every year by teams representing Indian cities. The league was founded by the Board of Control for Cricket in India (BCCI) in 2007. 
# The IPL is the most-attended cricket league in the world and ranks sixth among all sports league.
# 
# The data consists of two datasets : matches and deliveries.
# 
# matches dataset contains data of all IPL matches from 2008 season till 2017 season.
# 
# deliveries dataset contains ball by ball data of each IPL match.
# 
# 
# **Objective**
# 
# Aim is to provide some interesting insights by analyzing the IPL data.

# In[1]:


## Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


directory='../input/'


# In[3]:


## function to add data to plot
def annot_plot(ax,w,h):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))


# ### Reading Data 

# In[4]:


match_data=pd.read_csv(directory+'matches.csv')
deliveries_data=pd.read_csv(directory+'deliveries.csv')


# In[5]:


season_data=match_data[['id','season','winner']]

complete_data=deliveries_data.merge(season_data,how='inner',left_on='match_id',right_on='id')


# In[6]:


match_data.head()


# In[7]:


match_data['win_by']=np.where(match_data['win_by_runs']>0,'Bat first','Bowl first')


# In[8]:


match_data.shape


# In[9]:


deliveries_data.head(5)


# In[10]:


deliveries_data['runs']=deliveries_data['total_runs'].cumsum()


# In[11]:


deliveries_data.shape


# ###  Number of Matches played in each IPL season

# In[12]:


ax=sns.countplot('season',data=match_data,palette="Set2")
plt.ylabel('Matches')
annot_plot(ax,0.08,1)


# ### Matches Won By the Teams  
# Mumbai Indians won maximum number of matches followed by Chennai Super Kings.

# In[13]:


match_data.groupby('winner')['winner'].agg(['count']).sort_values('count').reset_index().plot(x='winner',y='count',kind='barh')


# In[14]:


ax=sns.countplot(x='winner',data=match_data)
plt.ylabel('Match')
plt.xticks(rotation=80)
annot_plot(ax,0.05,1)


# ### Win Percentage

# In[15]:


match=match_data.win_by.value_counts()
labels=np.array(match.index)
sizes = match.values
colors = ['gold', 'lightskyblue']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Match Result')
plt.axis('equal')
plt.show()


# In[16]:


sns.countplot('season',hue='win_by',data=match_data,palette="Set1")


# 
# ### Toss Decisions so far 

# In[17]:


toss=match_data.toss_decision.value_counts()
labels=np.array(toss.index)
sizes = toss.values
colors = ['gold', 'lightskyblue']
#explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Toss Result')
plt.axis('equal')
plt.show()


# In[18]:


sns.countplot('season',hue='toss_decision',data=match_data,palette="Set2")


# ### IPL  Winners

# In[21]:


final_matches=match_data.drop_duplicates(subset=['season'], keep='last')

final_matches[['season','winner']].reset_index(drop=True).sort_values('season')


# ### Orange Cap Winners

# In[22]:


Season_orange_cap = complete_data.groupby(["season","batsman"])["batsman_runs"].sum().reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)
Season_orange_cap= Season_orange_cap.drop_duplicates(subset=["season"],keep="first").sort_values(by="season").reset_index(drop=True)
ax=Season_orange_cap.plot(['batsman','season'],'batsman_runs',color='orange',kind='bar')
plt.xticks(rotation=80)
annot_plot(ax,0,10)
Season_orange_cap


# ### Purple Cap Winners 

# In[23]:


Season_purple_cap=complete_data[complete_data["dismissal_kind"]!="run out"]
Season_purple_cap=complete_data.groupby(["season","bowler"])["dismissal_kind"].count().reset_index().sort_values(by="dismissal_kind",ascending=False).reset_index(drop=True)
Season_purple_cap= Season_purple_cap.drop_duplicates(subset=["season"],keep="first").sort_values(by="season").reset_index(drop=True)
Season_purple_cap.columns= ["Season","Bowler","Wicket_taken"]
ax=Season_purple_cap.plot(['Bowler','Season'],'Wicket_taken',color='purple',kind='bar')
plt.xticks(rotation=80)
annot_plot(ax,0,1)
Season_purple_cap


# ## IPL Finals 

# IPL Finals venues and winners along with the number of wins.

# In[24]:


final_matches.groupby(['city','winner']).size()


# ###  Number of IPL seasons won by teams

# In[25]:


final_matches['winner'].value_counts()


# ### Win Percentage in Finals

# In[26]:


match=final_matches.win_by.value_counts()
labels=np.array(match.index)
sizes = match.values
colors = ['gold', 'lightskyblue']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Match Result')
plt.axis('equal')
plt.show()


# ### Toss Decision in Finals 

# In[27]:


toss=final_matches.toss_decision.value_counts()
labels=np.array(toss.index)
sizes = toss.values
colors = ['gold', 'lightskyblue']
#explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Toss Result')
plt.axis('equal')
plt.show()


# In[28]:


final_matches[['toss_winner','toss_decision','winner']].reset_index(drop=True)


# ### Man of the Match in final match 

# In[29]:


final_matches[['winner','player_of_match']].reset_index(drop=True)


# ### It is interesting that out of 10 IPL finals,7 times the team that won the toss was also the winner of IPL

# In[30]:


len(final_matches[final_matches['toss_winner']==final_matches['winner']]['winner'])


# ## Fours  & Sixes
# 
# ### Fours
# Fours hit by teams

# In[31]:


four_data=complete_data[complete_data['batsman_runs']==4]

four_data.groupby('batting_team')['batsman_runs'].agg({'runs by fours':'sum','fours':'count'})


# ### Fours by Players
# Gautam Gambhir hit maximum fours in IPL.

# In[32]:


batsman_four=four_data.groupby('batsman')['batsman_runs'].agg({'four':'count'}).reset_index().sort_values('four',ascending=0)
ax=batsman_four.iloc[:10,:].plot('batsman','four',kind='bar',color='brown')
plt.ylabel('Four')
annot_plot(ax,-0.1,5)


# ### Number of Sixes Hit in Each Season of IPL 

# In[33]:


ax=four_data.groupby('season')['batsman_runs'].agg({'four':'count'}).reset_index().plot('season','four',kind='bar')
plt.ylabel('Four')
annot_plot(ax,-0.1,10)


# ### Sixes 
# Six by Teams

# In[34]:


six_data=complete_data[complete_data['batsman_runs']==6]

six_data.groupby('batting_team')['batsman_runs'].agg({'runs by six':'sum','sixes':'count'})


# ### Six by Players
# Gayle Storm is at the top of this list with 266 sixes.

# In[35]:


batsman_six=six_data.groupby('batsman')['batsman_runs'].agg({'six':'count'}).reset_index().sort_values('six',ascending=0)
ax=batsman_six.iloc[:10,:].plot('batsman','six',kind='bar',color='brown')
plt.ylabel('Six')
annot_plot(ax,-0.1,5)


# ### Number of Sixes Hit in Each Season of IPL 

# In[36]:


ax=six_data.groupby('season')['batsman_runs'].agg({'six':'count'}).reset_index().plot('season','six',kind='bar')
plt.ylabel('Six')
annot_plot(ax,0,10)


# In[37]:


season_six_data=six_data.groupby(['season','batting_team'])['batsman_runs'].agg(['sum','count']).reset_index()


# ### IPL leading Run Scorer 
# Suresh Raina is at the top with 4548 Runs. There are 3 foreign players in this list. Among them Chris Gayle is the leading run scorer. 

# In[38]:


batsman_score=deliveries_data.groupby('batsman')['batsman_runs'].agg(['sum']).reset_index().sort_values('sum',ascending=False).reset_index(drop=True)
batsman_score=batsman_score.rename(columns={'sum':'batsman_runs'})
print("*** Top 10 Leading Run Scorer in IPL ***")
batsman_score.iloc[:10,:]


# In[39]:


No_Matches_player_dismissed = deliveries_data[["match_id","player_dismissed"]]
No_Matches_player_dismissed =No_Matches_player_dismissed .groupby("player_dismissed")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)
No_Matches_player_dismissed.columns=["batsman","No_of Matches"]
No_Matches_player_dismissed .head(5)


# ### Batting Average
# Amla is at the top of this list with the batting average of 44.38.

# In[41]:


Batsman_Average=pd.merge(batsman_score,No_Matches_player_dismissed ,on="batsman")
#merging the score and match played by batsman
Batsman_Average=Batsman_Average[Batsman_Average["batsman_runs"]>=500]
# taking Average for those player for having more than 500 runs under thier belt
Batsman_Average["Average"]=Batsman_Average["batsman_runs"]/Batsman_Average["No_of Matches"]
Batsman_Average['Average']=Batsman_Average['Average'].apply(lambda x: round(x,2))
Batsman_Average=Batsman_Average.sort_values(by="Average",ascending=False).reset_index(drop=True)

top_bat_avg=Batsman_Average.iloc[:10,:]
ax=top_bat_avg.plot('batsman','Average',color='green',kind='bar')
plt.ylabel('Average')
plt.xticks(rotation=80)
annot_plot(ax,0,1)


# ### Dismissals in IPL

# In[42]:


plt.figure(figsize=(12,6))
ax=sns.countplot(deliveries_data.dismissal_kind)
plt.xticks(rotation=90)
annot_plot(ax,0.2,100)


# ### Dismissal by Teams and their distribution 

# In[43]:


out=deliveries_data.groupby(['batting_team','dismissal_kind'])['dismissal_kind'].agg(['count'])

out.groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()),2)).reset_index().sort_values(['batting_team','count'],ascending=[1,0]).set_index(['batting_team','dismissal_kind'])


# In[44]:


wicket_data=deliveries_data.dropna(subset=['dismissal_kind'])


# In[45]:


wicket_data=wicket_data[~wicket_data['dismissal_kind'].isin(['run out','retired hurt','obstructing the field'])]


# ### IPL Most Wicket-Taking Bowlers 
# Malinga is at the top of this list with 154 wickets.

# In[46]:


wicket_data.groupby('bowler')['dismissal_kind'].agg(['count']).reset_index().sort_values('count',ascending=False).reset_index(drop=True).iloc[:10,:]


# ## Powerplays
# In IPL the Powerplay consists of first 6 overs.
# 
# During the first six overs, a maximum of two fielders can be outside the 30-yard circle.

# In[47]:


powerplay_data=complete_data[complete_data['over']<=6]


# ### Runs in Powerplays
# 

# In[48]:


powerplay_data[ powerplay_data['inning']==1].groupby('match_id')['total_runs'].agg(['sum']).reset_index().plot(x='match_id',y='sum',title='Batting First')
powerplay_data[ powerplay_data['inning']==2].groupby('match_id')['total_runs'].agg(['sum']).reset_index().plot(x='match_id',y='sum',title='Batting Second')


# ### Higgest Runs in PowerPlays 

# In[49]:


powerplay_data.groupby(['season','match_id','inning'])['total_runs'].agg(['sum']).reset_index().groupby('season')['sum'].max()


# ### Highest Runs in Powerplay :Batting First 

# In[50]:


pi1=powerplay_data[ powerplay_data['inning']==1].groupby(['season','match_id'])['total_runs'].agg(['sum'])
pi1.reset_index().groupby('season')['sum'].max()


# ### Highest Runs in Powerplay :Batting Second
# 

# In[51]:


pi2=powerplay_data[ powerplay_data['inning']==2].groupby(['season','match_id'])['total_runs'].agg(['sum'])

pi2.reset_index().groupby('season')['sum'].max()


# ### Maximum Wickets Fall in PowerPlay 

# In[52]:


powerplay_data.dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].max()


# ### First Innings 

# In[55]:


powerplay_data[ powerplay_data['inning']==1].dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].max()


# ### Second Innings

# In[56]:


powerplay_data[ powerplay_data['inning']==2].dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].max()


# **Thanks**,  Stay Tuned for more interesting Insights 
