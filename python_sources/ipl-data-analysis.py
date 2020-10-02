#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading Important Libraries
import pandas as pd # To work with data
import numpy as np # Stats and Fast array computations
import seaborn as sns # fot Visualization
import matplotlib.pyplot as plt # For Visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


deliveries = pd.read_csv("/kaggle/input/ipl/deliveries.csv") # Loading .csv File


# In[ ]:


deliveries.columns


# In[ ]:


deliveries.isnull().sum()


# In[ ]:


deliveries.shape


# ### three columns have high null value counts. Before dropping them, let's see if we can get anything.

# In[ ]:


deliveries.player_dismissed.value_counts().head()


# In[ ]:


fig,ax = plt.subplots(1,figsize=(8,5))
sns.barplot(deliveries.player_dismissed.value_counts().head(10),deliveries.player_dismissed.value_counts().head(10).index,ax = ax)


# ### Now we're gonna drop coluns containing high number of null values.

# In[ ]:


deliveries.drop(columns=['player_dismissed','dismissal_kind',"fielder"],inplace=True)


# In[ ]:


deliveries.isnull().sum()


# ### Now that we have no null values, let's go ahead with it.

# #### The first question that i have in mind, 'Which team has scored highest runs through ipl?" Let's see...

# In[ ]:


fig,ax = plt.subplots(1,figsize=(8,6))
sns.barplot(deliveries.groupby(by='batting_team').total_runs.sum().sort_values(ascending=False),deliveries.groupby(by='batting_team').total_runs.sum().sort_values(ascending=False).index,ax=ax)


# #### One more interesting observation could be, if we can get any pattern of how teams play thoughout the overs.

# In[ ]:


fig,ax = plt.subplots(1,figsize=(8,6))
sns.barplot(deliveries.groupby(by='over').total_runs.sum().sort_values(ascending=False).index,deliveries.groupby(by='over').total_runs.sum().sort_values(ascending=False),ax=ax)


# #### This might not be very trustworthy graph as it is combining all the teams. And it is not sure that every team follows the same pattern or not. O, let's see this for every team.

# In[ ]:


df1 = pd.pivot_table(deliveries,values='total_runs',index='over',columns='batting_team',aggfunc='sum')

fig,ax = plt.subplots(14,figsize=(10,50))
for i in range(14):
    df1.iloc[:,i].plot(ax=ax[i],kind='bar',legend=True)
del df1


# ### Yes, our graph was right. Every team seems to follow the same pattern where they tend to slow down there pace on the 7th over and then again pick it up from there.

# ### One more interesting thought in my mind right now is about MS DHoni and Virat kohli. These are two superstar players of indin cricket team. Let's see and compare their batting pattern.

# In[ ]:


df2 = pd.pivot_table(deliveries,values='batsman_runs',index='over',columns='batsman',aggfunc='sum').fillna(0)

fig,ax = plt.subplots(2,figsize=(10,10))
df2.loc[:,'MS Dhoni'].plot(kind='bar',legend=True,ax=ax[0])
df2.loc[:,'V Kohli'].plot(kind='bar',legend=True,ax=ax[1])

del df2


# #### So, sir M.S. Dhoni is indeed a great finisher. And as it seems, Virat kohli doesn't play very well in starting or ending overs.

# #### Let's bring in our second dataset.

# In[ ]:


matches = pd.read_csv("/kaggle/input/ipl/matches.csv")


# In[ ]:


matches.columns


# In[ ]:


matches.isnull().sum()


# In[ ]:


matches.shape


# #### Umpire3 column has all the null values. better drop this one.

# In[ ]:


matches.drop('umpire3',axis=1,inplace=True)
matches.dropna(inplace=True)


# In[ ]:


len(matches.city.unique())


# #### How many matches have we played per season ?

# In[ ]:


fig,ax = plt.subplots(1,figsize = (10,6))
sns.countplot(matches.season, ax = ax)


# #### How many individual matches the teams have won..?

# In[ ]:


fig,ax = plt.subplots(1,figsize = (6,4))
sns.countplot(matches.winner,ax=ax)
plt.xticks(rotation=90)


# ### how many toss winnings per team..?

# In[ ]:


fig,ax = plt.subplots(1,figsize = (6,4))
sns.countplot(matches.toss_winner,ax=ax)
plt.xticks(rotation=90)


# #### So, is winng a toss mean winng the match ?

# In[ ]:


matches['is_toss_equal_match'] = matches.toss_winner==matches.winner

matches.is_toss_equal_match = matches.is_toss_equal_match.map({True:'Yes',False:'No'})

sns.countplot(matches.is_toss_equal_match)

matches.drop('is_toss_equal_match',axis=1,inplace=True)


# #### Well, there is a slight correlation but it can't be considered so significant.

# ### Let's see which city has hosted most ipl matches.

# In[ ]:


sns.barplot(matches.city.value_counts().head(10).index,matches.city.value_counts().head(10).values)
plt.xticks(rotation=90)


# ### Stadiums with highest number of matches played.

# In[ ]:


sns.barplot(matches.venue.value_counts().head(10).index,matches.venue.value_counts().head(10).values)
plt.xticks(rotation=90)


# #### One interesting thing could be to get the batsman whoo has scored maximum overall runs in ipl.

# In[ ]:


batsman_score = deliveries.groupby(by='batsman').total_runs.sum().sort_values(ascending=False)

sns.barplot(batsman_score.head(10).index,batsman_score.head(10).values)
plt.xticks(rotation=90)

del batsman_score


# #### Many of us are also intertested in seeing batsman with most runs each season. So, let's do that.

# In[ ]:


new_df = deliveries.merge(matches.set_index('id').season,how ='left', left_on='match_id', right_on ='id')

new_df = new_df.groupby(by=['season','batsman']).total_runs.sum()

seasons_list = list(new_df.index.get_level_values(0).unique().astype(int))

season_wise_most_score = {2008:0, 2009:0, 2010:0, 2011:0, 2012:0, 2013:0, 2014:0, 2015:0, 2016:0, 2017:0}
season_wise_most_scorer = {2008:'', 2009:'', 2010:'', 2011:'', 2012:'', 2013:'', 2014:'', 2015:'', 2016:'', 2017:''}

for i in seasons_list:
    season_wise_most_score[i] = new_df.loc[i].sort_values(ascending=False)[0]
    season_wise_most_scorer[i] = new_df.loc[i].sort_values(ascending=False).index[0]

season_wise_most_score = pd.Series(season_wise_most_score)
season_wise_most_scorer = pd.Series(season_wise_most_scorer)

del seasons_list,new_df

season_most_scorer = pd.DataFrame(dict(batsman = season_wise_most_scorer, score = season_wise_most_score))

del season_wise_most_scorer,season_wise_most_score


# In[ ]:


season_most_scorer


# In[ ]:


sns.barplot(season_most_scorer.index , season_most_scorer.score ,hue = season_most_scorer.batsman)


# ### It seems that most of our analysis so far is based on batsmen performances. Let's go beyond this and see what we find.
# #### If We think about bowlers, let's see which bowler has highest number of wide bowls.

# In[ ]:


wide_bowl = deliveries.groupby('bowler').wide_runs.sum().sort_values(ascending=False).head(10)

sns.barplot(wide_bowl.index,wide_bowl.values)
plt.xticks(rotation=90)
del wide_bowl


# #### Let's do the same for no bowls.

# In[ ]:


no_bowl = deliveries.groupby('bowler').noball_runs.sum().sort_values(ascending=False).head(10)

sns.barplot(no_bowl.index,no_bowl.values)
plt.xticks(rotation=90)
del no_bowl


# #### Let's see which bowler has given most overall runs to the opposition team.

# In[ ]:


runs_by_bowler = deliveries.groupby('bowler').total_runs.sum().sort_values(ascending=False).head(10)

sns.barplot(runs_by_bowler.index,runs_by_bowler.values)
plt.xticks(rotation=90)
del runs_by_bowler


# #### We can get most runs giving bowler per season as ell, just as we did with batsmans. Let's do that.

# In[ ]:


new_df = deliveries.merge(matches.set_index('id').season,how ='left', left_on='match_id', right_on ='id')

new_df = new_df.groupby(by=['season','bowler']).total_runs.sum()

seasons_list = list(new_df.index.get_level_values(0).unique().astype(int))

season_wise_most_score = {2008:0, 2009:0, 2010:0, 2011:0, 2012:0, 2013:0, 2014:0, 2015:0, 2016:0, 2017:0}
season_wise_most_scorer = {2008:'', 2009:'', 2010:'', 2011:'', 2012:'', 2013:'', 2014:'', 2015:'', 2016:'', 2017:''}

for i in seasons_list:
    season_wise_most_score[i] = new_df.loc[i].sort_values(ascending=False)[0]
    season_wise_most_scorer[i] = new_df.loc[i].sort_values(ascending=False).index[0]

season_wise_most_score = pd.Series(season_wise_most_score)
season_wise_most_scorer = pd.Series(season_wise_most_scorer)

del seasons_list,new_df

season_most_scorer = pd.DataFrame(dict(bowler = season_wise_most_scorer, score = season_wise_most_score))

del season_wise_most_scorer,season_wise_most_score


# In[ ]:


season_most_scorer


# In[ ]:


sns.barplot(season_most_scorer.index , season_most_scorer.score ,hue = season_most_scorer.bowler)


# #### Let's see teams who've lukily got most extra runs.

# In[ ]:


extra_runs_by_teams = deliveries.groupby(by='bowling_team').extra_runs.sum().sort_values(ascending=False).head(10)

sns.barplot(extra_runs_by_teams.index,extra_runs_by_teams.values)
plt.xticks(rotation=90)
del extra_runs_by_teams


# In[ ]:


new_df = deliveries.merge(matches.set_index('id').season,how ='left', left_on='match_id', right_on ='id')


# In[ ]:


new_df.columns


# ### Now that we have a new dataframe with seasons and deliveries combined. Let's see how our superstars have performed throughout the seasons.

# In[ ]:


dhoni = new_df[new_df['batsman'] == 'MS Dhoni'].groupby(by = 'season').batsman_runs.sum()
kohli = new_df[new_df['batsman'] == 'V Kohli'].groupby(by = 'season').batsman_runs.sum()

dhoni_kohli_season = pd.DataFrame(dhoni.values,index=dhoni.index,columns=['Dhoni'])
dhoni_kohli_season['Kohli'] = kohli

del dhoni,kohli

dhoni_kohli_season.plot(kind='bar')


# #### Let's see which team has most bounderies.

# In[ ]:


temp = new_df[(new_df.batsman_runs==4 ) | (new_df.batsman_runs==6)].groupby(by='batting_team').total_runs.count().sort_values(ascending=False)

sns.barplot(temp.index,temp.values)
plt.xticks(rotation=90)


# #### Well, when we think about bounderies we think more about the batsman then the team. So, let's do that.

# In[ ]:


temp = new_df[(new_df.batsman_runs==4 ) | (new_df.batsman_runs==6)].groupby(by='batsman').batsman_runs.count().sort_values(ascending=False).head(10)

sns.barplot(temp.index,temp.values)
plt.xticks(rotation=90)


# #### Agagin, let's see which teams have highest bouneries against them.

# In[ ]:


temp = new_df[(new_df.batsman_runs==4 ) | (new_df.batsman_runs==6)].groupby(by='bowling_team').total_runs.count().sort_values(ascending=False).head(10)

sns.barplot(temp.index,temp.values)
plt.xticks(rotation=90)


# ### One more interesting insight could be to get details about how many time a team has got into finals and how many times it has won it.

# In[ ]:


temp = []
for season,data in new_df.groupby(by='season'):
    temp.append((season,data.match_id.unique()[-1]))


# In[ ]:


matches.team1.unique()


# In[ ]:


qualify = {'Sunrisers Hyderabad':0, 'Mumbai Indians':0, 'Gujarat Lions':0,
           'Royal Challengers Bangalore':0,
       'Kolkata Knight Riders':0, 'Delhi Daredevils':0, 'Kings XI Punjab':0,
       'Chennai Super Kings':0, 'Rajasthan Royals':0, 'Deccan Chargers':0,
       'Kochi Tuskers Kerala':0, 'Pune Warriors':0, 'Rising Pune Supergiants':0,'Rising Pune Supergiant':0}
wins = {'Sunrisers Hyderabad':0, 'Mumbai Indians':0, 'Gujarat Lions':0,
       'Royal Challengers Bangalore':0,
       'Kolkata Knight Riders':0, 'Delhi Daredevils':0, 'Kings XI Punjab':0,
       'Chennai Super Kings':0, 'Rajasthan Royals':0, 'Deccan Chargers':0,
       'Kochi Tuskers Kerala':0, 'Pune Warriors':0, 'Rising Pune Supergiants':0,'Rising Pune Supergiant':0}


# In[ ]:


for tup in temp:
    season = tup[0]
    match = tup[1]
    qualify[matches[(matches.season==season) & (matches.id == match)].team1.values[0]]+=1
    qualify[matches[(matches.season==season) & (matches.id == match)].team2.values[0]]+=1
    wins[matches[(matches.season==season) & (matches.id == match)].winner.values[0]]+=1


# In[ ]:


plt.subplot(1,2,1)
sns.barplot(list(qualify.keys()),list(qualify.values()))
plt.xticks(rotation=90)
plt.title('Qualifiers')
plt.subplot(1,2,2)
sns.barplot(list(wins.keys()),list(wins.values()))
plt.xticks(rotation=90)
plt.title('Winners')


# #### As it looks, some of them could never qualify for the finals. And some of them have never won it.

# ##### Let's see teams who have nver qualified and have never won it.

# In[ ]:


qualify = pd.Series(qualify)
qualify[qualify==0]


# In[ ]:


wins = pd.Series(wins)

wins[wins==0]


# In[ ]:


new_df = deliveries.merge(matches[['id','umpire1','umpire2']],how='left',left_on='match_id',right_on='id')

temp = new_df.groupby(by='umpire1').noball_runs.sum().sort_values(ascending=False).head(10)
sns.barplot(temp.index,temp.values)
plt.xticks(rotation=90)
plt.title('Most No Balls by umpire1')


# In[ ]:


new_df = deliveries.merge(matches[['id','umpire1','umpire2']],how='left',left_on='match_id',right_on='id')

temp = new_df.groupby(by='umpire2').noball_runs.sum().sort_values(ascending=False).head(10)
sns.barplot(temp.index,temp.values)
plt.xticks(rotation=90)
plt.title('Most No Balls by umpire2')


# #### Most man of the match title winner players.

# In[ ]:


man_of_the_match = matches.player_of_match.value_counts().head(10)

sns.barplot(man_of_the_match.index,man_of_the_match.values)
plt.xticks(rotation=90)


# #### One more interesting question is to know wether perticuler city favors a perticuler team? Or, do teams perform better on their home grounds?

# In[ ]:


new_df = deliveries.merge(matches,how='left',left_on='match_id',right_on='id')
pivot = pd.pivot_table(data=new_df,values='total_runs',index='batting_team',columns='city',aggfunc='sum').fillna(0)

fig,ax = plt.subplots(30,figsize=(5,100))
flag=0
for i in matches.city.unique():
    data = pivot.loc[:,i].sort_values(ascending=False)
    sns.barplot(data,data.index,ax=ax[flag])
    flag+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




