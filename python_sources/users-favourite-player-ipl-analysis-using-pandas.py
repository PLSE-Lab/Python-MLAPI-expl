#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


deliveries=pd.read_csv('../input/ipl/deliveries.csv',sep=',')


# In[ ]:


matches=pd.read_csv('../input/ipl/matches.csv',sep=',')


# In[ ]:


deliveries


# In[ ]:


matches


# In[ ]:


deliveries.shape


# In[ ]:


matches.shape


# In[ ]:


matches.shape


# In[ ]:


matches.head()


# In[ ]:


matches['match_id']=matches['id']
matches.head()


# In[ ]:


ipl=deliveries.merge(matches,on='match_id',how='inner')


# In[ ]:


ipl


# In[ ]:


ipl.shape


# In[ ]:


ipl.isnull().any()


# In[ ]:


match=matches


# In[ ]:


match.isnull().any()


# In[ ]:


match.shape


# In[ ]:


match.isnull().any()


# In[ ]:


match.fillna(0)


# In[ ]:


ipl.describe()


# In[ ]:


ipl.shape


# In[ ]:


ipl.head()


# In[ ]:


ipl.columns


# In[ ]:


no_of_years=ipl['season'].unique()
no_of_years


# In[ ]:


team_names1=ipl['team1'].unique()


# In[ ]:


team_names1


# In[ ]:


team_names2=ipl['bowling_team'].unique()


# In[ ]:


team_names2


# In[ ]:


team_names1.shape


# In[ ]:


team_names2.shape


# In[ ]:


csk_details=ipl['team1'].str.contains('Chennai Super Kings') | ipl['team2'].str.contains('Chennai Super Kings')
csk_details


# In[ ]:


cskmatches=ipl[csk_details]


# In[ ]:


cskmatches


# In[ ]:


cskmatches.shape


# In[ ]:


everycskmatch=cskmatches['match_id'].unique()
everycskmatch


# In[ ]:


csk_match=ipl.loc[ipl.match_id.isin(everycskmatch)]
csk_match


# In[ ]:


csk_match.shape


# In[ ]:


csk_years=csk_match.season.unique()
csk_years


# In[ ]:


cskmatchlist=ipl.match_id.loc[everycskmatch]


# In[ ]:


cskmatchlist.shape


# In[ ]:


onlycskmatch=matches['team1'].str.contains('Chennai Super Kings') | matches['team2'].str.contains('Chennai Super Kings')
onlycskmatch


# In[ ]:


onlycskmatches=matches[onlycskmatch]


# onlycskmatches=matches[onlycskmatch]

# In[ ]:


onlycskmatches


# In[ ]:


uniqueplayerofmatches=onlycskmatches['player_of_match'].unique()


# In[ ]:


uniqueplayerofmatches


# In[ ]:


cskwins=onlycskmatches.winner=='Chennai Super Kings'


# In[ ]:


cskwinmatches=onlycskmatches[cskwins]


# In[ ]:


cskwinmatches


# In[ ]:


cskwinmatches.shape


# In[ ]:


pomat=cskwinmatches.player_of_match.value_counts()
pomat


# In[ ]:


highwinners=matches.winner.value_counts()
highwinners


# In[ ]:


manofmatches=matches.player_of_match.value_counts()
manofmatches


# In[ ]:


umargul=matches.player_of_match=='Umar Gul'
umargulmatch=matches[umargul]
umargulmatch


# In[ ]:


#csd=ipl.match_id=='112'
#csd
performumargul=ipl[ipl.match_id==112]
performumargul


# In[ ]:


performumargul.shape


# In[ ]:


onlyhis=performumargul.batsman.str.contains('Umar Gul')|performumargul.bowler.str.contains('Umar Gul')


# In[ ]:


onlyhisper=performumargul[onlyhis]


# In[ ]:


onlyhisper


# In[ ]:


onlyhisper.shape


# In[ ]:


onlyhisper.columns


# In[ ]:


umargulbat=onlyhisper.batsman.str.contains('Umar Gul')


# In[ ]:


umargulbatting=onlyhisper[umargulbat]
umargulbatting


# In[ ]:


umargulruns=umargulbatting.batsman_runs
umargulruns


# In[ ]:


s=0
for di in umargulruns:
    s=s+di
print(s)


# In[ ]:


umarbowl=onlyhisper.bowler.str.contains('Umar Gul')
umarbowling=onlyhisper[umarbowl]
umarbowling


# In[ ]:


wickets=umarbowling['player_dismissed']
valwic=wickets.dropna()


# In[ ]:


valwic


# In[ ]:


wicketstook=valwic.count()


# In[ ]:


print(s,wicketstook)


# In[ ]:


showme=umarbowling[[ 'wide_runs',
       'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs',
       'batsman_runs', 'extra_runs', 'total_runs', 'player_dismissed',
       'dismissal_kind']]
showme


# In[ ]:


umarbowlruns=umarbowling[[ 'wide_runs',
       'noball_runs',
       'batsman_runs']]
umarbowlruns


# In[ ]:


wides=(umarbowlruns.wide_runs>0 )| (umarbowlruns.noball_runs>0) |(umarbowlruns.batsman_runs>0) 
wide=umarbowlruns[wides]
runs=wide.batsman_runs
#wideru=wide.wide_runs.value_counts()
#wideru


# In[ ]:


runs


# In[ ]:


runss=wide.noball_runs
runsss=wide.wide_runs
runss


# In[ ]:


runsss


# In[ ]:


d=0
for c in runs:
    d=d+c
e=0
for f in runss:
    e=e+f
h=0
for g in runsss:
   h=h+g
xs=d+e+h
print(xs,s)


# to find player data on user choice

# In[ ]:


pla=input('enter player name')


# In[ ]:


print(pla)


# In[ ]:


ipl


# In[ ]:


ipl.shape


# In[ ]:


matches


# In[ ]:


matches.shape


# do you want to find how many player of match he won?

# In[ ]:


pomuser=(matches.player_of_match==pla)


# In[ ]:


pla_pom=matches[pomuser]


# In[ ]:


pla_pom


# In[ ]:


pla_pom.player_of_match.count()


# how much runs he scored in ipl from 2008-2017?

# In[ ]:


ipl.head()


# In[ ]:


ipl.columns


# In[ ]:


onlyhisbat=ipl[ipl['batsman']==pla]
onlyhisbat


# In[ ]:


hisruns=onlyhisbat['batsman_runs'].sum()


# In[ ]:


hisruns


# In[ ]:


checkruns=onlyhisbat['batsman_runs']
checkruns


# In[ ]:


su=0;
for dp in checkruns:
    su=su+dp
print(su)


# u want to find wickets?
# 

# In[ ]:


onlyhisbowl=ipl[ipl.bowler==pla]
onlyhisbowl


# In[ ]:


onlyhisbowl.dismissal_kind.unique()


# In[ ]:


takewicket=[ 'caught', 'bowled', 'lbw', 'caught and bowled']


# In[ ]:


wickets=onlyhisbowl[onlyhisbowl.dismissal_kind.isin(takewicket)]
wickets


# In[ ]:


wickets.dismissal_kind.count()


# In[ ]:


pws=ipl[['season','bowler','dismissal_kind']].groupby(['season','bowler']).count().reset_index()
pws


# In[ ]:


pws=pws.sort_values('dismissal_kind',ascending=False)


# In[ ]:


pws=pws.drop_duplicates('season',keep='first')


# In[ ]:


pws


# In[ ]:


pws.sort_values('season')


# In[ ]:


rpes=ipl[ipl.batsman==pla]
rpes


# In[ ]:


runperseason=rpes[['match_id','season','batsman_runs']].groupby(['match_id','season']).count().reset_index()


# In[ ]:


runperseason=runperseason.groupby(['season']).sum()


# In[ ]:


runperseason.drop(['match_id'],axis=1,inplace=True) 


# In[ ]:


sur=runperseason.reset_index()


# In[ ]:


import matplotlib.pyplot as mlt


# In[ ]:


sur.columns
#seasons=runperseason.unique()
#seasons


# In[ ]:


seasons=sur.season.unique().tolist()
seasons


# In[ ]:


runs=sur.batsman_runs.tolist()
runs


# In[ ]:


mlt.figure(figsize=(10,5))
ax=mlt.bar(seasons,runs)
# Label the axes
mlt.xlabel('season')
mlt.ylabel('runs')


#label the figure
mlt.title('runs per year')
mlt.show()


# In[ ]:


pla='JH Kallis'
bwst=ipl[ipl.bowler==pla]
bwst


# In[ ]:


bowling=bwst[['season','dismissal_kind']].groupby('season').count()
bowling


# In[ ]:





# In[ ]:


bowlin=bowling.reset_index()


# In[ ]:


mlt.plot(bowlin['season'].values,bowlin['dismissal_kind'].values)

mlt.xlabel('season')
mlt.ylabel('wickets')


#label the figure
mlt.title('wickets per year')
mlt.show()


# In[ ]:





# In[ ]:




