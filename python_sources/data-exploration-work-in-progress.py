#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# ## Background
# 
# The data set consists of data for eight seasons from the national india cricket league (IPL). Cricket is played in turns like in baseball, golf, american football or chess and in contrast to football, boxing or basketball which allows easier data capturing. The IPL playes T20 cricket, where 20 overs (a 6 balls) are played and one team bats and the other bowls(fields) first. The dataset describes the possible outcomes of each of these balls throwns.  
# 
# Only the batting team scores until one on two ressources is depleted, the balls are the facing or the players able to score (wickets).
# 
# ## Approach
# We explore first the data set technically and analyse its data quality. Afterwards we cleanse and enrich the data set, esp. introduced the perspective of looking at overs. The next chapter performs describe statistics and visualisation on the data. Then we make an effort derive hidden structures in the dataset and prepare it for some machine learning. The following notebook should be considered as a work-in-progress data exploration, rather than well told and completed data story.
# 
# The author has watch numerous games (mostly partially) of cricket whilst living in the UK, however never played a game. He takes no sides in the IPL.
# 
# 
# ## Change Log
# 
# 13/08/2017 Serious start at the data set<br>
# 14/08/2017 Chase chart<br>
# 16/08/2017 Minor work <br>
# 17/08/2017 Data enriching<br>
# 20/08/2017 Match grouper <br>
# 21/08/2017 Add wickets to the chase<br>
# 31/08/2017 Sanity checks on winning teams<br>
# 01/09/2017 Stats on wins<br>
# 03/09/2017 Stats on wins<br>
# 04/09/2017 Runs per over<br>
# 05/09/2017 Use categoricals, refactor<br>
# 27/09/2017 Some commentary <br>
# 28/09/2017 Commentary and review <br>
# 29/09/2017 Stats on win percentage by batting team<br>
# 01/10/2017 More enriching on basic frames<br>
# 02/10/2017 Required run rate<br>
# 03/10/2017 Graph on required runrate<br>
# 04/10/2017 Fall of wickets by over<br>
# 05/10/2017 Hattricks, Maiden<br>
# 06/10/2016 Streaks<br>
# 07/10/2017 Lookup Playoff games in wikipedia<br>
# 08/10/2017 Start on batsman's stats<br>
# 09/10/2017 Batman stats<br>
# 10/10/2017 Tidy up <br>
# 11/10/2017 Commentary, batting pairplot <br>
# 23/10/2017 upload after technical fault <br>
# 

# In[ ]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-pastel')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
dl=pd.read_csv(r"../input/deliveries.csv")
mt=pd.read_csv(r"../input/matches.csv")


# # Enrich data & Data cleansing
# 
# ## Matches - mt
# * inn1bat - Team batting in 1st inning 
# * inn2bat - Team batting in 2nd inning
# * overwrite winner to {'tie', 'no result'} if match is tied/abandoned 
# * winner - filling nulls with "tie" and "no result"
# * winner2 - (not team name) {bat_wins, field_bat, no result, tie}
# * wkts_fallen\_inn\_"n" - Number of wickets fallen in "n"-th inning (n={1,2} for regular play, {3,4} for super over)
# * runs\_inn\_"n" - Number of wickets fallen in "n"-th inning (n={1,2} for regular play, {3,4} for super over)
# * Stage - enriched from wikipedia {'Group', 'Semifinal','Qualifier1-2','Qualifier3-4','Loser1-2Winner3-4','Final'}
# 
# 
# ## Over - ov
# new frame, simple aggregations by over
# 
# * ball - balls in that over (incl. extras)
# * total_runs - runs per over
# * innpart - part of the inning; powerplay (1-6), Middle (7-15), Death (16-20)
#   (following http://www.espncricinfo.com/story/_/id/19065251/ipl-best-men-powerplay-middle-death)
# * cum_total_runs - cumulative sum of runs per inning
# * cumrunrate - cumulative runrate after the over.
# * reg_run - required run rate (only 2nd inning)
# 
# 
# ## Deliveries - dl
# 
# * cum_over - Number of over as decimal fraction per ball. 
# * cum_runs - cumulative score
# * cum_ball - cumulative ball in over
# 
# ## Batting
# new frame, aggregation for batmans
# * batting_order - position the batsman entered match
# 

# In[ ]:


NUMBER_OF_GAMES_ = mt.id.max()
TEAMS_ = mt.team1.unique()


# In[ ]:


# Matches - mt
#inn1bat, inn2bat
mt.loc[(mt.toss_decision=='bat') & (mt.toss_winner==mt.team1)
       |(mt.toss_decision=='field') & (mt.toss_winner==mt.team2),
       'inn1bat']=mt.team1
mt.loc[(mt.toss_decision=='bat') & (mt.toss_winner==mt.team2)
       |(mt.toss_decision=='field') & (mt.toss_winner==mt.team1),
       'inn1bat']=mt.team2
mt.loc[mt.inn1bat==mt.team1,'inn2bat'] = mt.team2
mt.loc[mt.inn1bat==mt.team2,'inn2bat'] = mt.team1

#winner, winner2
mt.loc[mt.result=='tie'      ,'winner']  = 'tie'
mt.loc[mt.result=='no result','winner']  = 'no result'

mt.loc[mt.result=='tie'      ,'winner2'] = 'tie'
mt.loc[mt.result=='no result','winner2'] = 'no result'
mt.loc[mt.winner==mt.inn1bat ,'winner2'] = 'bat_wins'
mt.loc[mt.winner==mt.inn2bat ,'winner2'] = 'field_wins'

# wickets fallen in inning 1
mt=pd.merge(mt,
            dl.groupby(['match_id', 'inning'] )\
                ['player_dismissed'].count()\
                .to_frame()\
                .loc[:,'player_dismissed']\
                .unstack('inning')\
                .rename(columns={1:'wkts_fallen_inn_1',
                                 2:'wkts_fallen_inn_2',
                                 3:'wkts_fallen_inn_3',
                                 4:'wkts_fallen_inn_4'})\
                .reset_index(),
            how='inner',left_on='id', right_on='match_id' )

# runs in inning
mt=pd.merge(mt,
            dl.groupby(['match_id','inning'])['total_runs'].sum().to_frame().loc[:,'total_runs']\
                .unstack('inning')\
                .rename(columns={1:'runs_inn_1',
                                 2:'runs_inn_2',
                                 3:'runs_inn_3',
                                 4:'runs_inn_4'})\
                .reset_index(),
            how='inner',left_on='id', right_on='match_id' )

# Stage (From wikipedia)
mt.loc[:, 'Stage'] = 'Group'
mt.loc[mt.id.isin([56,57,113,114,173,174])             , 'Stage'] = 'Semifinal'
mt.loc[mt.id.isin([58,115,175,248,322,398,458,517,577]), 'Stage'] = 'Final'
mt.loc[mt.id.isin([245,319,395,455,514,574])           , 'Stage'] = 'Qualifier1-2'
mt.loc[mt.id.isin([246,320,396,456,515,575])           , 'Stage'] = 'Qualifier3-4'
mt.loc[mt.id.isin([247,321,397,457,516,576])           , 'Stage'] = 'Loser1-2Winner3-4'


# In[ ]:


# Overs - ov
ov=dl.groupby(['match_id', 'inning', 'over'],as_index=False)     .agg({'ball':'max',
           'total_runs':'sum'})

# innpart,     
ov['innpart']=pd.cut(ov.over,
                     bins=[0,6,15,99],
                     labels=['powerplay','middle','death'])

# cum_total_runs
ov['cum_total_runs'] = ov.groupby(['match_id','inning'])                          ['total_runs'].transform(pd.Series.cumsum)
ov['cumrunrate']=ov.cum_total_runs/ov.over

# required run rate (only for 2nc inning)
ov['run_inn_1']=pd.merge(ov,
                         mt,
                         how='inner', left_on='match_id', right_on='id')['runs_inn_1']

# reqRR
ov['reqRR']=(ov['run_inn_1']-ov['cum_total_runs'])/(20-ov.over)
ov.loc[ov.inning!=2,'reqRR']=np.nan
ov.drop('run_inn_1', axis=1)

ov.head()


# In[ ]:


dl=pd.merge(dl,ov, how='left', left_on=['match_id', 'inning', 'over'],
           right_on=['match_id', 'inning', 'over'], suffixes=['','_ov'])
dl['dummy1']=1
dl['cum_ball'] = dl.groupby(['match_id','inning'])['dummy1'].transform(pd.Series.cumsum)


# In[ ]:


for m in range(1,NUMBER_OF_GAMES_+1):
    for i in range(1,2+1):
        dl.loc[(dl.match_id==m) & (dl.inning==i),'cum_over']=            dl[(dl.match_id==m) & (dl.inning==i)].over-1            +dl[(dl.match_id==m) & (dl.inning==i)].ball            /dl[(dl.match_id==m) & (dl.inning==i)].ball_ov
        dl.loc[(dl.match_id==m)& (dl.inning==i),'cum_runs']=            dl[(dl.match_id==m)& (dl.inning==i)].total_runs.cumsum()


# In[ ]:


batting=pd.merge(dl.groupby(['match_id','inning', 'batsman'], as_index=False)                   .agg({'batsman_runs':'sum', 'cum_ball':'first'}),
                 mt[['id','season', 'inn1bat', 'inn2bat']],
                 how='inner', left_on='match_id', right_on='id')
batting.loc[batting.inning==1,'team']=batting['inn1bat']
batting.loc[batting.inning==2,'team']=batting['inn2bat']
batting=batting[['season','match_id', 'inning' ,'team', 'batsman', 'batsman_runs', 'cum_ball' ]]               .sort_values(['season','match_id', 'inning', 'cum_ball'])
batting['batting_order'] = batting.groupby(['season','match_id','inning'])['batsman_runs'].cumcount()+1


# # Popular statistics
# 
# ## "The chase"
# One of the most common statistics in tv coverage is a "worm" graph aka "the chase". It displays the cumulative runs a team scores during the overs of an inning. It allows a comparison of both teams in a single graph. Usually the fall of wickets is displayed as well. 

# In[ ]:


def drawChase(id):
    fig, ax = plt.subplots(1,1)
    team1, team2=mt.loc[mt.id==id,['inn1bat', 'inn2bat']].values[0]
    for i,t, c in zip((1,2), (team1, team2), ('green', 'blue')):
        ax=plt.plot(dl[(dl.match_id==id) & (dl.inning==i)].cum_over.values,
                    dl[(dl.match_id==id) & (dl.inning==i)].cum_runs.values,
                    label=t, color=c, alpha=0.5)
        ax=plt.scatter(y=dl[(dl.match_id==id) & (dl.inning==i)&(~dl.player_dismissed.isnull())]                           ['cum_runs'],
                       x=dl[(dl.match_id==id) & (dl.inning==i)&(~dl.player_dismissed.isnull())]\
                           ['cum_over'],
                       label=t+" Wickets", color=c)
    
    plt.legend(loc='best')
    plt.xlim(-1,21)
    plt.ylim(-1,250)
    plt.title('Chase');

drawChase(577) #577 the last final(2017)


# # High Level Statistics
# 
# ## Seasons
# 
# The modus in which a season has been played and which teams have participated in the league has changed numerous times. Best to consult different sources as this is not explained in the data set. https://en.wikipedia.org/wiki/Indian_Premier_League

# In[ ]:


mt.groupby('season', as_index=False)  ['id'].count()  .plot(x='season',kind='bar', legend=False)
plt.title('Games per Season');


# In[ ]:


mt.groupby(['team1', 'season'], as_index=False)    .agg({'id':lambda x: 1})    .set_index(['team1', 'season'])    .unstack('season')    .fillna("")


# ## Matches / Games
# 
# A cricket match end with a normal result, i. e. one of the team win. Unusual is the case of tie; in this case an extra over is played as a tie breaker. Another usual result is that the play is stopped e.g. due to weather conditions and no winner can be determined. This is marked as no result. 
# In some cases, if the played has to be stopped a winner can be extrapolated by statistical method "D/L". This is classified as a normal result.

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(14,5))
mt.groupby(['result'])   ['id'].count()   .plot(kind='pie', ax=ax[0])
ax[0].set_ylabel("")    
ax[0].set_title("'Result Type' of the matches");

g=mt.groupby(['dl_applied'],as_index=False)    ['id'].count()
g.loc[:,'dl_applied']=g.loc[:,'dl_applied'].map({0:'not applied', 1: 'applied'})
g=g.set_index('dl_applied')
g['id'].plot(kind='pie', ax=ax[1])
ax[1].set_ylabel("")
ax[1].set_title('D/L applied');


# In[ ]:


print('# Total Matches', len(mt))
print('# D/L applied:', len(mt[mt.dl_applied==1]))
print('# Winners bats first', len((mt[(mt.winner==mt.inn1bat)&(mt.dl_applied==0)])))
print('# Winners by runs', len(mt[(mt.win_by_runs>0)&(mt.dl_applied==0)]))
print('# Winners bats second', len(mt[(mt.winner==mt.inn2bat)&(mt.dl_applied==0)]))
print('# Winners by wickets', len(mt[(mt.win_by_wickets>0)&(mt.dl_applied==0)]))

print('# no winner', len(mt[mt.winner.isin(['tie','no result'])]))


# In due course we analyse the perspecive of both team; the battting (first) team and the fielding (first) team.
# The batting team can only win a game by runs, and they will lose a game by wickets et vice versa. Applied D/L is corner case which we exclude here.
# 
# ### Wins by field team
# 
# The histogram shows that the majorities of wins by the batting team have marging smaller than 30 runs. Afterwards the curves decreases steeper.  
# 

# In[ ]:


mt.loc[(mt.winner==mt.inn1bat)&(mt.dl_applied==0),['win_by_runs']]  .plot(kind='hist', bins=20)
plt.title('Batting teams wins by number of runs');


# In[ ]:


field_wins =     pd.merge(mt.loc[(mt.winner==mt.inn2bat)&(mt.dl_applied==0)], 
         dl, how='inner',
         left_on='id', right_on='match_id')
field_wins[field_wins.inning==1]     .groupby(['id','inning'])     .agg({'total_runs':'sum',
          'win_by_wickets':'first'})\
     .plot(kind='scatter', x='total_runs', y='win_by_wickets');
plt.title('Field team win by wickets, despite number of runs by bat team');


# ### Wins by batting team

# In[ ]:


mt.loc[(mt.winner==mt.inn2bat)&(mt.dl_applied==0),['win_by_wickets']]  .plot(kind='hist', bins=10);


# In[ ]:


g=pd.merge(mt.loc[(mt.winner==mt.inn1bat)&(mt.dl_applied==0)], 
           dl, how='inner',
           left_on='id', right_on='match_id')
g=g[(g.winner2=='bat_wins')& (g.inning==1)]   .groupby(['id'])   .agg({'total_runs':'sum',
         'win_by_runs':'first'})

ax1 = sns.jointplot("total_runs", "win_by_runs", data=g, ratio=5)
ax1.fig.suptitle('Bat team win by runs, based on number of runs');
ax1.set_axis_labels('Total runs by batting (first) team',
                    'Difference in runs to field / batting second team')
plt.subplots_adjust(top=0.95)


# Question is how many runs do the batting team need to ensure a win

# In[ ]:


g=pd.merge(mt.loc[(mt.dl_applied==0)], dl,
           how='inner', left_on='id', right_on='match_id')
g=g.groupby(['winner2','id','inning'])    .agg({'total_runs': 'sum'})    .unstack(['inning'])
g.loc['bat_wins','total_runs'].loc[:,1].plot(kind='kde', label='bat_wins')
g.loc['field_wins','total_runs'].loc[:,1].plot(kind='kde', label='field_wins')
plt.legend()
plt.title('Runs by batting team and result of the match');


# In[ ]:


g=pd.merge(mt.loc[(mt.dl_applied==0) & (mt.winner2!='no result')], dl,
           how='inner', left_on='id', right_on='match_id')
g=g[g.inning==1].groupby(['winner2','id','inning'], as_index=False)    .agg({'total_runs': 'sum'})    .sort_values('total_runs')
g['range']=pd.cut(g.total_runs, range(0, 300, 10), right=False)
gg=g.groupby(['range','winner2'])    ['id'].count()    .to_frame()    .unstack('winner2')    .fillna(0)    .loc[:,'id']
gg['games']=gg['bat_wins']+gg['field_wins']+gg['tie']
for c in ['bat_wins','field_wins','tie']:
    gg[c+'_pct']=gg[c]/gg['games']
gg[['bat_wins_pct','tie_pct','field_wins_pct']].plot.bar(stacked=True);
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.title('Wins percentage based on batting teams runs');
#todo: Clean spaces in the categorical


# ## Wickets 

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5))
mt.groupby('wkts_fallen_inn_1')['id'].count().plot.bar(ax=ax1)
ax1.set_ylim(0,100)
ax1.set_title('Wickets fallen in first inning')
mt.groupby('wkts_fallen_inn_2')['id'].count().plot.bar(ax=ax2)
ax2.set_ylim(0,100)
ax2.set_title('Wickets fallen in second inning');


# In[ ]:


fwkts=dl[dl.player_dismissed.notnull()].groupby(['match_id','inning','over'])          ['player_dismissed'].count()          .reset_index()
g=fwkts.groupby(['inning','over'])['player_dismissed'].sum().reset_index()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5), sharey=True)
g[g.inning==1]['player_dismissed'].plot.bar(ax=ax1)
ax1.set_xticklabels(np.arange(1,21,1))
ax1.set_title('Fall of wickets in 1st inning, n-th over')
g[g.inning==2]['player_dismissed'].plot.bar(ax=ax2, xticks=np.arange(1,20,1))
ax2.set_xticklabels(np.arange(1,21,1))
ax2.set_title('Fall of wickets in 2nd inning, n-th over');


# In[ ]:


df=dl[dl.player_dismissed.notnull()].copy() # copy required
print('Number of hattricks')
df[(df['cum_ball']==df['cum_ball'].shift(1)-1) &(df['cum_ball']==df['cum_ball'].shift(2)-2)].shape[0]


# ## Delivery analysis
# 
# ### Overs
# Usual an over consits of six balls; in case a ball is thrown incorrectly, it has to be repeated. Obviously an over can also be shorter than six balls, e.g. when the game ends

# In[ ]:


ov.groupby('ball')['over'].count().to_frame().T


# ## Phases of an inning
# 
# We follow Sundararaman and split an inning into three phases
# (http://www.espncricinfo.com/story/_/id/19065251/ipl-best-men-powerplay-middle-death ) i.e. "the Powerplay (1-6 overs), the middle overs (7-15) and the death (16-20)."
# 

# In[ ]:


ov.groupby('total_runs')    ['match_id'].count()    .reindex(np.linspace(0,ov.total_runs.max(),dtype='int32'), fill_value=0 )    .plot(kind='bar');
plt.title('Number of overs with x runs');


# In[ ]:


print('Maiden')
ov.groupby('total_runs')    ['match_id'].count()[0]


# In[ ]:


fig, ax = plt.subplots(1,3, figsize=(16,5))
for i,p,c in zip(range(3),['powerplay', 'middle', 'death'], ['g','y','k']):
    ov[ov.innpart==p].groupby('total_runs')        ['match_id'].count()        .reindex(np.linspace(0,ov.total_runs.max(),dtype='int32'), fill_value=0 )        .plot(kind='bar', ax=ax[i], color=c);
    ax[i].set_title('Number of overs with x runs (%s)'%p)    
    ax[i].set_ylim(0,1200)


# In[ ]:


fig, ax = plt.subplots(1,1)
ax=plt.bar(height=ov[ov.innpart=='powerplay'].groupby(['over'])['total_runs'].mean().values,
           left=np.linspace(0,6,6), color='g')
ax=plt.bar(height=ov[ov.innpart=='middle'].groupby(['over'])['total_runs'].mean().values,
           left=np.linspace(7,15,9), color='y')
ax=plt.bar(height=ov[ov.innpart=='death'].groupby(['over'])['total_runs'].mean().values,
           left=np.linspace(16,20,5), color='k')
#plt.title('Average runs per over in the match');


# # Required Run Rate
# 
# Initial analysis on the required run rate ignores the numbers of wickets. The lines shows the movement of the required runs throughout the game. Colors indicated the match outcome (Red:field_wins, Green, bat_wins)

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(14,10))
cm={'bat_wins': 'r', 'field_wins': 'g', 'tie': 'y'}
for i in range(1,NUMBER_OF_GAMES_):
    if mt[mt.id==i]['result'].values[0]!='no result':
        df=ov[(ov.inning==2) & (ov.match_id==i)& (ov.reqRR>0)][['reqRR', 'over']]
        df.plot(x='over',y='reqRR', ax=ax, legend=False,
                color=cm[mt[mt.id==i].winner2.values[0]],
                alpha=0.5, linewidth=1)
        plt.ylim(0,30)


# We can take cuts through that graph at certain points (e.g. after 1,5,10,15,19 overs) and plot the distributions.

# In[ ]:


watchpoints=[1,5,10,15,19]
fig, ax = plt.subplots(len(watchpoints),2, figsize=(14,5*len(watchpoints)))
for i in range(len(watchpoints)):
    g=pd.merge(ov[(ov.inning==2) & (ov.over==watchpoints[i])]                 [['match_id','reqRR', 'over']],
               mt[['id', 'result', 'winner2']],
               how='outer', left_on='match_id', right_on='id')
    g[g.match_id.isnull()].groupby('winner2')                           ['id'].count()                          .plot.bar(ax=ax[i,0], ylim=(0,130), rot=0)
    
    ax[i,0].set_title('Games ended before %s overs (2nd inning)'%watchpoints[i])
    g[g.winner2=='bat_wins']['reqRR'].replace(np.inf,20)       .plot.hist(bins=20, color='r', alpha=0.7,stacked=True, ax=ax[i,1], xlim=(0,20) )
    g[g.winner2=='field_wins']['reqRR'].replace(np.inf,20)       .plot.hist(bins=20, color='g', alpha=0.7, stacked=True, ax=ax[i,1], xlim=(0,20))
    ax[i,1].set_title('Match result based on the required run rate afer %s overs (2nd inning)'%watchpoints[i])


# In[ ]:


streaks=[]
for t in TEAMS_:   
    df = mt[(mt['team1']==t) | (mt['team2']==t)].copy()
    df['result2']= np.nan
    df['team'] = t
    df.loc[(df.winner==t) & (df.result=='normal'),'result2']= 'win'
    df.loc[(df.winner!=t) & (df.result=='normal'),'result2']= 'loss'
    df.loc[df.result=='tie', 'result2']= 'tie'
    df.loc[df.result=='no result', 'result2']= 'n/r'
    streaks.append(df[['season','result2','team']])
streaks = pd.concat(streaks, axis=0)

streaks = pd.concat([streaks,
                    streaks.groupby(['team', 'season'], as_index=False).cumcount()],
                    axis=1).rename(columns={0:'idx'})

def color_result(val):
    colormap = {'loss':'red','win':'green', 'tie':'brown', 'n/r':'grey', '':'grey' }
    color=colormap[val]
    return 'color: %s' % color

streaks.set_index(['season','team','idx'])       .unstack(['idx'])       .loc[:,'result2']       .replace(np.nan,'')       .style.applymap(color_result)

# Things don't add up; due to games canceled in advance


# # Players

# In[ ]:


batting.head()


# In[ ]:


df=batting.groupby('batsman')          .agg({'batsman_runs':sum,'match_id':np.count_nonzero})          .rename(columns={'batsman_runs':'total_runs','match_id':'games_played'})
df['runs_per_game']=df['total_runs']/df['games_played']
df[df.games_played>10].sort_values(by='runs_per_game', ascending=False).head(10)


# In[ ]:


df.plot.scatter(x='games_played',y='runs_per_game');


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(14,5))
g=batting.groupby('batsman_runs', as_index=False)         ['batsman'].count()
g['pct']=g['batsman']/g.batsman.sum()
g['pct'].plot.bar();
ax.set_xlim(-1,50)
ax.set_title('Number of runs a batsman has scored');


# In[ ]:


df=batting.groupby('batting_order')          .agg({'batsman_runs':sum,
                'match_id':np.count_nonzero})\
          .rename(columns={'batsman_runs':'total_runs',
                           'match_id'    :'games_played'})
df['runs_per_game']=df['total_runs']/df['games_played']
df['runs_per_game'].plot.bar()
plt.title('Runs by batting position');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




