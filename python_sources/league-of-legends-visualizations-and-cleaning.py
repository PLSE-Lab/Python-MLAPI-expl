#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


champs = pd.read_csv("../input/champs.csv")
champs.head()


# In[ ]:


matches = pd.read_csv("../input/matches.csv")
matches.head()


# In[ ]:


participants = pd.read_csv('../input/participants.csv')
participants.tail()


# In[ ]:


stats1 = pd.read_csv('../input/stats1.csv')
stats1.head(2)


# In[ ]:


stats2 = pd.read_csv('../input/stats2.csv')
stats2.head(2)


# In[ ]:


stats = stats1.append(stats2)
stats.shape


# In[ ]:


stats.head()


# ## Some Data Cleaning

# #### putting all together in one DataFrame

# In[ ]:


df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))

df = pd.merge(df , champs, how = 'left', left_on= 'championid', right_on='id'
             ,suffixes=('', '_y') )

df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id'
              , suffixes=('', '_y'))


# In[ ]:


df.columns


# ## Some Data Cleaning

# In[ ]:


def final_position(col):
    if col['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):
        return col['role']
    else:
        return col['position']


# In[ ]:


df['adjposition'] = df.apply(final_position, axis = 1)


# In[ ]:


df.head()


# In[ ]:


df['team'] = df['player'].apply(lambda x: '1' if x <= 5 else '2')
df['team_role'] = df['team'] + ' - ' + df['adjposition']


# In[ ]:


df.head()


# ### removing matchid with duplicate roles

# In[ ]:


remove_index = []
for i in ('1 - MID', '1 - TOP', '1 - DUO_SUPPORT', '1 - DUO_CARRY', '1 - JUNGLE',
          '2 - MID', '2 - TOP', '2 - DUO_SUPPORT', '2 - DUO_CARRY', '2 - JUNGLE'):
    df_remove = df[df['team_role'] == i].groupby('matchid').agg({'team_role':'count'})
    remove_index.extend(df_remove[df_remove['team_role'] != 1].index.values)


# ###  remove unclassified BOT, as correct ones should be DUO_SUPPORT OR DUO_CARRY

# In[ ]:


remove_index.extend(df[df['adjposition'] == 'BOT']['matchid'].unique())
remove_index = list(set(remove_index))


# ## Before & After Cleaning

# In[ ]:


print('# matches in dataset before cleaning:{}'.format(df['matchid'].nunique()))
df = df[~df['matchid'].isin(remove_index)]
print('# matches in dataset after cleaning: {}'.format(df['matchid'].nunique()))


# In[ ]:


df.columns


# ### The Columns we need

# In[ ]:


df = df[['id', 'matchid', 'player', 'name', 'adjposition', 'team_role',
         'win', 'kills', 'deaths', 'assists', 'turretkills','totdmgtochamp',
         'totheal', 'totminionskilled', 'goldspent', 'totdmgtaken', 'inhibkills',
         'pinksbought', 'wardsplaced', 'duration', 'platformid',
         'seasonid', 'version']]
df.head()


# ## EDA (Exploratory Data Analysis)

# In[ ]:


df_v = df.copy()
# Putting ward limits
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x<30 else 30)
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x>0 else 0)

df_v['wardsplaced'].head()


# In[ ]:


plt.figure(figsize=(12,10))
sns.violinplot(x='seasonid', y= 'wardsplaced', hue='win', data= df_v, split = True
              , inner= 'quartile')
plt.title('Wardsplaced by season : win & lose')


# we can notice that wards are getting more popular and growing everyseason in both winning & losing games.

# In[ ]:


df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

m = np.zeros_like(df_corr.corr(), dtype=np.bool)
m[np.triu_indices_from(m)] = True

plt.figure(figsize=(16,10))
sns.heatmap(df_corr.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f',
            linewidths=.5, mask = m)

plt.title('Correlations - win vs factors (all games)')


# if you never played the game, you would find these info interesting !
# * deaths affect badly on win rate
# * kills goes well with goldspent & totdmgtochamp 
# * deaths propotional with duration & totdmgtaken
# * more goldspent at late game ( more duration )
# * totminionkilled aka farming goes well with totdmgtochamp aka damaging enemy champs ALSO more  goldspent ofcourse.

# #### This is kinda generic so we will split the heatmap into:
#  games less than 25mins
#  & games more than 25min 

# In[ ]:


df_corr_2 = df._get_numeric_data()
# for games less than 25mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]
df_corr_2 = df_corr_2.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

m = np.zeros_like(df_corr_2.corr(), dtype=np.bool)
m[np.triu_indices_from(m)] = True

plt.figure(figsize=(16,10))
sns.heatmap(df_corr_2.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f',
            linewidths=.5, mask = m)

plt.title('Correlations - win vs factors (for games last less than 25 mins)')


# Correlations here are stronger and more obvisious:
# * kills & deaths affect strongly the winning process
# * also assits & turretkills affect the winning process 
# * kills has strong relation with goldspent
# * more goldspent means more totdamagetochamp means more likely to earn kills

# In[ ]:


df_corr_3 = df._get_numeric_data()
# for games more than 40mins
df_corr_3 = df_corr_3[df_corr_3['duration'] > 2400]
df_corr_3 = df_corr_3.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

m = np.zeros_like(df_corr_3.corr(), dtype=np.bool)
m[np.triu_indices_from(m)] = True

plt.figure(figsize=(16,10))
sns.heatmap(df_corr_3.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f',
            linewidths=.5, mask = m)

plt.title('Correlations - win vs factors (for games last less than 40 mins)')


# So in the late game as gamers call it OR after 40 mins of game time we find that:
# * deaths & kills doesnt even matter alot and have very poor correlation with game winning.
# * inhibkills & turretkills have about 25% chance of winning the game(still not big correlation).
# * kills have high correlation with goldspent & totdmgtochamp.
# * assists have 40% corr with wardsplaced ( as this is the support's job) also -43% with totminionkilled( supports don't farm alot) and -32% with kills.
# 

# ### Top win rate champions:

# In[ ]:


pd.options.display.float_format = '{:,.1f}'.format


df_win_rate = df.groupby('name').agg({'win': 'sum','name': 'count',
                                     'kills':'mean','deaths':'mean',
                                     'assists':'mean'})
df_win_rate.columns = ['win' , 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win'] / df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate',ascending= False)
df_win_rate = df_win_rate[['total matches', 'win rate' , 'K' , 'D', 'A', 'KDA']]


print('Top 10 win rate')
print(df_win_rate.head(10))
print('Least 10 win rate')
print(df_win_rate.tail(10))


# In[ ]:


df_win_rate.reset_index(inplace= True)


# In[ ]:


# plotting the result visually
plt.figure(figsize=(16,30))
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="win rate", y="name", hue='KDA',
                     palette=cmap, sizes=(10, 200),
                     data=df_win_rate)


# In[ ]:


df_win_rate.head()


# ## Counter pick advices !

# In[ ]:


df_2 = df.sort_values(['matchid','adjposition'], ascending = [1,1])

df_2['shift 1'] = df_2['name'].shift()
df_2['shift -1'] = df_2['name'].shift(-1)

def matchup(x):
    if x['player'] <= 5:
        if x['name'] < x['shift -1']:
            name_return = x['name'] + ' vs ' + x['shift -1']
        else:
            name_return = x['shift -1'] + ' vs ' + x['name']
    else:
        if x['name'] < x['shift 1']:
            name_return = x['name'] + ' vs ' + x['shift 1']
        else:
            name_return = x['shift 1'] + ' vs ' + x['name']
    return name_return

df_2['matchup'] = df_2.apply(matchup, axis = 1)
df_2['win_adj'] = df_2.apply(lambda x: x['win'] if x['name'] == x['matchup'].split(' vs')[0]
                            else 0, axis = 1)

df_2.head()


# In[ ]:


df_matchup = df_2.groupby(['adjposition', 'matchup']).agg({'win_adj': 'sum', 'matchup': 'count'})
df_matchup.columns = ['win matches', 'total matches']
df_matchup['total matches'] = df_matchup['total matches'] / 2
df_matchup['win rate'] = df_matchup['win matches'] /  df_matchup['total matches']  * 100
df_matchup['dominant score'] = df_matchup['win rate'] - 50
df_matchup['dominant score (ND)'] = abs(df_matchup['dominant score'])
df_matchup = df_matchup[df_matchup['total matches'] > df_matchup['total matches'].sum()*0.0001]

df_matchup = df_matchup.sort_values('dominant score (ND)', ascending = False)
df_matchup = df_matchup[['total matches', 'dominant score']]                   
df_matchup = df_matchup.reset_index()

print('Dominant score +/- means first/second champion dominant:')

for i in df_matchup['adjposition'].unique(): 
        print('\n{}:'.format(i))
        print(df_matchup[df_matchup['adjposition'] == i].iloc[:,1:].head(5))


# In[ ]:


df_matchup['adjposition'].unique()

df_matchup_TOP = df_matchup.loc[df_matchup['adjposition'] == 'TOP']
df_matchup_JUNGLE = df_matchup.loc[df_matchup['adjposition'] == 'JUNGLE']
df_matchup_MID = df_matchup.loc[df_matchup['adjposition'] == 'MID']
df_matchup_DUO_CARRY = df_matchup.loc[df_matchup['adjposition'] == 'DUO_CARRY']
df_matchup_DUO_SUPPORT = df_matchup.loc[df_matchup['adjposition'] == 'DUO_SUPPORT']


print(df_matchup_TOP.shape)
print(df_matchup_JUNGLE.shape)
print(df_matchup_MID.shape)
print(df_matchup_DUO_CARRY.shape)
print(df_matchup_DUO_SUPPORT.shape)


# In[ ]:


# plotting duo carry 
plt.figure(figsize=(16,60))
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_DUO_CARRY,
            label="Total", color="b")


# If we plot the ADC ( DUO_CARRY) for an example, we notice:
# * the negative values means the LEFT champion dominates ( kalista vs kogmaw scored -12.5 means kalista dominates by far)
# * The positive values  means the RIGHT champion dominates (Graves vs Tristana scored +5.5 means Tristana dominates by far)
# * While we approach zero from both sides means both champions have balanced dominance points ( MissFortune vs Caitlyn). so its totally up to your skills ;) 

# In[ ]:


# plotting TOP

plt.figure(figsize=(16,200))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_TOP,
            label="Total", color="c")


# In[ ]:


# plotting jungle

plt.figure(figsize=(16,100))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_JUNGLE,
            label="Total", color="g")


# In[ ]:


# plotting mid

plt.figure(figsize=(16,100))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_MID,
            label="Total", color="r")


# In[ ]:


# plotting support

plt.figure(figsize=(16,100))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_DUO_SUPPORT,
            label="Total", color="m")


# ## Thanks, Thats all for now
