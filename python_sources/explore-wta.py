#!/usr/bin/env python
# coding: utf-8

# <h1><B>Women's Tennis Association</b> </h1>

# In this project , we will use the CRISP-DM methodology.

# <h1><b> 1. Business understanding</b></h1>

# Tennis is a racket sport that can be played individually against a single opponent (singles) or between two teams of two players each (doubles).
# The object of the game is to maneuver the ball in such a way that the opponent is not able to play a valid return. The player who is unable to return the ball will not gain a point, while the opposite player will.

# <h2><b>1.1 WTA : </b></h2>
# 

# The Women's Tennis Association (WTA), founded in 1973 by Billie Jean King, is the principal organising body of women's professional tennis. It governs the WTA Tour which is the worldwide professional tennis tour for women and was founded to create a better future for women's tennis.

# <h2><b>1.2 Ranking method : </b></h2>
# 

# The WTA rankings are based on a rolling 52-week, cumulative system. A player's ranking is determined by her results at a maximum of 16 tournaments for singles and 11 for doubles and points are awarded based on how far a player advances in a tournament.

# <h2><b>1.3 Tournament categories : </h2></b>

# <ul>
# <li>Grand Slam tournaments (4)</li>
# <li>Year-ending WTA Tour Championships (1)</li>
# <li>Premier (20) </li>
# <li>International tournaments (32) </li>
# <li>WTA 125k Series </li>
# </ul>

# <h2><b> 1.4 courts   </b></h2>
# 

# <li>
# hard 
# </li>
# <li>
# clay 
# </li>
# <li>
# grass
# </li>
# <li>carpet </li>

# <h2><b>2. Data understanding</b></h2>

# <h2><b>2.1 Collect initial data </b></h2>

# Dataset from Kaggle : <a><href>https://www.kaggle.com/residentmario/exploring-wta-players/notebook</href></a>
# 
# 

# <h2><b> 2.2 Data Description </b></h2>

# <b>players.csv </b>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

players = pd.read_csv('../input/players/players.csv', encoding='latin1', index_col=0)

# Top column is misaligned.
players.index.name = 'ID'
players.columns = ['First' , 'Last', 'Handidness', 'DOB', 'Country']

# Parse date data to dates.
players = players.assign(DOB=pd.to_datetime(players['DOB'], format='%Y%m%d'))

# Handidness is reported as U if unknown; set np.nan instead.
import numpy as np
players = players.assign(Handidness=players['Handidness'].replace('U', np.nan))

players.head()


# <b>matches.csv</b>

# In[ ]:



matches = pd.read_csv('../input/player/matches.csv', encoding='latin1', index_col=0)
matches.head(5)


# <b>rankings.csv</b>
# 

# In[ ]:


import pandas as pd
rankings = pd.read_csv('../input/player/rankings.csv', encoding='latin1', index_col=0)
rankings.head(5)


# <b>importing libraries : </b>

# In[ ]:


#Data manipulation libraries : 
import numpy as np  #numpy
import pandas as pd  #pandas

#System libraries
import glob #The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell



#Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#math operations lib 
import math
from math import pi

#date manipulation 
import datetime as dt

#Impute missing data
from sklearn.preprocessing import Imputer 



#Splitting data to test and train 
from sklearn.model_selection import train_test_split

import datetime

import os


# <b>let's see the relation between countrys and players : </b>

# In[ ]:


#countrys with moste nomber of players 
players.Country.value_counts().head(20).plot.bar(
    figsize=(12, 6),
    title='WTA Player Country Representing'
)


# In[ ]:



matches['winner_ioc'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA country with Most Matches Win'
)


# In[ ]:


#WTA country with Most Matches lost

matches['loser_ioc'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA country with Most Matches lost'
)


# <b>let's see the players with most win/lost matches : </b>

# In[ ]:


pd.concat([matches['winner_name'], matches['loser_name']]).value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players with Most Matches Played'
)


# In[ ]:


matches['winner_name'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players with Most Matches Won'
)


# <li> 5  players from RUS : kusnetsova , sharapova , dementieva , petrova , zvonareva </li>
# <li> 2  players from USA  : serena and venus  </li>
# <li> facing a player from RUS / USA could be a probleme </li>
# <li>let's see the players  with most matches lost </li>

# In[ ]:


matches['loser_name'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players with Most Matches Lost'
)


# can we say that they are the the weakest players ? 
# <li>grand slam winners : 5 </li>
# <li>grand slam finalist : 5 </li>
# <li>world number 1  : 1 </li>

# we need more informations like serve , aces , winners ... so ! let's add some extra data !!! 
# 

# In[ ]:


import pandas as pd

t11 = pd.read_csv('../input/player/wta_matches_2011.csv', encoding='latin1', index_col=0)
t12 = pd.read_csv('../input/player/wta_matches_2012.csv', encoding='latin1', index_col=0)
t13 = pd.read_csv('../input/player/wta_matches_2013.csv', encoding='latin1', index_col=0)
t14 = pd.read_csv('../input/player/wta_matches_2014.csv', encoding='latin1', index_col=0)
t15 = pd.read_csv('../input/player/wta_matches_2015.csv', encoding='latin1', index_col=0)
t16 = pd.read_csv('../input/player/wta_matches_2016.csv', encoding='latin1', index_col=0)

wta = pd.concat([t11, t12,t13,t14,t15,t16])
wta.head(5)


# <h2><b> 3. DATA PREPARATION </b></h2>
# 

# <h3> 3.1. duplicated raws </h3>

# <b>wta and players</b> : no duplicated raws 

# In[ ]:


wta.duplicated().sum()


# In[ ]:


players.duplicated().sum()


# some duplicated rows in <b>ranking</b>

# In[ ]:


rankings.shape


# In[ ]:


rankings.duplicated().sum()


# In[ ]:


rankings.drop_duplicates(keep='first').shape


# <h3> 3.2. missing values </h3>

# <b>wta : </b>

# In[ ]:


wta.isnull().sum(axis=0)


# In[ ]:


wta.shape


# In[ ]:



wta['surface'].value_counts(dropna=False)


# In[ ]:


wta['surface'].fillna(value='Hard', inplace=True)


# In[ ]:


wta['winner_hand'].fillna(value='R', inplace=True)


# In[ ]:


wta['loser_hand'].fillna(value='R', inplace=True)


# In[ ]:


wta['loser_entry'].value_counts(dropna=False)


# In[ ]:


wta['loser_entry'].fillna(value='S', inplace=True)


# In[ ]:


wta['winner_entry'].fillna(value='S', inplace=True)


# In[ ]:


wta = wta.drop('l_SvGms', 1)


# In[ ]:


wta = wta.drop('w_SvGms', 1)


# In[ ]:


wta.isnull().sum(axis=0)


# In[ ]:


wta['l_bpFaced'].fillna((wta['l_bpFaced'].mean()), inplace=True)


# In[ ]:


wta['l_bpSaved'].fillna((wta['l_bpSaved'].mean()), inplace=True)


# In[ ]:


wta['l_2ndWon'].fillna((wta['l_2ndWon'].mean()), inplace=True)


# In[ ]:


wta['l_1stWon'].fillna((wta['l_1stWon'].mean()), inplace=True)


# In[ ]:


wta['l_1stIn'].fillna((wta['l_1stIn'].mean()), inplace=True)


# In[ ]:


wta['l_svpt'].fillna((wta['l_svpt'].mean()), inplace=True)


# In[ ]:


wta['l_df'].fillna((wta['l_df'].mean()), inplace=True)


# In[ ]:


wta['l_ace'].fillna((wta['l_ace'].mean()), inplace=True)


# In[ ]:


wta['w_bpFaced'].fillna((wta['w_bpFaced'].mean()), inplace=True)


# In[ ]:


wta['w_bpSaved'].fillna((wta['w_bpSaved'].mean()), inplace=True)


# In[ ]:


wta['w_ace'].fillna((wta['w_ace'].mean()), inplace=True)


# In[ ]:


wta['w_df'].fillna((wta['w_df'].mean()), inplace=True)


# In[ ]:


wta['w_1stIn'].fillna((wta['w_1stIn'].mean()), inplace=True)


# In[ ]:


wta['w_svpt'].fillna((wta['w_svpt'].mean()), inplace=True)


# In[ ]:


wta['w_1stWon'].fillna((wta['w_1stWon'].mean()), inplace=True)


# In[ ]:


wta['w_2ndWon'].fillna((wta['w_2ndWon'].mean()), inplace=True)


# <b>matches: </b>

# In[ ]:


matches.isnull().sum(axis=0)


# In[ ]:


matches.shape


# In[ ]:


matches['loser_rank'].fillna(value='40.0', inplace=True)


# In[ ]:


matches['year'].value_counts(dropna=False)


# In[ ]:


matches['year'].fillna(value='2002.0', inplace=True)


# In[ ]:


matches['round'].value_counts(dropna=False)


# In[ ]:


matches['round'].fillna(value='R32', inplace=True)


# In[ ]:


matches['surface'].fillna(value='Hard', inplace=True)


# In[ ]:


matches['winner_hand'].fillna(value='R', inplace=True)


# In[ ]:


matches['loser_hand'].fillna(value='R', inplace=True)


# In[ ]:


matches['loser_rank_points'].value_counts(dropna=False)


# In[ ]:


matches['loser_rank_points'].fillna(value='400.0', inplace=True)


# In[ ]:


matches['minutes'].fillna((matches['minutes'].mean()), inplace=True)


# In[ ]:


matches['winner_ht'].value_counts(dropna=False)


# In[ ]:


matches['winner_ht'].fillna(value='170.0', inplace=True)


# In[ ]:


matches['winner_entry'].fillna(value='S', inplace=True)


# In[ ]:



matches = matches.drop('Unnamed: 32', 1)


# <h3><b>3.3 column preparation : </b></h3>

# In[ ]:


winners = list(np.unique(wta.winner_name))
losers = list(np.unique(wta.loser_name))

all_players = winners + losers
players = np.unique(all_players)

players_wta = pd.DataFrame()
players_wta['Name'] = players
players_wta['Wins'] = players_wta.Name.apply(lambda x: len(wta[wta.winner_name == x]))
players_wta['Losses'] = players_wta.Name.apply(lambda x: len(wta[wta.loser_name == x]))

players_wta['PCT'] = np.true_divide(players_wta.Wins,players_wta.Wins + players_wta.Losses)
players_wta['Games'] = players_wta.Wins + players_wta.Losses
#%%
plt.style.use('fivethirtyeight')
wta['Year'] = wta.tourney_date.apply(lambda x: str(x)[0:4])
wta['Sets'] = wta.score.apply(lambda x: x.count('-'))
wta['Rank_Diff'] =  wta['loser_rank'] - wta['winner_rank']
wta['ind'] = range(len(wta))
wta['Rank_Diff_Round'] = wta.Rank_Diff.apply(lambda x: 10*round(np.true_divide(x,10)))
wta = wta.set_index('ind')

surfaces = ['Hard','Grass','Clay','Carpet']
for surface in surfaces:
    players_wta[surface + '_wins'] = players_wta.Name.apply(lambda x: len(wta[(wta.winner_name == x) & (wta.surface == surface)]))
    players_wta[surface + '_losses'] = players_wta.Name.apply(lambda x: len(wta[(wta.loser_name == x) & (wta.surface == surface)]))
    players_wta[surface + 'PCT'] = np.true_divide(players_wta[surface + '_wins'],players_wta[surface + '_losses'] + players_wta[surface + '_wins'])
    
serious_players = players_wta[players_wta.Games>40]
serious_players['Height'] = serious_players.Name.apply(lambda x: list(wta.winner_ht[wta.winner_name == x])[0])
serious_players['Best_Rank'] = serious_players.Name.apply(lambda x: min(wta.winner_rank[wta.winner_name == x]))
serious_players['Win_Aces'] = serious_players.Name.apply(lambda x: np.mean(wta.w_ace[wta.winner_name == x]))
serious_players['Lose_Aces'] = serious_players.Name.apply(lambda x: np.mean(wta.l_ace[wta.loser_name == x]))
serious_players['Aces'] = (serious_players['Win_Aces']*serious_players['Wins'] + serious_players['Lose_Aces']*serious_players['Losses'])/serious_players['Games']


# <h1><b>4.Data Exploration </b></h1>

# <h2><b>4.1 Surfaces :</b> </h2>

# In[ ]:


wta.surface.value_counts(normalize=True).plot(kind='bar')


# as we said before , most of the matches are on hard courts ! let's see the aces per surface 

# In[ ]:


wta['Aces'] = wta.l_ace + wta.w_ace

plt.bar(1,np.mean(wta.Aces[wta.surface == 'Hard']))
plt.bar(2,np.mean(wta.Aces[wta.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(wta.Aces[wta.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(wta.Aces[wta.surface == 'Carpet']), color ='y')
plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('More Aces on Grass')


# In[ ]:


wta['df'] = wta.l_df + wta.w_df

plt.bar(1,np.mean(wta.df[wta.surface == 'Hard']))
plt.bar(2,np.mean(wta.df[wta.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(wta.df[wta.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(wta.df[wta.surface == 'Carpet']), color ='y')
plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('More double faults on  Hard ')


# In[ ]:


wta['bps'] = wta.l_bpFaced + wta.w_bpFaced

plt.bar(1,np.mean(wta.bps[wta.surface == 'Hard']))
plt.bar(2,np.mean(wta.bps[wta.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(wta.bps[wta.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(wta.bps[wta.surface == 'Carpet']), color ='y')
plt.ylabel('break point saved  per surface')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('easier to break serve on Clay ')


# In[ ]:


plt.bar(1,np.mean(matches.minutes[matches.surface == 'Hard']))
plt.bar(2,np.mean(matches.minutes[matches.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(matches.minutes[matches.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(matches.minutes[matches.surface == 'Carpet']), color ='y')
plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('less time  on Grass')


# you need more time to win a match on clay courts ! 

# In[ ]:


print('Average time on HARD courts ', np.mean(matches.minutes[matches.surface == 'Hard']))


# In[ ]:


print('Average time on Clay courts ', np.mean(matches.minutes[matches.surface == 'Clay']))


# In[ ]:


print('Average time on Grass courts ', np.mean(matches.minutes[matches.surface == 'Grass']))


# <h3><b>Conlusion : </b></h3>

# <li> if you have a good serve probably you are a good grass courts player  </li>
# <li>it's not easy to serve on clay courts </li>
# <li>matches are longer on clay and hard courts , so you have to be more patient</li>

# <h2><b>4.2 Height</b></h2>

# In[ ]:


avg_height = []
years = np.arange(2011,2016)
for year in years:
    avg_winner = np.mean(wta.winner_ht[wta.Year == str(year)])
    avg_loser = np.mean(wta.winner_ht[wta.Year == str(year)])
    avg_height.append(np.mean([avg_winner,avg_loser]))

plt.bar(years,avg_height)
plt.ylim([165,175])
plt.xlabel('Year')
plt.ylabel('Average Height')
plt.title('Are tennis players getting taller?')


# In[ ]:


pd.concat([wta['winner_ht'], wta['loser_ht']]).value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players height '
)


# <b>most of the players are tall !   </b>

# In[ ]:


wta['loser_ht'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA hieght with Most Matches Lost'
)


# In[ ]:


wta['winner_ht'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players height with Most Matches win'
)


# let's try to combine hieghts with surfaces !! 

# In[ ]:


print('Average winner height on HARD courts ', np.mean(wta.winner_ht[wta.surface == 'Hard']))
print('Average winner height on CLAY courts ', np.mean(wta.winner_ht[wta.surface == 'Clay']))
print('Average winner height on GRASS courts ', np.mean(wta.winner_ht[wta.surface == 'Grass']))
print('Average winner height on CARPET courts ', np.mean(wta.winner_ht[wta.surface == 'Carpet']))


# let's see <b>Average Player Heigh</b> and <b>Average Height of no.1 Rank</b>

# In[ ]:


print('Average Player Height', np.mean(serious_players.Height))
print('Average Height of no.1 Rank ',np.mean(serious_players.Height[serious_players.Best_Rank == 1]))


# <b>Informations  : </b><br>
#     <li>the tallest w.no1 are dinara safina and lindsy davenport : 1.89</li>
#     <li>the smallest w.no1 is tracy austin : 1.65 </li>
#     <li> the current hieght oh w.no1 (simona halep ) : 1.68</li><br>
# <h3><b>Conclusion : </b> </h3>
#         <li> player's height don't give a clear vision   </li>

# <h2> 4.3 Age : </h2>

# In[ ]:


import numpy as np

players.set_index('DOB').resample('Y').count().Country.plot.line(
    linewidth=1, 
    figsize=(12, 4),
    title='WTA Player Year of Birth'
)


# <b>tennis is getting more popular !</b>

# In[ ]:


wta.winner_age.plot(kind='hist')


# In[ ]:


matchesWon = matches[(matches['round'] == 'F') & 
(matches['tourney_name'] =='US Open')| \
(matches['tourney_name'] =='French Open')|(matches['tourney_name'] =='Wimbledon')| \
                     (matches['tourney_name'] =='Australian Open')]
matchesWon.winner_age.plot(kind='hist',color='gold',label='Age',bins = 150                            ,linewidth=0.01,grid=True,figsize = (12,10))
plt.xlabel('Age')              # label = name of label
plt.ylabel('No of Wins')
plt.suptitle('Number of Grandslams wins based on Age of the player', x=0.5, y=.9, ha='center', fontsize='xx-large')


# In[ ]:


df70 = wta[(wta.loser_age - wta.winner_age > 0)]
df71 = wta[(wta.loser_age - wta.winner_age < 0)]
labels = 'the winner is the youngest', 'the winner is the oldest'
values = [6589, 7419]
colors = ['red', 'pink']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()


# In[ ]:


plt.bar(1,np.mean(matches.winner_age[matches['round'] == 'R128']))
plt.bar(2,np.mean(matches.winner_age[matches['round'] == 'R64']), color = 'g')
plt.bar(3,np.mean(matches.winner_age[matches['round'] == 'QF']), color ='r')
plt.bar(4,np.mean(matches.winner_age[matches['round'] == 'F']), color ='y')


plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['R128','R64','qF','F'])
plt.title('age / rounds ')


# <h3><b> Conclusion : </b></h3><br>
# <li>tennis is becoming more popular </li>
# <li>the perfect age of player's carrer is between 22 and 27 </li>
# <li>the ideal age to announce your retirement is between 30 and 35 </li>
# <li>in the final rounds the youngest player could have more chances to win the match and the tournament </li>

# <h2><b>4.4 upsets : </h2></b>

# In[ ]:


plt.style.use('fivethirtyeight')

(matches
     .assign(
         winner_seed = matches.winner_seed.fillna(0).map(lambda v: v if str.isdecimal(str(v)) else np.nan),
         loser_seed = matches.loser_seed.fillna(0).map(lambda v: v if str.isdecimal(str(v)) else np.nan)
     )
     .loc[:, ['winner_seed', 'loser_seed']]
     .pipe(lambda df: df.winner_seed.astype(float) >= df.loser_seed.astype(float))
     .value_counts()
).plot.bar(title='Higher Ranked Seed Won Match')


# <b>choking information !!! </b>

# #A look at Comebacks!! <br>
# Overall Comeback Percentage. Ignoring all cases where,<br>
# 
# one of the players had retired in the second/third set (characterised by 'RET' in the data - and performing by checking for 'R'
# the game finished in the second set, characterised by 'nan' in the third set score.<br>
# third set had 'unfinished'<br>
# third set had 'DEF'<br>
# Using re.sub to replace scores that went to tie-break with a '(space)' eg: 6-7(3)

# In[ ]:


import re
wta['Set_1'], wta['Set_2'], wta['Set_3'] = wta['score'].str.split(' ',2).str
comeback = 0
for item,row in wta.iterrows():
	if 'R' not in str(row['Set_2']):
		if 'R' not in str(row['Set_3']) and str(row['Set_3']) != 'nan' and 'u' not in str(row['Set_3']) and str(row['Set_3']) != '6-0 6-1' and 'D' not in str(row['Set_3']):
			set_score_Set_2 = re.sub("\(\d+\)"," ",row['Set_2'])
			set_score_Set_3 = re.sub("\(\d+\)"," ",row['Set_3'])
			Set_3 = float(set_score_Set_3.split('-')[0]) - float(set_score_Set_3.split('-')[1])
			Set_2 = float(set_score_Set_2.split('-')[0]) - float(set_score_Set_2.split('-')[1])
			if Set_3 * Set_2 > 0:
				comeback += 1

print ('Comeback %% = %f'%(100*float(comeback)/float(len(wta))))


# <h2><b> 4.5 injuries : </b></h2>

# In[ ]:


plt.bar(1,np.sum([(wta.surface == 'Hard') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(2,np.sum([(wta.surface == 'Clay') & ( 'RET' in str(row['score'])  ) ] ), color = 'r')
plt.bar(3,np.sum([(wta.surface == 'Grass') & ( 'RET' in str(row['score'])  ) ] ), color ='g')
plt.bar(4,np.sum([(wta.surface == 'Carpet') & ( 'RET' in str(row['score'])  ) ] ), color ='y')


plt.ylabel('retirement per surface')
plt.xticks([1,2,3,4], ['Hard','Clay','Grass','Carpet'])
plt.title('retirement / surface ')


# In[ ]:


plt.bar(1,np.sum([(wta['round'] == 'R128') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(2,np.sum([(wta['round'] == 'R64') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(3,np.sum([(wta['round'] == 'R32') & ( 'RET' in str(row['score'])  ) ] ), color = 'r')
plt.bar(4,np.sum([(wta['round'] == 'R16') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(5,np.sum([(wta['round'] == 'QF') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(6,np.sum([(wta['round'] == 'SF') & ( 'RET' in str(row['score'])  ) ] ), color ='g')
plt.bar(7,np.sum([(wta['round'] == 'F') & ( 'RET' in str(row['score'])  ) ] ), color ='y')


plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5,6], ['R128','R64','R32','R16','QF','SF','F'])
plt.title('retirement / round ')


# In[ ]:


plt.bar(1,np.sum([(wta['loser_rank'] < 30 ) & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(2,np.sum([(wta['loser_rank'] >30   ) & (wta['loser_rank'] <50 ) & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(3,np.sum([(wta['loser_rank'] >50   ) & (wta['loser_rank'] <100 ) & ( 'RET' in str(row['score'])  ) ] ))

plt.bar(4,np.sum([(wta['loser_rank'] > 100 ) & ( 'RET' in str(row['score'])  ) ] ))


plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4], ['<30','[30 .. 50]','[50 .. 100]','>100'])
plt.title('retirement / rank ')


# <h3><b>Conclusion : </b></h3>
#     <li>high number of injuries in clay courts compared to the number of clay court's tournaments </li>
#     <li>in early stages of the draw chances of getting injured are higher than final stages </li>

# <h2>4.6 Seeds : </2>

# let's see teh performance of S,WC,Q,LL and ALT

# In[ ]:


plt.bar(1,np.sum (wta['winner_entry'] == 'S'  )  )  
plt.bar(2,np.sum (wta['winner_entry'] == 'Q'  )  ) 
plt.bar(3,np.sum (wta['winner_entry'] == 'WC'  )  )
plt.bar(4,np.sum (wta['winner_entry'] == 'LL'  )  ) 
plt.bar(5,np.sum (wta['winner_entry'] == 'ALT'  )  ) 



plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5], ['S','Q','WC','LL','ALT'])
plt.title('S , Q , WC and LL matche win ')


# let's see the performance of S , Q , WC and LL in finals !

# In[ ]:


plt.bar(1,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'S'  )  ) ] ))
plt.bar(2,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'Q'  )  ) ] ))
plt.bar(3,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'WC'  )  ) ] ))
plt.bar(4,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'LL'  )  ) ] ))
plt.bar(5,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'ALT'  )  ) ] ))



plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5], ['S','Q','WC','LL','ALT'])
plt.title('S , Q , WC and LL in finals ')


# let's see their performance in GS !!!

# In[ ]:


plt.bar(1,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'S'  )  ) ] ))
plt.bar(2,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'Q'  )  ) ] ))
plt.bar(3,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'WC'  )  ) ] ))
plt.bar(4,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'LL'  )  ) ] ))
plt.bar(5,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'ALT'  )  ) ] ))



plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5], ['S','Q','WC','LL','ALT'])
plt.title('S , Q , WC and LL in GS ')


# <h3>Conclusion : </h3><br>
#     <li>lucky losers and Alternatives don't have any impact </li>
#         <li>qualifieres and WC  can  makes some trouble </li>

# <h1>5.Conclusion </h1>

# This is a fantastic dataset with a lot of exploratory potential. Hopefully this notebook has given you some ideas of further exploration you can do with it!<br>
# this work some deterministic factors to own the match , some informations that can be usefull for players and some facts .
# 

# <h1>6.Perspectives</h1><br>
# <li>Does the rank correlates with the money earn by the player? </li>
# <li>Which player did the most rapidly climb the ranks through the years? </li>
# <li>is it the same results with ATP ? </li>
# 
