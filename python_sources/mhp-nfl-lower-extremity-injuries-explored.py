#!/usr/bin/env python
# coding: utf-8

# By: Nick Tarsi and Brandon Snyder

# # Problem Definition
# 
# > Your challenge is to characterize any differences in player movement between the playing surfaces and identify specific scenarios (e.g., field surface, weather, position, play type, etc.) that interact with player movement to present an elevated risk of injury. The NFL is challenging Kagglers to help them examine the effects that playing on synthetic turf versus natural turf can have on player movements and the factors that may contribute to lower extremity injuries.
# 
# ### Hypotheses:
# * There is a significant increase in injury occurence for plays on synthetic turf vs. plays on natural turf.
# * However, due to the extremely low proportion of plays with injuries overall, turf type alone will barely explain any of the variation in injury occurence. 
# * While no other individual play-level factor will explain much more of the variation in injury occurence, those that do will have significant interaction effects with turf type, suggesting that synthetic turf's elevates injury risk through amplifying other risk factors.
# 
# ### Plan:
# 1. Import playtrack data (with some creative workarounds to converse memory), calculate additional columns representing change between observations (acceleration, rotation, etc.)
# 1. Group playtrack data into play-level aggregate features, as well as features representing the end of movement during the play (assuming players do not continue tracked movement after experiencing an injury)
# 1. Join play-level aggregate features to player-level data (roster role) and game-level data (turf type, stadium type, weather, play role, play role group)
# 1. Statistical significance tests for differences in injury occurence:
#     * Split play-level records by turf type and conduct two-sample t-test
#     * Further split play-level records by other categorical/binary features and conduct multiple logistic regressions to identify significant predictors among categorical features, continuous features, and interaction effects.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plot
import datetime

from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
import statistics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

pd.options.display.float_format = '{:.3f}'.format #prevent scientific notation in dataframes, display #.### instead

get_ipython().run_line_magic('whos', '#outputs table of variables and their info')

## used to expand Jupyter Notebook to full browser width for easier reading of long lines
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

## template for monitoring/recording run-times for blocks of code
overall_startDT = datetime.datetime.now()
print('Started at', overall_startDT)
#code
print(datetime.datetime.now() - overall_startDT)
# usually around 0:##:##.# on my laptop, ##m##s on Kaggle


# ## Import playtrack data with some initial pre-processing

# In[ ]:


## import playtrack data
startDT = datetime.datetime.now()
print('Starting import of playtrackDF at', startDT)
#original key without padding
col_dtypes = {'time':float, 
              'x':float, 'y':float, 
              'dir':float, 'dis':float, 'o':float, 's':float}
use_cols = ['PlayKey', 'time', 'dir', 'dis', 'o', 's']
#playtrackDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv', dtype=col_dtypes, usecols=use_cols)
playtrackDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv', dtype=col_dtypes)


## there are 2 rows with NaN values for 'dir' and 'o', they appear to be meaningless glitches and deletable
#print(playtrackDF.columns)
#print(playtrackDF.isna().sum())
#dropableRows = list(playtrackDF[playtrackDF['dir'].isna()].index)
#for row in dropableRows:
#    display(playtrackDF[row-2:row+3])
playtrackDF.dropna(axis='index', subset=['dir', 'o'], inplace=True)

print(playtrackDF.shape)
print(datetime.datetime.now() - startDT, 'to import playtrack data and drop NA rows')
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs for initial import')
#usually around 1m45s on Kaggle


# Because this dataframe utilizes a large amount of memory and Kaggle's cloud session restarts if it runs out of memory, some workarounds are implemented to use smaller datatypes to store the values. Specifically, floats (i.e. decimals) that are only precise to the hundredth's place (#.##) are multipied by 100 and then converted to integers. That mutliplication will be reversed once playtrack data has been grouped to play-level data, which requires a much smaller record count and therefore can contain floats without utilizing too much memory. (https://docs.scipy.org/doc/numpy/user/basics.types.html)
# 
# Additionally, the "PlayKey" valyes are reformatted so they can be sorted properly (01,02...11,12,etc. instead of 1,11,12,2,etc.) and also stored as numbers rather than strings.

# In[ ]:


## reducing memory usage of this massive dataframe by converting columns to *100 ints
startDT = datetime.datetime.now()
for col in playtrackDF.columns:
    if playtrackDF[col].dtype != object:
        print(col, playtrackDF[col].min(), playtrackDF[col].max())
        playtrackDF[col] = playtrackDF[col]*100
        if playtrackDF[col].min() < 0:
            playtrackDF[col] = playtrackDF[col].astype(np.int16) #stores -32,768 to 32,767
        else:
            playtrackDF[col] = playtrackDF[col].astype(np.uint16) #stores 0-65,535
print(datetime.datetime.now() - startDT, 'to reduce memory usage') ## usually around 15s on Kaggle
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs for *100 ints')

## reformatting the PlayKey from 12345-1-1 to 1234512123 to enable storage as number with proper sorting
startDT = datetime.datetime.now()
print('Starting rekeying at', startDT)
playtrackDF.loc[:, 'PlayKey'] = playtrackDF['PlayKey'].apply(lambda v: '{0:0>5}{1:0>2}{2:0>3}'.format(*v.split('-') ) )
playtrackDF.loc[:, 'PlayKey'] = playtrackDF['PlayKey'].astype(np.int64) #uint32's max value is just below the largest key
print(datetime.datetime.now() - startDT, 'to rekey DF') ## usually around 2m on Kaggle
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs for reduced/rekeyed')


# # import injury and plays data
# 
# The same rekeying from the playtrack data is applied to the "PlayKey", "GameID", and "PlayerKey" columns in these dataframes.

# In[ ]:


#import injuries and plays data
injuriesDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
print("injuriesDF shape", injuriesDF.shape)
playsDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')
print("playsDF shape", playsDF.shape)

startDT = datetime.datetime.now()
injuriesDF['PlayerKey'] = injuriesDF['PlayerKey'].apply(lambda v: '{0:0>5}'.format(v) )
injuriesDF['GameID'] = injuriesDF['GameID'].apply(lambda v: '{0:0>5}{1:0>2}'.format(*v.split('-') ) )
injuriesDF['PlayKey'].fillna(value='0-0-0', inplace=True)
injuriesDF['PlayKey'] = injuriesDF['PlayKey'].astype(str).apply(lambda v: '{0:0>5}{1:0>2}{2:0>3}'.format(*v.split('-') ) )
playsDF['PlayerKey'] = playsDF['PlayerKey'].apply(lambda v: '{0:0>5}'.format(v) )
playsDF['GameID'] = playsDF['GameID'].apply(lambda v: '{0:0>5}{1:0>2}'.format(*v.split('-') ) )
playsDF['PlayKey'] = playsDF['PlayKey'].apply(lambda v: '{0:0>5}{1:0>2}{2:0>3}'.format(*v.split('-') ) )
injuriesDF.loc[:, 'PlayKey'] = injuriesDF['PlayKey'].astype(np.int64)
playsDF.loc[:, 'PlayKey'] = playsDF['PlayKey'].astype(np.int64)
print(datetime.datetime.now() - startDT, 'to rekey smaller dataframes') # usually around 1s on Kaggle

print('Dataframes created from injuries and plays CSVs (playtrack CSV to be handled separately due to size)')


# In[ ]:


#collapse player-level info into smaller DF
playersDF = playsDF.groupby(by=['PlayerKey', 'RosterPosition']).size().reset_index().rename(columns={0:'Plays'})
print("playersDF shape", playersDF.shape)

#collapse game-level info into smaller DF
groupCols = [playsDF['PlayerKey'], playsDF['GameID'], playsDF['StadiumType'].fillna('unknown'), playsDF['FieldType'], playsDF['Temperature'], playsDF['Weather'].fillna('unknown'), playsDF['PlayerDay']]
gamesDF = playsDF.groupby(by=groupCols).size().reset_index().rename(columns={0:'Plays'})
print("gamesDF shape", gamesDF.shape)
gamesDF = gamesDF.sort_values(by=['PlayerKey', 'PlayerDay'])
gamesDF.reset_index(drop=True, inplace=True)

print(len(injuriesDF['PlayerKey'].unique()), 'unique players in injuriesDF')

display(gamesDF.head())


# Some data cleaning is needed for the "StadiumType", "Weather", and "Temperature" columns. 
# 
# Given the focus on the turf, new columns are created that contain simpler categorical variables representing whether the field is exposed and/or wet, rather than overwrite the original values for "StadiumType" and "Weather". 
# 
# Some "Temperature" values needed to be imputed as well, specifically for indoor games that originally were reported to have a temperature of -999. These invalid values will be replaced by the average temperature of other indoor games.

# In[ ]:


injuryGames = pd.DataFrame(injuriesDF.groupby(by=['GameID']).size())
injuryGames.reset_index(inplace=True)
gamesDF['InjOcc'] = 0
for idVal in injuryGames['GameID']:
    gamesDF.loc[gamesDF['GameID'] == idVal, 'InjOcc'] = 1
## there's 104 unique GameIDs for injured players and 5712 unique GameIDs in the play data (gamesDF)

## cleaning StadiumType and determining FieldExposed
#print(gamesDF['StadiumType'].unique())
outdoorList = ['Open', 'Outdoor', 'Oudoor', 'Outdoors', 'Ourdoor', 'Outddors', 'Heinz Field', 'Outdor', 'Outside']
indoorList = ['Indoors', 'Closed Dome', 'Domed, closed', 'Dome', 'Indoor', 'Domed', 'Retr. Roof-Closed', 'Outdoor Retr Roof-Open', 'Indoor, Roof Closed', 'Retr. Roof - Closed', 'Retr. Roof-Open', 'Dome, closed', 'Indoor, Open Roof', 'Domed, Open', 'Domed, open', 'Retr. Roof - Open', 'Retr. Roof Closed', 'Retractable Roof']
unknownList = ['unknown', 'Bowl', 'Cloudy']
gamesDF.loc[gamesDF['StadiumType'].isin(outdoorList), 'FieldExposed'] = 1
gamesDF.loc[gamesDF['StadiumType'].isin(indoorList), 'FieldExposed'] = 0
gamesDF.loc[gamesDF['StadiumType'].isin(unknownList), 'FieldExposed'] = 1 ## assuming exposed b/c not specified
print('FieldExposed vals:', gamesDF['FieldExposed'].unique())

## cleaning Weather and determining FieldWet
wetDescs = []
dryDescs = []
unkDescs = []
for desc in list(gamesDF['Weather'].unique()):
    if 'rain' in desc or 'Rain' in desc or 'showers' in desc or 'Showers' in desc or 'snow' in desc or 'Snow' in desc:
        wetDescs.append(desc)
    else: 
        if 'unknown' in desc or 'Unknown' in desc:
            unkDescs.append(desc)
        else:
            dryDescs.append(desc)
gamesDF.loc[(gamesDF['Weather'].isin(wetDescs)) & (gamesDF['FieldExposed'] == 1), 'FieldWet'] = 1
gamesDF.loc[gamesDF['Weather'].isin(dryDescs), 'FieldWet'] = 0
gamesDF.loc[gamesDF['FieldExposed'] == 0, 'FieldWet'] = 0
gamesDF.loc[gamesDF['Weather'].isin(unkDescs), 'FieldWet'] = 0 ## assuming field is not wet b/c not specified
print('FieldWet vals:', gamesDF['FieldWet'].unique())

## cleaning Temperature
avgIndoorTemp = gamesDF.loc[(gamesDF['FieldExposed'] != 1) & (gamesDF['Temperature'] != -999), 'Temperature'].mean()
#print('Avg (valid) temp for indoor games is', round(avgIndoorTemp, 0))
gamesDF.loc[(gamesDF['FieldExposed'] != 1) & (gamesDF['Temperature'] == -999), 'Temperature'] = round(avgIndoorTemp, 0)

## making a one-hot variable for turf type
print('FieldType vals:', gamesDF['FieldType'].unique())
gamesDF['Synthetic'] = 1 * (gamesDF['FieldType'] == 'Synthetic')

display(gamesDF.head())


# # further pre-processing of ALL playtrackDF rows

# In[ ]:


print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs before additional columns')
startDT = datetime.datetime.now()
playtrackDF.loc[:, 'twist'] = abs(playtrackDF['dir'].astype(np.int32) - playtrackDF['o'].astype(np.int32))
playtrackDF.loc[playtrackDF['twist'] > 18000, 'twist'] = 36000 - playtrackDF.loc[playtrackDF['twist'] > 18000, 'twist']
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs with twist col added')
playtrackDF.loc[:, 'twist'] = playtrackDF['twist'].astype(np.int16)
print(datetime.datetime.now() - startDT, 'to calculate twist column') ## usually around 5s on Kaggle
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs with twist col as int16')

cols_for_diffs = ['dir', 'o', 's', 'twist']

for col in cols_for_diffs:
    startDT = datetime.datetime.now()
    playtrackDF.loc[:, 'd_'+col ] = abs(playtrackDF[col].astype(np.int32).diff())
    playtrackDF.loc[0, 'd_'+col ] = 0 ## these rows represent a new play and shouldn't be compared to the row above
    playtrackDF.loc[:, 'd_'+col] = playtrackDF['d_'+col].astype(np.int32)
    print(datetime.datetime.now() - startDT, 'to calculate', 'd_'+col, 'column') ## each around 2-5s on Kaggle
    
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs currently')

display(playtrackDF.head())


# In[ ]:


for col in playtrackDF.columns:
    if playtrackDF[col].dtype != object:
        print(col, playtrackDF[col].dtype, round(playtrackDF[col].memory_usage() / 1024**3, 3), 'GBs', playtrackDF[col].min(), playtrackDF[col].max() )
    else:
        print(col, playtrackDF[col].dtype, round(playtrackDF[col].memory_usage() / 1024**3, 3), 'GBs' )


# Memory utilization is starting to creep up again with the additional columns added, but the playtrack data is going to be grouped into play-level records now, so this huge dataframe won't need to be processed much longer and doesn't threaten to overwhelm Kaggle's memory limit.
# 
# ## Calculating play-level features and joining to player-level and game-level features

# In[ ]:


startDT = datetime.datetime.now()

playLens = pd.DataFrame(playtrackDF[['PlayKey']].groupby(by=['PlayKey']).size())
playLens.columns = ['obs']

## any stats here are divided by 100 to account for earlier *100 for sake of reduced memory via int datatypes 

## identifying the field "coordinates" for the player's final location per play
playFinalLocs = playtrackDF[['PlayKey', 'x', 'y']].groupby(by=['PlayKey']).tail(1).set_index('PlayKey')/100
## assuming each quadrant of the field is interchangeable, the location values should be 
## relative to midfield rather than a corner of the field
playFinalLocs.loc[:, 'x'] = abs(playFinalLocs['x'] - (120/2)) # converting to yds away from midfield
playFinalLocs.loc[:, 'y'] = abs(playFinalLocs['y'] - (53.3/2)) # converting to yds away from midfield
playFinalLocs.columns = 'rel_' + playFinalLocs.columns + '_final'

## aggregate between-observation changes per play, first as sums...
playSums = playtrackDF[['PlayKey','d_dir', 'd_o', 'd_s', 'd_twist']].groupby(by=['PlayKey']).sum()/100
playSums.columns = playSums.columns + '_sum'

## ... then as means
playAvgs = playtrackDF[['PlayKey','d_dir', 'd_o', 'd_s', 'd_twist']].groupby(by=['PlayKey']).mean()/100
playAvgs.columns = playAvgs.columns + '_avg'

## join all features together
playStats = playsDF[['PlayKey', 'FieldType']].drop_duplicates() ## assumed to be consistent for all plays with same gameID prefix in PlayKey
playStats = playStats.merge(playFinalLocs, on='PlayKey')
playStats = playStats.merge(playLens, on='PlayKey')
playStats = playStats.merge(playSums, on='PlayKey')
playStats = playStats.merge(playAvgs, on='PlayKey')

print(datetime.datetime.now() - startDT, 'to generate playStats') ## about 20s on Kaggle

display(playStats.head(2))


# In[ ]:


## just verifying no weird values resulted from the calculations above
for col in playStats.columns:
    if playStats[col].dtype != 'object':
        print(col, '\t', playStats[col].dtype, '\t', round(playStats[col].min(),3), '\t', round(playStats[col].max(), 3) )


# In[ ]:


# identifying injury plays
print(playStats.shape)
startDT = datetime.datetime.now()
injPlays = list(injuriesDF['PlayKey'].unique())
playStats['Inj'] = 0
playStats.loc[playStats['PlayKey'].isin(injPlays), 'Inj'] = 1
print(datetime.datetime.now() - startDT, 'to identify Inj plays') ## about 0.05s on Kaggle
print(playStats.shape)

display(playStats[playStats['Inj'] == 1].head())


# ## Initial simple analysis

# In[ ]:


display(playStats.groupby(['Inj', 'FieldType']).size().round(3))


# In[ ]:


display(playStats.groupby(['Inj', 'FieldType']).mean().round(3))


# WITHOUT FILTERING PRE- AND POST-PLAY OBSERVATIONS
# * Injury plays tended to end further outside the hashmarks (3.0833 yds rel_y_final = on hashmark), especially on synturf
# * plays slightly longer on natturf for non inj plays, but shorter for inj plays; exaggerates differences in _sum stats
#     * simplest explanation would be that synthetic turf is dangerous when plays are drawn out
# * Movement _avg stats all increase in injury plays
#     * increase for synturf vs natturf is exaggerated for d_twist_avg (torque?) and d_dir_avg (re-routing)
#     * natturf actually had higher increases in d_o_avg (pivoting) and d_s_avg (accel)
# 

# In[ ]:


display(playStats.groupby(['Inj', 'FieldType']).median().round(3))


# WITHOUT FILTERING PRE- AND POST-PLAY OBSERVATIONS
# * Medians almost universally lower than averages, suggesting more outlier plays with very high stats skewing avg higher
# * Same trends in median play end location (relative to hashmarks) and lengths as was found in means
# * _avg stats increase for injury plays again
#     * d_twist_avg (torque?) larger increase for synturf
#     * d_dir_avg (re-routing), d_o_avg (pivoting)  larger increase for natturf
#     * d_s_avg (accel) equivalent increase for both fieldtypes

# To help better highlight the interaction of "FieldType" and "Inj" upon these metrics, interaction plots can be generated of the levels among the 4 segments created by these two binary categorical features. 
# 
# One categorical feature forms the x-axis, while the other determines which points are connected by lines. Comparing the slopes of these lines shows the "difference of differences" that defines the interaction of the variables. Plots where the lines have drastically different slopes illustrate the most noteworthy interaction effects - such as for "rel_y_final" (second plot).

# In[ ]:


startDT = datetime.datetime.now()
plot_fts = playStats.columns[2:-1]
for i_1, feature_1 in enumerate( plot_fts ):
    simple_stats = playStats[['Inj', 'FieldType', feature_1]].copy(deep=True)
    simple_stats['plays'] = 1
    pivot_for_interax = simple_stats.groupby(['Inj', 'FieldType']).mean()
    #features = list(pivot_for_interax.columns)
    axes = list(pivot_for_interax.index.names)
    pivot_for_interax.reset_index(inplace=True)
    for col in axes:
        pivot_for_interax.loc[:, col] = pivot_for_interax[col].astype('str')
    #print(pivot_for_interax)
    ## https://www.statsmodels.org/dev/generated/statsmodels.graphics.factorplots.interaction_plot.html
    interax_plot = interaction_plot(x = pivot_for_interax['FieldType'], trace = pivot_for_interax['Inj'], response = pivot_for_interax[feature_1], plottype = 'both')
    interax_plot.suppressComposite #output plot only once, otherwise two copies are displayed
print(datetime.datetime.now() - startDT, 'to plot interactions between FieldType and Inj') ## about 1.0s on Kaggle


# ## Further processing

# In[ ]:


# adding additional columns for player-level, game-level, and play-level info
mlDF = playStats.copy(deep=True)
print(mlDF.shape)
mlDF['PlayerKey'] = mlDF['PlayKey'].astype(str).str[0:5]
mlDF['GameID'] = mlDF['PlayKey'].astype(str).str[0:7]
mlDF['endOutsideHash'] = 1*(mlDF['rel_y_final'] > (18.5/3/2) ) ## 3.0833 yds from midfield to hashmarks

mlDF = mlDF.merge(gamesDF[['GameID', 'Temperature', 'FieldExposed', 'FieldWet']], left_on='GameID', right_on='GameID')
mlDF = mlDF.merge(playersDF[['PlayerKey', 'RosterPosition']], left_on='PlayerKey', right_on='PlayerKey')
mlDF = mlDF.merge(playsDF[['PlayKey', 'Position', 'PositionGroup']], left_on='PlayKey', right_on='PlayKey')

print(mlDF.shape)
mlDF.head()


# In[ ]:


## this isn't strictly necessary but it helps with keeping the data concise

mlDF.loc[mlDF['RosterPosition'] == 'Quarterback', 'RosterPosition'] = 'QB'
mlDF.loc[mlDF['RosterPosition'] == 'Wide Receiver', 'RosterPosition'] = 'WR'
mlDF.loc[mlDF['RosterPosition'] == 'Linebacker', 'RosterPosition'] = 'LB'
mlDF.loc[mlDF['RosterPosition'] == 'Running Back', 'RosterPosition'] = 'RB'
mlDF.loc[mlDF['RosterPosition'] == 'Defensive Lineman', 'RosterPosition'] = 'DL'
mlDF.loc[mlDF['RosterPosition'] == 'Tight End', 'RosterPosition'] = 'TE'
mlDF.loc[mlDF['RosterPosition'] == 'Safety', 'RosterPosition'] = 'S'
mlDF.loc[mlDF['RosterPosition'] == 'Cornerback', 'RosterPosition'] = 'CB'
mlDF.loc[mlDF['RosterPosition'] == 'Offensive Lineman', 'RosterPosition'] = 'OL'
mlDF.loc[mlDF['RosterPosition'] == 'Kicker', 'RosterPosition'] = 'K'

mlDF['RosterPosition'].unique()


# To enable greater analysis flexibility, the categorical variables "FieldType", "RosterPosition", "Position" (i.e. role for play), and "PositionGroup" (i.e role group for play) will need to be converted one-hot variables (aka dummy variables).

# In[ ]:


## Because this process needs to be repeated for three very similar variables, the resulting column names need to be prefixed.
## pd.get_dummies() could have been used, but it was using too much memory so this semi-manual process was used instead.

## convert position categorical features into one-hot feature sets
print('Converting RosterPosition...')
for rosterPos in mlDF['RosterPosition'].unique():
    #print(rosterPos)
    mlDF['ros_' + rosterPos] = 0
    mlDF.loc[mlDF['RosterPosition'] == rosterPos, 'ros_' + rosterPos] = 1
mlDF.drop('RosterPosition', axis='columns', inplace=True)

print('Converting (play) Position...')
for playPos in mlDF['Position'].unique():
    #print(playPos)
    mlDF['play_' + playPos] = 0
    mlDF.loc[mlDF['Position'] == playPos, 'play_' + playPos] = 1
mlDF.drop('Position', axis='columns', inplace=True)

print('Converting (play) PositionGroup...')
for playPosGrp in mlDF['PositionGroup'].unique():
    #print(playPosGrp)
    mlDF['playGrp_' + playPosGrp] = 0
    mlDF.loc[mlDF['PositionGroup'] == playPosGrp, 'playGrp_' + playPosGrp] = 1
mlDF.drop('PositionGroup', axis='columns', inplace=True)

mlDF.drop(['PlayerKey', 'GameID'], axis='columns', inplace=True)

mlDF['SynTurf'] = 0
mlDF.loc[mlDF['FieldType'] == 'Synthetic', 'SynTurf'] = 1
mlDF.drop('FieldType', axis='columns', inplace=True)

print('Done!')

## verifying results with a side-scrollable display of the dataframe
HTML(mlDF.head().to_html())


# ## Statistical analysis using significance tests
# 
# First, it should be pointed out that this problem is pretty much the definition of "needle in a haystack" due to the incredibly small variance that will be nearly impossible to model effectively.

# In[ ]:


print('Variance in whether plays result in injury:', mlDF['Inj'].var() )


# Testing the first hypothesis is the simplest - the plays are segmented into samples based on whether they were on synthetic turf, then the proportion of injurious plays in each sample is tested to determine if the difference is statistically significant based on the size of the samples.

# In[ ]:


pd.options.display.float_format = '{:.10f}'.format ## need  greater level of detail due to tiny injury occurence proportions

display(mlDF[['Inj', 'SynTurf']].groupby('SynTurf').mean())
display(ttest_ind(mlDF.loc[mlDF['SynTurf']==1,'Inj'], mlDF.loc[mlDF['SynTurf']==0,'Inj']))

## p-value of 0.0436 suggests statistically significant difference in likelihood of injury based on turf type

inj_turf_corr = mlDF['Inj'].corr(mlDF['SynTurf'])
print('The correlation of turf type and injury occurence is only', round(inj_turf_corr, 6), 'which means only', round((inj_turf_corr**2), 10), ' of the variance in injury occurence would be explainable by turf type (possibly through other factors)' )


# In order to determine if other features might be better predictors, either alone or in interaction with turf type, further testing is necessary. First, do any features correlate better with injury occurence than turf type?

# In[ ]:


corrDF = pd.DataFrame(columns = ['Feature', 'Corr_w_Inj'])

feat_for_corr = list(mlDF.columns)[1:-1] #exclude PlayKey and SynTurf
feat_for_corr.remove('Inj')

for i_col, col in enumerate(feat_for_corr):
    if col == 'Inj':
        print(col)
    corrDF.loc[i_col] = [col, mlDF[col].corr(mlDF['Inj'])]

corrDF['R2'] = corrDF['Corr_w_Inj']**2

display(corrDF.sort_values('R2').loc[corrDF['R2'] > (inj_turf_corr**2)])

features_to_test = list(corrDF.sort_values('R2').loc[corrDF['R2'] > (inj_turf_corr**2)]['Feature'])

features_cat = list()
features_con = list()

for ft in features_to_test:
    if(mlDF[ft].max() == 1 and mlDF[ft].min() == 0):
        features_cat.append(ft)
    else:
        features_con.append(ft)


# Several features had stronger correlation with injury occurence than turf type:
# * All 8 of the play-level aggregate features (4 sums, 4 avgs), 
# * The continuous "rel_y_final" (lateral position relative to the midpoint between the hashmarks, inverse of proximity to sideline)
# * Temperature
# * Three inherently similar roles (rostered Offensive Lineman, play Offensive Guard (specific spot on line), and play Offensive Line)
# 
# ## Categorical features, individual significance testing
# 
# The categorical variables can be tested in the same way that turf type was.

# In[ ]:


print('Categorical:', features_cat)


# In[ ]:


startDT = datetime.datetime.now()
catFtsDF = pd.DataFrame(columns=['Feature', 'FalseInj', 'TrueInj', 'p-val'])

for i_cat, cat_ft in enumerate(features_cat):
    inj_ps = mlDF[['Inj', cat_ft]].groupby(cat_ft).mean()['Inj']
    pval = ttest_ind(mlDF.loc[mlDF[cat_ft]==1,'Inj'], mlDF.loc[mlDF[cat_ft]==0,'Inj']).pvalue
    catFtsDF.loc[i_cat] = [cat_ft, inj_ps[0], inj_ps[1], pval]
    
display(catFtsDF)
print(datetime.datetime.now() - startDT, 'to test all categorical features') ## about 0.1s on Kaggle


# Individually, each categorical feature segments plays into samples with statistically significant injury likelihoods, all four more significant than turf type (p-values below 0.0436).
# 
# ## Categorical features, interaction effects
# 
# Given that some of these variables are obviously (or at least intuitively) connected to each other or turf type, isolating interaction effects between pairs of categorical features might help determine whether the significance of any of them depends on another.

# In[ ]:


startDT = datetime.datetime.now()
model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf', data=mlDF).fit()
print(model_InjCouple.pvalues)
print()

for cat_ft in features_cat:
    model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf*' + cat_ft, data=mlDF).fit()
    print(model_InjCouple.pvalues)
    print()
    
for cat_ft1 in features_cat:
    for cat_ft2 in features_cat:
        if cat_ft1 != cat_ft2:
            print(cat_ft1, '*', cat_ft2)
            both1 = mlDF[(mlDF[cat_ft1] == 1) & (mlDF[cat_ft2] == 1)].shape[0]
            both0 = mlDF[(mlDF[cat_ft1] == 0) & (mlDF[cat_ft2] == 0)].shape[0]
            first1 = mlDF[(mlDF[cat_ft1] == 1) & (mlDF[cat_ft2] == 0)].shape[0]
            second1 = mlDF[(mlDF[cat_ft1] == 0) & (mlDF[cat_ft2] == 1)].shape[0]
            if(both1 == 0 or both0 == 0 or first1 == 0 or second1 == 0):
                print("One feature is a subset of the other, no interaction effect necessary")
            else:
                model_InjCouple = sm.Logit.from_formula('Inj ~' + cat_ft1 + '*' + cat_ft2, data=mlDF).fit()
                print(model_InjCouple.pvalues)
            print()
print(datetime.datetime.now() - startDT, 'to test interactions between categorical features') ## about 15s on Kaggle


# Lots of output to read through, but the key takeaways were:
# * Each of the 3 offensive line/guard variables depend heavily on turf type for their significance, based on their p-values rising while the p-value for "SynTurf" remains significant when the interaction effect is isolated
# * As expected, those 3 variables are subsets of one another, which means that can't be modeled together; for example, there are no record where the player was an offensive guard for the play but not on the offensive line.
# 
# These interactions can also easily be visualized with the same plot format as was used earlier:

# In[ ]:


startDT = datetime.datetime.now()
plot_fts = ['SynTurf'] + features_cat
for i_1, feature_1 in enumerate( plot_fts ):
    for i_2, feature_2 in enumerate( plot_fts ):
        if i_1 < i_2:
            simple_mlDF = mlDF[['Inj', feature_1, feature_2]].copy(deep=True)
            simple_mlDF['plays'] = 1

            pivot_for_interax = simple_mlDF.groupby([feature_1, feature_2]).sum()

            for col in pivot_for_interax.columns:
                pivot_for_interax.loc[:, col] = pivot_for_interax[col] / pivot_for_interax['plays']

            pivot_for_interax.drop(labels='plays', axis='columns', inplace=True)
            #features = list(pivot_for_interax.columns)
            axes = list(pivot_for_interax.index.names)
            pivot_for_interax.reset_index(inplace=True)
            for col in axes:
                pivot_for_interax.loc[:, col] = pivot_for_interax[col].astype('str')

            #print(pivot_for_interax)

            ## https://www.statsmodels.org/dev/generated/statsmodels.graphics.factorplots.interaction_plot.html

            interax_plot = interaction_plot(x = pivot_for_interax[feature_1], trace = pivot_for_interax[feature_2], response = pivot_for_interax['Inj'], plottype = 'both')
            interax_plot.suppressComposite #output plot only once, otherwise two copies are displayed
print(datetime.datetime.now() - startDT, 'to plot interactions between categorical features') ## about 1.0s on Kaggle


# ## Continuous variables, individual significance testing
# 
# The continuous variables that were strongly correlated with injury occurences on an individual basis can be tested for significance in a similar manner:

# In[ ]:


print('Continuous:', features_con)


# First, the significance of each continuous feature's relationship, in terms of whether an increase in that feature is associated with an increase in injury occurence, can be tested individually.

# In[ ]:


startDT = datetime.datetime.now()
for con_ft in features_con:
    model_InjCouple = sm.Logit.from_formula('Inj ~ ' + con_ft, data=mlDF).fit()
    print(model_InjCouple.pvalues)
    print()
print(datetime.datetime.now() - startDT, 'to test each continuous feature') ## about 10s on Kaggle


# Again, lots of output due the number of features, but each continuous variable tested appears to have a significant relationship with injury occurence. There are noteworthy tiers within these variables though: 
# * 'd_twist_avg', 'd_dir_avg', 'Temperature', 'd_twist_sum', and 'd_dir_sum' each had p-values between 0.02 to 0.04
# * 'd_s_avg', 'd_s_sum', and 'rel_y_final' had p-values an order lower (0.00#... ; more significant)
#     * the two 'd_s' features are measures of acceleration experienced, which matches a basic physics intuition that higher acceleration equals higher force equals more injuries
# * 'd_o_sum' and 'd_o_avg' had p-values even lower p-values
#     * these features are measures of spinning experienced, which matches a biological common sense of lower-body injuries generally occuring when a player's feet/legs aren't kept parallel to one another
# 
# ## Continuous features, interaction effects
# 
# Same as the categorical variables, the interactions between these continuous variables and turf type, as well as with each other, should be examined to determine if any feature depends on another for significance. Interaction plots are helpful for continuous variables as well, but with 250k+ records, they would be resource-intensive to produce.

# In[ ]:


startDT = datetime.datetime.now()
model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf', data=mlDF).fit()
print(model_InjCouple.pvalues)
print()

for con_ft in features_con:
    model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf*' + con_ft, data=mlDF).fit()
    print(model_InjCouple.pvalues)
    print()
print(datetime.datetime.now() - startDT, 'to test interaction effects between turf type and continuous features') ## about 10s on Kaggle


# Again, key takeaways from that wall of output:
# * The significance of almost every continuous variable is muddled when interacting with turf type ("SynTurf"), although some were still close to being significant after the interaction effect was isolated, with others were almost entirely insignificant (p-values near 1.0)
# * "d_o_sum" and "d_o_avg" were both still significant after their interactions with turf type were isolated; while neither have a significant interaction with turf type, they differ in that the significane of turf type is actually lessened by "d_o_sum", while "d_o_avg" appears to be independent from turf type
#     * This suggests that turf type has a relatively stronger correlation with "d_o_sum" than "d_o_avg", which would likely be due to another relatively strong correlation with the length of plays ("obs" in mlDF). Both mini-hypotheses will be checked briefly below via correlation calculation and box-plot distribution comparison.

# In[ ]:


print(mlDF['SynTurf'].corr(mlDF['d_o_avg']) )
box_data = [mlDF.loc[mlDF['SynTurf'] == 0, 'd_o_avg'], mlDF.loc[mlDF['SynTurf'] == 1, 'd_o_avg']]
display(plot.boxplot(box_data, vert=False, showfliers = False))


# In[ ]:


print(mlDF['SynTurf'].corr(mlDF['d_o_sum']) )
box_data = [mlDF.loc[mlDF['SynTurf'] == 0, 'd_o_sum'], mlDF.loc[mlDF['SynTurf'] == 1, 'd_o_sum']]
display(plot.boxplot(box_data, vert=False, showfliers = False))


# In[ ]:


print(mlDF['SynTurf'].corr(mlDF['obs']) )
box_data = [mlDF.loc[mlDF['SynTurf'] == 0, 'obs'], mlDF.loc[mlDF['SynTurf'] == 1, 'obs'] ]
display(plot.boxplot(box_data, vert=False, showfliers = False))


# Both mini-hypotheses appear to be confirmed! All 3 correlations are all pretty weak in absolute terms, but binary variables and continuous variables are difficult to assess with correlation. The box-plots illustrate the difference between natural turf and synthetic turf plays is much more noticable for "d_o_sum" and "obs" than for "d_o_avg".
# 
# One more extremely long output is needed to examine the interaction effects between each of the continuous variables - see below, with takeaways to follow.

# In[ ]:


startDT = datetime.datetime.now()
for i_ft1, con_ft1 in enumerate(features_con):
    for i_ft2, con_ft2 in enumerate(features_con):
        if i_ft1 < i_ft2:
            print(con_ft1, '*', con_ft2)
            model_InjCouple = sm.Logit.from_formula('Inj ~' + con_ft1 + '*' + con_ft2, data=mlDF).fit()
            print(model_InjCouple.pvalues)
            print()
print(datetime.datetime.now() - startDT, 'to test interactions within continuous features') ## about 45s on Kaggle


# Similar to the takeaways for the categorical variables' interaction effects with one another:
# * Most variable pairs muddled each other's significance, although to varying degrees
# * The following variables' relationship with injury occurence remained significant after isolating interaction effect with at least one other variable (note that these are not necessarily reciprocal):
#     * "d_twist_avg": "d_s_avg", "rel_y_final", 
#     * "d_s_avg": "d_twist_avg", "d_dir_avg"
#     * "d_o_avg": "d_twist_avg", "d_dir_avg", "d_s_avg", "rel_y_final", "d_o_sum"
#     * "d_dir_avg": "rel_y_final", "d_o_avg"
#     * "rel_y_final": "d_dir_avg", "d_dir_sum", "d_twist_sum", "d_s_sum", "d_o_sum"
#     * "Temperature": "d_s_sum"
#     * "d_s_sum": "Temperature"
#     * "d_twist_sum": "rel_y_final"
#     * "d_o_sum": "d_twist_sum", "d_dir_sum", "d_s_avg", "d_s_sum", "rel_y_final", "d_o_avg"
#     * "d_dir_sum": "rel_y_final"
#     * "d_s_avg": "rel_y_final", "d_o_sum", "d_o_avg"
#     * "d_s_sum": "rel_y_final", "d_o_sum"
# * The only interaction that was significant was d_s_sum:rel_y_final (p-value 0.008801), but both variables' relationship with injury occurence was still easily significant after isolating their interaction.
#     * This suggests a very intuitive relationship between a players' experienced acceleration and their proximity to the sidelines; i.e. the players who are changing speeds most often are generally playing closer to the sidelines, while lineman on both offense and defense stay closer to the middle and don't change speeds as much. This correlation will be calculated below to confirm this mini-hypothesis.
#     * The only other two interactions that were even close to being significant were d_twist_avg:d_s_avg (p-value 0.089105) and d_twist_sum:rel_y_final (p-value 0.093208); again, both are intuitive relationships, but do not need to be explored further due to lack of true significance.

# In[ ]:


mlDF['d_s_sum'].corr(mlDF['rel_y_final'])


# Another mini-hypothesis confirmed!
# 
# ## Final classified model effectiveness
# 
# Before attempting to a final model, it's worth remembering the miniscule amount of variation in injury occurence that is attempting to be explained. The signal-to-noise ratio is extremely low, and none of the individual factors reviewed so far have been truly strong predictors individually.
# 
# Based on the factors identified as the (relatively) strongest predictors, a logistic regression model can be fitted to a training subset of the plays and then tested on the remaining test set. As a logistic regression, its predictions are probabilities of injuries occuring on the test plays, and the actual classification as Inj = 0 or Inj = 1 depends on the threshold. 
# 
# To evaluate the model's performance, many thresholds are processed to calculate numbers of True Positives (TPs) and False Positives (FP), which are then plotted against each other to form a Receiver Operating Characteristic (ROC) curve. Ideally a classifier has a greater proportion of TP than FP at all thresholds and will be above a diagonal line on this plot. Even in that case though, the statistical metric for classifier quality is the Area Under the Curve (AUC), which can be interpreted on a scale from 1 to 0:
# * 1.0 is best, perfect alignment of predicted classification to actual classification
# * 0.0 is worst, where every record is classified incorrectly
# * 0.5 is NOT considered middling performance, rather it means the model is unable to differentiate the classes at all

# In[ ]:


startDT = datetime.datetime.now()
model_fts = ['SynTurf', 'd_o_avg', 'd_o_sum', 'rel_y_final', 'd_s_sum', 'd_s_avg']
X_train, X_test, y_train, y_test = train_test_split(mlDF[model_fts], mlDF['Inj'], test_size=0.33, random_state=42)
print(sum(y_train == 1), 'injuried in training set,', sum(y_test == 1), 'injuried in test set',)
logreg = LogisticRegression(penalty='none', solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
print('AUC:', logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plot.figure()
plot.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plot.plot([0, 1], [0, 1],'r--')
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.title('Receiver operating characteristic')
plot.legend(loc="lower right")
plot.savefig('Log_ROC')
plot.show()
print(datetime.datetime.now() - startDT, 'to fit, test, and evaluate logistic regression model') ## about 10s on Kaggle


# As expected, the AUC of this model is 0.5.
# 
# ## Conclusions
# 
# Speaking of hypotheses, it time to revisiting those stated at the beginning of the workbook:
# * There is a significant increase in injury occurence for plays on synthetic turf vs. plays on natural turf.
#     * CONFIRMED based on test result p-value of 0.04355063587737655
# * However, due to the extremely low proportion of plays with injuries overall, turf type alone will barely explain any of the variation in injury occurence. 
#     * CONFIRMED based on an R^2 value of 0.0000152604; although it is worth noting, as was stated immediately prior to the calculation of that R^2, there's not much variance in injury occurence among the plays to be explained ("needle-in-a-haystack" problem)
# * While no other individual play-level factor will explain much more of the variation in injury occurence, those that do may have significant interaction effects with turf type, suggesting that synthetic turf's elevates injury risk through amplifying other risk factors.
#     * UNCONFIRMED
#     * No game-level factors such as whether the field was exposed (based on StadiumType) or wet (based on weather) were found to be noteworthy predictors
#     * No interactions with turf type were found to be significant, but most variables' significance was reduced when their interaction with turf type was isolated and factored out, suggesting that there is some sort of more complex interaction involving additional confounding variables affecting the variables' relationship with injury occurence. Modeling interactions between 3+ variables is a much more involved process and was left for future exploration.
#     * When including interactions between variables other than turf type, the only interaction found to be significant was between 'd_s_sum' and 'rel_y_final'.
#     * The best features by correlation strength and significance testing p-value were:
#         * 'd_o_avg', 'd_o_sum' - how much a player "spun" during a play
#             * These were more important injury risks than turf type and didn't appear to be amplified by synthetic turf
#         * 'rel_y_final' - distance from the middle of the field (i.e. a line perpendicular to the yardlines), inverse of proximity to sideline
#             * This was also an important injury risk, but possibly related to turf type
#         * 'd_s_sum', 'd_s_avg' - how much a player accelerated during a play
#             * These were important injury risks, but it was unclear whether they were related-to / independent-from turf type
#         * (no categorical features were as predictive as these continuous variables, not even turf type)
#         
# ## Final Thoughts
# 
# Synthetic turf does appear to be related to a higher risk for non-contact lower-extremity injuries, but a few other simple features calculated from player movement data have relationships of equal or greater statistical significance. Without access to cost information on the removal/replacement of synthetic turf vs. these injuries, it's difficult to say whether a decision can or should be made regarding turf types. 
# 
# It is very clear though, that removing synthetic turf from the NFL is not a silver-bullet solution to this injury issue. More research is needed to monitor the injury risk of players' spin, proximity to the sideline, and acceleration. While there is probably no way to reduce these tendencies without harming the game of football; there may be policies, equipment, or other steps that can be taken to make them safer. If nothing else, based on this movement tracking data, players who experience these risk factors at extreme levels should be examined soon after for subtle injuries that might be exacerbated by further risky movement.

# # OVERALL RUN TIME

# In[ ]:


print('Entire workbook runs in ', datetime.datetime.now() - overall_startDT)

