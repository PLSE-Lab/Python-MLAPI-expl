# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#  This is a attempt to use the Colley method ( simple versio) to rank the teams in the group stage and when they get to the round 16 stage 
# Probably some very convoluted code but its a bit of a work in progress ..
# ANyway hope you can find it useful and any comments improvements would be welcome . Thank you ..

matches  = pd.read_csv("../input/WorldCupMatches.csv")
matches = pd.DataFrame(matches,  columns= ['Year','Datetime', 'Stage', 'Stadium', 'City', 'Home Team Name',
       'Home Team Goals', 'Away Team Goals', 'Away Team Name',
       'Win conditions', 'Attendance', 'Half-time Home Goals',
       'Half-time Away Goals', 'Referee', 'Assistant 1', 'Assistant 2',
       'RoundID', 'MatchID', 'Home Team Initials', 'Away Team Initials'] )


matches = matches.set_index('Year')
matches = matches.drop(['Stadium','City', 'Win conditions', 'Attendance', 'Referee', 'Assistant 1', 'Assistant 2', 'RoundID', 'MatchID'], axis=1)
# Just pick 2 of the latest years to rank although I only use 2010 
m2 = matches[(matches.index >= 2010)]

m10 = matches[(matches.index == 2010)]
#m14 = matches[(matches.index == 2014)]

srt10 = m10.sort_values(by=['Stage'],  ascending = True)
#srt14 = m14.sort_values(by=['Stage'],  ascending = True)
# set all the teams to equal rank 
srt10['r'] = 1/2


# Test
test = srt10.iloc[0:7]
test = test.where(test['Stage']=='Group A').dropna()



test = test.sort_values(by = 'Datetime' , ascending=True)# order by matches
test['home win'] = (test['Home Team Goals'] >  test['Away Team Goals']) == 1 
test['away win']  = (test['Home Team Goals'] <  test['Away Team Goals']) == 1

# no of matches played per group

n = len(test)/2
# get the wins per team ..
hw = test.pivot(index = 'Datetime', columns = 'Home Team Initials', values='home win')
aw = test.pivot(index = 'Datetime', columns = 'Away Team Initials', values='away win')

allresults = pd.concat([hw, aw]) 
grpteams = allresults.columns

totalres = pd.DataFrame(columns = grpteams)
# sum of the wins per group 
# simple version of Colley method for ranking teams 
t1 = pd.DataFrame(np.sum(allresults))
cr_q = 2 + n

t1['cr'] = (1 + t1.iloc[0:] ) /cr_q

# All Groups >>>
# equivalent of doing a where in statement >>> 
groups = ['Group A', 'Group B','Group C', 'Group D', 'Group E', 'Group F','Group G', 'Group H']
test2 = srt10[srt10['Stage'].isin(groups)].dropna()

test2 = test2.sort_values(by = 'Datetime' , ascending=True)# order by matches
test2['home win'] = (test2['Home Team Goals'] >  test2['Away Team Goals']) == 1 
test2['away win']  = (test2['Home Team Goals'] <  test2['Away Team Goals']) == 1

# no of matches played per group
# 3 matches /grp 48 teams/ 3 = 16 

n = 3

hw = test2.pivot(index = 'Datetime', columns = 'Home Team Initials', values='home win')
aw = test2.pivot(index = 'Datetime', columns = 'Away Team Initials', values='away win')


allresults = pd.concat([hw, aw]) 
grpteams = allresults.columns

totalres = pd.DataFrame(columns = grpteams)

#sumgrps = pd.DataFrame(, columns = 'Totwins')
sumgrps = pd.DataFrame(np.sum(allresults))
cr_q = 2 + n
sumgrps['cr'] = (1 + sumgrps.iloc[0:] ) /cr_q

rnd16 = ['Round of 16']

round16 = srt10[srt10['Stage'].isin(rnd16)].dropna()

round16 = round16 .sort_values(by = 'Datetime' , ascending=True)# order by matches
round16 ['home win'] = (round16 ['Home Team Goals'] >  round16 ['Away Team Goals']) == 1 
round16 ['away win']  = (round16 ['Home Team Goals'] <  round16 ['Away Team Goals']) == 1

hw16 = round16.pivot(index = 'Datetime', columns = 'Home Team Initials', values='home win')
aw16 = round16.pivot(index = 'Datetime', columns = 'Away Team Initials', values='away win')
results16 = pd.concat([hw16, aw16]) 
totalres16 = pd.DataFrame(columns = grpteams)
sumgrps16 = pd.DataFrame(np.sum(results16))
cr_q = 2 + 1
sumgrps16['cr16'] = (1 + sumgrps16.iloc[0:] ) /cr_q

# now match the round 16 to the groups results >>>>>

match16_grp = pd.merge(sumgrps, sumgrps16, how = 'inner',left_index = True, right_index = True)  

match16_grp['ranknow'] = match16_grp['cr']* match16_grp['cr16']

plt.plot(match16_grp['cr'])
plt.plot(match16_grp['cr16'])
plt.show()


# Any results you write to the current directory are saved as output.