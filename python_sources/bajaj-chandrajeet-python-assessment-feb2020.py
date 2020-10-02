#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:



#Question - 01

import numpy as np
import pandas as pd

odi = pd.read_csv('/kaggle/input/trainings/odi-batting.csv')
#odi.head()
print("Sachin has played", odi[(odi['Player'] == 'Sachin R Tendulkar')].shape[0], "matches.")
print("Rahul has played", odi[(odi['Player'] == 'Rahul Dravid')].shape[0], "matches.")
print("Together they have played", odi[(odi['Player'] == 'Sachin R Tendulkar') | (odi['Player'] == 'Rahul Dravid')]['URL'].nunique(), "unique matches in total.")
odi_s = odi[(odi['Player'] == 'Sachin R Tendulkar')]['URL']
odi_r = odi[(odi['Player'] == 'Rahul Dravid')]['URL']
odi_common = pd.merge(odi_s, odi_r, how='inner')
print("They have played", odi_common.shape[0] , "unique matches in common (i.e. both were in same match).")


# In[ ]:


#Question - 02

import numpy as np
import pandas as pd

def f_InactiveYears(pODI, pName):
    pODI = pODI[(pODI['Player'] == pName)]
    if len(pODI) == 0:
        return -99 #-99 is a result when player is not found.
    pODI = pODI['MatchDate'].str[-4:]
    pODI = pODI.astype(np.int)
    return pODI.max() - pODI.min() + 1 - len(np.unique(pODI))
    
odi_data = pd.read_csv('/kaggle/input/trainings/odi-batting.csv')
print(f_InactiveYears(odi_data, 'Sachin R Tendulkar'))
print(f_InactiveYears(odi_data, 'Rahul Dravid'))
print(f_InactiveYears(odi_data, 'Rahul Ganguly'))

odi_data_players = odi_data[['Player']]
odi_data_players = odi_data_players.drop_duplicates()

odi_data_players_inactiveyears = pd.DataFrame(columns=['Player','InactiveYears'])
for x in odi_data_players['Player']:
    #print(x, f_InactiveYears(odi_data, x))
    odi_data_players_inactiveyears = odi_data_players_inactiveyears.append({'Player':x, 'InactiveYears':f_InactiveYears(odi_data, x)}, ignore_index=True)

odi_data_players_inactiveyears.sort_values(by=['InactiveYears'],ascending=False).head(10)


# In[ ]:


#Question - 03

import numpy as np
import pandas as pd
from datetime import datetime

def f_YearsFor2000Runs(pODI, pName):
    pODI = pODI[(pODI['Player'] == pName)]
    if len(pODI) == 0:
        return -99 #-99 is a result when player is not found.

    pODI = pODI[['Player','MatchDate','Runs']]
    pODI['MatchDate'] = pd.to_datetime(pODI['MatchDate'])
    pODI = pODI.sort_values(by=['Player','MatchDate'])

    runs = 0
    matches = 0
    for idx, idxrow in pODI[(pODI['Player'] == pName)].iterrows():
        runs = runs + idxrow['Runs']
        matches = matches + 1
        if runs > 1999:
            return matches
    return -98 #-98 is a result when player has not scored 2000 runs in his career.
        
odi_data = pd.read_csv('/kaggle/input/trainings/odi-batting.csv')

odi_data_players = odi_data[['Player']]
odi_data_players = odi_data_players.drop_duplicates()
#print(odi_data_players.shape)

#BELOW CAN BE ADDED TO MAKE SET SMALLER BY TAKING ONLY THOE PLAYERS WHO HAVE SCORED 2000+ RUNS
#odi_data_players = odi_data.groupby(['Player'])['Runs'].agg('sum').reset_index()
#odi_data_players = odi_data_players[(odi_data_players['Runs'] > 1999)] 

odi_data_players_years2000 = pd.DataFrame(columns=['Player','Years2000'])

for x in odi_data_players['Player']:
    y = f_YearsFor2000Runs(odi_data, x)
    #print(x,y)
    if y > 0:
        odi_data_players_years2000 = odi_data_players_years2000.append({'Player':x, 'Years2000':y}, ignore_index=True)

odi_data_players_years2000.sort_values(by=['Years2000'],ascending=True).head(10)


# In[ ]:


#Question - 04

import numpy as np
import pandas as pd
from datetime import datetime

def f_MatchesFor10Hundreds(pODI, pName):
    pODI = pODI[(pODI['Player'] == pName)]
    if len(pODI) == 0:
        return -99 #-99 is a result when player is not found.

    pODI = pODI[['Player','MatchDate','Runs']]
    pODI['MatchDate'] = pd.to_datetime(pODI['MatchDate'])
    pODI = pODI.sort_values(by=['Player','MatchDate'])

    hundreds = 0
    matches = 0
    for idx, idxrow in pODI[(pODI['Player'] == pName)].iterrows():
        matches = matches + 1
        if idxrow['Runs'] > 99:
            hundreds = hundreds + 1
            if hundreds > 9:
                return matches
    return -98 #-98 if the player has not scored more than 10 hundreds
        
odi_data = pd.read_csv('/kaggle/input/trainings/odi-batting.csv')
#print(f_MatchesFor10Hundreds(odi_data,'Sachin R Tendulkar'))

odi_data_players_hundreds = pd.DataFrame(columns=['Player','MatchesForHundres'])

odi_data_players = odi_data[['Player']]
odi_data_players = odi_data_players.drop_duplicates()
print(odi_data_players.shape)

for x in odi_data_players['Player']:
    y = f_MatchesFor10Hundreds(odi_data, x)
    if y > 0:
        odi_data_players_hundreds = odi_data_players_hundreds.append({'Player':x, 'MatchesForHundres':y}, ignore_index=True)

odi_data_players_hundreds.sort_values(by=['MatchesForHundres'],ascending=True).head(10)
odi_data_players_hundreds.sort_values(by=['MatchesForHundres'],ascending=True).head(10).plot.bar(x='Player',y='MatchesForHundres')


# In[ ]:


#Question - 05



# In[ ]:


#Question - 06

import numpy as np
import pandas as pd

myDF = pd.DataFrame([100, 104, 99, 100, 100, 100, 98, 105, 105, 100, 110, 110,110, 110, 100], columns =['BaseData'])
myDF['AbsoluteDiff'] = myDF.BaseData.diff()
myDF['PercentDiff'] = myDF.BaseData.pct_change() * 100

constantpatches = 0
prevdiff = 10

for x in myDF['AbsoluteDiff']:
    if x == 0:
        if prevdiff != 0:
            constantpatches = constantpatches + 1
    prevdiff = x
    #print(prevdiff)

print('Number of Constant patches :' ,constantpatches)

print('Number of 5% Positive changes :' , myDF[(myDF['PercentDiff'] >= 5)].shape[0])


# In[ ]:



#Question - 07

import numpy as np
import pandas as pd
from datetime import datetime

bank_data = pd.read_csv('/kaggle/input/trainings/bank-full.csv',';')

myanswer = bank_data[(bank_data['age'] < 30) & (bank_data['job'] == 'student') & (bank_data['y'] == 'yes')].shape[0]
print('Number of students, under 30 years of age who said YES are :',myanswer)


# In[ ]:



#Question - 08

import numpy as np
import pandas as pd
from datetime import datetime

bank_data = pd.read_csv('/kaggle/input/trainings/bank-full.csv',';')

mypt = pd.pivot_table(bank_data, values='campaign', index=['job'], columns=['marital'], aggfunc='sum', fill_value=0, margins=True)
mypt.head(1000)


# In[ ]:


#Question - 09

import numpy as np
import pandas as pd
from datetime import datetime

rest_data = pd.read_csv('/kaggle/input/trainings/restaurant_reviews.csv')

rest_data['TotalReviews'] = rest_data.rev_count.str.extract(r'^(?:\S+\s){0}(\S+)', expand=True)
rest_data['TotalFollowers'] = rest_data.rev_count.str.extract(r'^(?:\S+\s){3}(\S+)', expand=True)
rest_data.head()


# In[ ]:


#Question - 10

#First execute Question9

#Top10 users with max followers
rest_data[['rev_name','TotalFollowers']].drop_duplicates().sort_values(by=['TotalFollowers'],ascending=False).head(10)

#Top10 rests with highest avg review
rest_data['ReviewFloat'] = rest_data.rating.str.extract(r'^(?:\S+\s){1}(\S+)', expand=True).astype(float)
rest_avg_rating = rest_data.groupby(['res_id','res_name'])['ReviewFloat'].agg('mean').reset_index()
rest_avg_rating.sort_values(by=['ReviewFloat'],ascending=False).head(10)

#min, max dates and avg number of reviews in a day
rest_data['date'] = pd.to_datetime(rest_data['date']).dt.date
print('Min review date :',rest_data['date'].min())
print('Max review date :',rest_data['date'].max())
print('Total reviews :',rest_data['date'].count())
print('Unique days',rest_data['date'].drop_duplicates().count())
print('Avg reviews per day',rest_data['date'].count()/rest_data['date'].drop_duplicates().count())

#Daily ratings
daily_ratings = rest_data.groupby(['date'])['ReviewFloat'].agg('count').reset_index()
daily_ratings.plot.line(x='date',y='ReviewFloat')

