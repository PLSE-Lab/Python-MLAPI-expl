# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input/wta"]).decode("utf8"))

players = pd.read_csv("../input/wta/players.csv",  encoding='latin', index_col=0)

# Any results you write to the current directory are saved as output.
# Top column is misaligned.
players.index.name = 'ID'
players.columns = ['First' , 'Last', 'Hand', 'DOB', 'Country']
# Parse date data to dates.
players = players.assign(DOB=pd.to_datetime(players['DOB'],format='%Y%m%d'))
# Handidness is reported as U if unknown; set np.nan instead.
players = players.assign(Hand=players['Hand'].replace('U', np.nan))
players['YOB'] = players['DOB'].apply(lambda b: b.year)
# Players after 80  and right
print('After 80 and right hand:')
print(players.loc[(players["DOB"] >  "1980-01-01") & (players["Hand"] =="R" ) ,["First","Last"]])
# players after 90
print('After 90: ',players.loc[players["DOB"] >  "1990-01-01" ,["First","Last"]])
# Before 70:
print('Before 70: ',players.loc[players["DOB"] <  "1970-01-01" ,["First","Last"]])
# Players representing USA
print('USA Players')
print(players.loc[players["Country"]  == "USA" ,["First","Last"]])
print('Graph:')

#players.groupby('Country').size().plot(title="Country representation over the years",figsize=(20,10))
players.YOB.value_counts(dropna=False).plot.bar(figsize=(12, 6),
                                                       title='WTA Player Hand')
                  
players.groupby(['DOB', 'Hand']).size().unstack('Hand').replace(np.NaN, 0).plot(title="Players' hand over the years",figsize=(20,10)) 
                  
#### Matches:
matches = pd.read_csv("../input/wta/matches.csv", encoding='latin1', index_col=0)
## Most Wins - Top 10. : 
print('~~~',matches.groupby('winner_name').winner_name.count().nlargest(10))

# Based on surface 
#print('~~~~',matches.agg('winner_name','surface'))
