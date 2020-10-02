#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


matches = pd.read_csv("../input/matches.csv")
players = pd.read_csv("../input/players.csv")
teams = pd.read_csv("../input/teams.csv")
matchLineups = pd.read_csv("../input/matchLineups.csv")
matchIDs = pd.read_csv("../input/matchIDs.csv")
eventIDs = pd.read_csv("../input/eventIDs.csv")
joinMatchEvent = pd.read_csv("../input/joinMatchEvent.csv")


# ###Map Histogram

# In[ ]:


maps = matches['Map'].value_counts().plot(kind='bar',title='Map Histogram')
maps.set_xlabel("Map")
maps.set_ylabel("Number of matches")


# ###Team Statistics
# Top 20 teams with highest win percentage (minimum 100 matches).

# In[ ]:


#Team Statistics
y=[]
for i in np.array(matches):
	won,lose,k=(2,8,3) if i[4]>i[10] else (8,2,9)
	diff=abs(i[4]-i[10])
	diff = diff if i[k]=='CT' else -1*diff 
	y.append([i[1],i[won],diff,i[lose]])
df=pd.DataFrame(y,columns=['Map','Won','Diff wrt CT','Lost'])
won = df.groupby('Won')['Won'].count().reset_index(name='count')
lose = df.groupby('Lost')['Lost'].count().reset_index(name='count')
ratio = []
for i in np.array(lose):
	try:
		ind = list(won['Won']).index(i[0])
		r = won['count'][ind]*100/(i[1]+won['count'][ind])
	except:
		r=0.0
	ratio.append([i[0],(i[1]+won['count'][ind]),r])
for i in np.array(won):
	if i[0] not in np.array(lose['Lost']):
		ratio.append([i[0],i[1],100])

ratio = pd.DataFrame(ratio,columns=['TeamID','TotalMatch','Won%']).sort_values('Won%',ascending=False)
ratio[ratio['TotalMatch'] > 100].head(20)

