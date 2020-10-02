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


# Clash Royale - Data Wrangling
# 
# The raw data is in json format and we need to convert them in dataframe for analysis purposes. Let us use the json packages to convert them into dataframes.

# In[ ]:


#Source data is in json format need to convert them to dataframe for further analysis
import json
from pandas.io.json import json_normalize


# In[ ]:


with open('../input/matches.txt') as file:
    CR = [ x.strip() for x in file.readlines()]


# In[ ]:


len(CR)


# There are more than 700k records in the document which is huge data to start our Clash Royale analysis. Thus we are going to only 10k records from the source and do our analysis.

# In[ ]:


deserialize_cr =[json_normalize(eval(r1))for r1 in CR[0:10000]]
df_cr =pd.concat(deserialize_cr,ignore_index=True)
df_cr.columns = ['Left Clan','Left Deck','Left Player','Left Trophy','Right Clan','Right Deck','Right Player','Right Trophy','Result','Time','Type']
df_cr.head()


# From the above data we are able to see that it contains the Left and right deck troops used to attack one another. Let us try to extract some insight from the deck columns and see what it actually has.

# In[ ]:


LD = [len(left_deck) for left_deck in df_cr['Left Deck']]
RD = [len(right_deck) for right_deck in df_cr['Right Deck']]
(set(LD),set(RD))


# The above result clearly tells us that every battle goes with 8 different types of troops by both sides.

# In[ ]:


Left_Troops =  list(np.hstack([[x[0] for x in left_deck] for left_deck in df_cr['Left Deck']]))
Right_Troops = list(np.hstack([[x[0] for x in right_deck] for right_deck in df_cr['Right Deck']]))
distinct_troops = set(np.hstack([Left_Troops,Right_Troops]))
len(distinct_troops)


# From the above results,we are able to conclude that there are 74 different troops that can be used in a battle.
# 
# Now let us break the deck list and merge it to the existing dataframe.

# In[ ]:


RightArmy_colNames = np.hstack([["Right Troop "+str(i+1) for i in range(8)],["Right Troop Count "+str(i+1) for i in range(8)]])
LeftArmy_colNames = np.hstack([["Left Troop "+str(i+1) for i in range(8)],["Left Troop Count "+str(i+1) for i in range(8)]])
RightArmy = pd.DataFrame(data=[np.hstack([[army[0] for army in x],[int(army[1]) for army in x]]) for x in df_cr['Right Deck']],columns = RightArmy_colNames)
LeftArmy = pd.DataFrame(data=[np.hstack([[army[0] for army in x],[int(army[1]) for army in x]]) for x in df_cr['Left Deck']],columns=LeftArmy_colNames)
finalCR_data = pd.concat([df_cr,LeftArmy,RightArmy],axis=1,join='inner')
finalCR_data.head(2)


# The Result column seem to have the stars won by each side and it is present as a list. Let us have that information individually for our analysis.

# In[ ]:


finalCR_data['Left Crowns Won'] = [int(stars[0]) for stars in finalCR_data['Result']]
finalCR_data['Right Crowns Won'] = [int(stars[1]) for stars in finalCR_data['Result']]


# In[ ]:


finalCR_data.head()


# Based on the star wons by each side we can decide who won the match and are there any tie possibility ?

# In[ ]:


finalCR_data['Battle Result'] = [ 'Left' if(left > right) else 'Right' if(left<right) else 'Tie' for left,right in zip(finalCR_data['Left Crowns Won'],finalCR_data['Right Crowns Won'])]


# Let us find out the Percentage of wins by each side.

# In[ ]:


finalCR_data[['Result','Battle Result']].groupby('Battle Result').count().apply(lambda x:(x/x.sum())*100)


# From the Result set - 5.2 % battles were drawn and most battles were won by Left players.
# 
# Let us find the top 5 winning Left  & Right battle strategies

# In[ ]:


finalCR_data[(finalCR_data['Battle Result']=='Left')][['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8','Result']].groupby(['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8']).count().sort_values(by='Result',ascending=False).head(1)


# In[ ]:


finalCR_data[(finalCR_data['Battle Result']=='Right')][['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8','Result']].groupby(['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8']).count().sort_values(by='Result',ascending=False).head(1)


# Irrespective of the sides the best winning strategy seem to be the same ! Folks who are playing Clash Royale, please note this point and troops to be used for battles.
# Hog Rider - Tornado - Princess - The Log - knight - Rocket - Goblins - Ice Spirit
# Yes, Of Course you need to have the troop levels maxed out :)
# 
# Now Lets see the strategies used by left and right sides to secure a three crown win !

# In[ ]:


finalCR_data[(finalCR_data['Battle Result']=='Left') & (finalCR_data['Left Crowns Won']==3)][['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8','Result']].groupby(['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8']).count().sort_values(by='Result',ascending=False).head(1)


# In[ ]:


finalCR_data[(finalCR_data['Battle Result']=='Right') & (finalCR_data['Right Crowns Won']==3)][['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8','Result']].groupby(['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8']).count().sort_values(by='Result',ascending=False).head(1)


# So both sides follow a different strategy to get a three crown win.
# 
# We shall dig deeper into this data and draw some visual insights.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


matches = finalCR_data[['Left Clan','Type']].groupby('Type',as_index=False).count()
sns.set_color_codes("muted")
sns.barplot(x="Type",y="Left Clan",data=matches,color="b")


# Out of the three types of battles, users love to play the ladder type battles than Challenges & Tourneys.
# 
# Let us categorize the players by their trophy level.

# In[ ]:


finalCR_data[['Left Trophy','Right Trophy']] = finalCR_data[['Left Trophy','Right Trophy']].apply(pd.to_numeric)
finalCR_data['Left Category'] = pd.cut(finalCR_data['Left Trophy'],5)
finalCR_data['Right Category'] = pd.cut(finalCR_data['Right Trophy'],5)
finalCR_data[['Left Category']].drop_duplicates().sort_values(by=['Left Category'])


# In[ ]:


finalCR_data[['Right Category']].drop_duplicates().sort_values(by=['Right Category'])


# In[ ]:


player_categories = finalCR_data[['Left Category','Right Category','Result','Type']].groupby(['Right Category','Left Category','Type'],as_index=False).count().sort_values(by=['Left Category','Right Category'],ascending=True)
#player_categories
graph = sns.FacetGrid(player_categories,row='Left Category',col='Type',size=3.0,aspect =2.5,sharey=False)
graph.map(sns.barplot,'Right Category','Result',color='b',ci=None)


# The Above graph shows us that the 'Ladder' battles played between users with similar range of trophies  but it is not  for Challenge and Tourney Battles. 
# Have a look at top left chart, the left users with less than 1083 trophies have challenged only against right users greater than 1086 trophies. 3rd chart in second row, left players between 1086 - 2170 trophies battled against right players more than 3000 trophies only
# . Good insight right ?.

# Let us see how many players have defended well and managed to win with three crowns

# In[ ]:


threecrowns = finalCR_data[(finalCR_data['Left Crowns Won']-finalCR_data['Right Crowns Won']==3) | (finalCR_data['Left Crowns Won']-finalCR_data['Right Crowns Won']==-3)][['Battle Result','Type','Result']].groupby(['Battle Result','Type'],as_index=False).count().sort_values(by='Type',ascending=True)
histgraph = sns.FacetGrid(threecrowns,col='Type',size=2.7,aspect=1.7,sharey=False)
histgraph.map(sns.barplot,'Battle Result','Result')


# Left players seem to dominate right players in all type of battles.

# More Analysis on this data to come in upcoming versions. I appreciate your feedback on my work.
