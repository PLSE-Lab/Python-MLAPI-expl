#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv(r"../input/matches.csv")


# In[ ]:


#display the ipl data
data


# In[ ]:


data2008 = data[data['season'] == 2008]
data2009 = data[data['season'] == 2009]
data2010 = data[data['season'] == 2010]
data2011 = data[data['season'] == 2011]
data2012 = data[data['season'] == 2012]
data2013=data[data['season'] == 2013]
data2014=data[data['season'] == 2014]
data2015=data[data['season'] == 2015]
data2016=data[data['season'] == 2016]


# <p style="font-family: Arial; font-size:2.75em;color:purple; font-style:bold"><br>
# Data Cleaning</p><br>

# In[ ]:


#data.info()
#data2014.iloc[-4].winner
#data2014.describe()
#data2014.info()
#data2014.season.count() - used to count the number of matches in 2014 season.
data.info()


# In[ ]:


before_rows = data.shape
print(before_rows)


# In[ ]:


del data['umpire3']
data.dropna(inplace = True)


# In[ ]:


after_rows = data.shape
print(after_rows)
#decreased column number because of the umpire3 column.


# In[ ]:


data.isnull().any()
#no null values due to 'dropna' 


# <p style="font-family: Arial; font-size:2.75em;color:purple; font-style:bold"><br>
# Analysis</p><br>

# In[ ]:


df = data2009[data2009['winner'] == 'Delhi Daredevils']
len(df)
win_csk = data2016[data2016['winner'] == 'Chennai Super Kings']
len(win_csk)
#len(df)
#data2012
## The value will be 0 as CSK didnt feature in CSK
df


# In[ ]:


data2010.iloc[0].winner


# In[ ]:


data2011.iloc[-1].winner


# In[ ]:


(data2008.iloc[-1].winner ,data2009.iloc[-1].winner ,data2010.iloc[-1].winner , data2011.iloc[-1].winner,data2012.iloc[-1].winner,data2013.iloc[-1].winner,data2014.iloc[-1].winner, data2015.iloc[-1].winner , data2016.iloc[-1].winner)
##the winner as a tuple


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Percentage of Matches won by teams
# </p>

# In[ ]:



#(data.city.value_counts(normalize=True) *100).plot(kind='pie',title='Percentage Of Matches Played in Cities(All Seasons)',figsize=(12,8))
(data.winner.value_counts(normalize =True)*100).plot(kind = 'barh' , title='Percentage of matches won by Teams',figsize = (20,10))


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Percentage of toss decisions
# </p>

# In[ ]:


#Percentage of toss decisions
(data.toss_decision.value_counts(normalize=True)*100).plot(kind='barh',title='Percentage of toss decisions(All Seasons)')


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# Maximum Man Of the Match
# </p>

# In[ ]:



data2015.player_of_match.value_counts().head().plot(kind='barh',title="Top Players Become max times--\'Man of The Match'",grid=True) #No of the man of the match per player
#data2016.player_of_match.value_counts().head().plot(kind='barh',title="Top Players Become max times--\'Man of The Match'",grid=True) #No of the man of the match per player


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Favourite Venue
# </p>

# In[ ]:


#Number of Maximum Choosen Venue
venue = data.venue.value_counts()
data.venue.value_counts().plot(kind='bar',title='Fav Grounds' , figsize=(15,10) , grid =(15,10) ).legend(bbox_to_anchor=(1.2, 0.5))


# In[ ]:


#Winning Percent of teams at Chinnaswamy Stadium
chinna = data[data.venue=='M Chinnaswamy Stadium']['winner']
chinna.value_counts()
#chinna_win = data[data.venue=='M Chinnaswamy Stadium']['winner'].value_counts(normalize=True)*100
#chinna_win.plot(kind = 'line' , title='winning percent at Chinnaswamy' , figsize = (15,10) , grid=True).legend(bbox_to_anchor=(1.2, 0.5))

#df = data[data.venue=='M Chinnaswamy Stadium']['winner'].value_counts(normalize=True)*100
#df.plot(kind = 'barh' , title = 'winning percent of teams at Chinnaswamy' , figsize = (10,10) , grid = True)


# In[ ]:


#print (data.win_by_runs.mean() ) #Average win by runs
#print (data.win_by_wickets.mean()  ) #Average win by wicket
#data.describe()
#data.win_by_wickets.mean()
#data.win_by_runs.mean()
data.info()


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Winning w.r.t toss decisions and various miscllaneous analysis
# </p>

# In[ ]:


#pd.crosstab(data.winner,data.season)
pd.crosstab( data2011.winner , data2011.toss_winner ).plot(kind = 'bar' , title = 'toss winner in each match in 2011 season' , figsize=(10,10))


# In[ ]:


pd.crosstab(data.winner,data.toss_decision).plot(kind='bar',title='Winning w.r.t toss decisions overall')


# In[ ]:


pd.crosstab(data2016.season,data2016.player_of_match).plot(kind='bar', title='Player of match in 2016 ').legend(bbox_to_anchor=(1.2, 0.5))


# In[ ]:


pd.crosstab(data2010.winner,data2010.toss_decision).plot(kind='bar',title='Winning w.r.t toss decisions in 2010')


# In[ ]:


pd.crosstab(data2011.winner,data2011.toss_decision).plot(kind='bar',title='Winning w.r.t toss decisions in 2011')


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Winning w.r.t Cities
# </p>

# In[ ]:


pd.crosstab(data2008.winner,data2008.city)


# In[ ]:


pd.crosstab(data2008.winner,data2008.city).plot(kind='bar',title='Winning w.r.t cities in 2008',figsize=(10,8))


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Favourite Umpires
# </p>

# In[ ]:


fav_umpire=data.umpire1.value_counts().head(10)
#fav_umpire.plot(kind = 'barh')
fav_umpire


# In[ ]:


#From Seaborn
plt.subplots(figsize=(8,5))
sns.barplot(x=fav_umpire.values,y=fav_umpire.index,palette="Blues_d")


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Maximum Toss Winners
# </p>

# In[ ]:


data.toss_winner.value_counts()


# In[ ]:


#Normal plot
data.toss_winner.value_counts().plot(kind='bar')


# In[ ]:


#By seaborn
plt.subplots(figsize=(8,5))
sns.barplot(x=data.toss_winner.value_counts().values,y=data.toss_winner.value_counts().index)


# **Thats it folks :D**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




