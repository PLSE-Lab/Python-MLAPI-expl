#!/usr/bin/env python
# coding: utf-8

# ## Analysing Olympic Data

# #### Introduction:

# The dataset consist of details on the olympics held from 1896 to 2016.This dataset provides an opportunity to deep dive into analysis of various cavets of olympics history.Some of the points which I will consider for analysis include the countries participation,age distribution based on the game,medals tally,gold winners and more ...Lets begin.

# #### Importing the necessary libraries:

# In[ ]:


import numpy as np #Linear algebra
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
warnings.filterwarnings('ignore')


# #### Reading the dataset:

# In[ ]:


games=pd.read_csv('../input/athlete_events.csv')
noc=pd.read_csv('../input/noc_regions.csv')


# In[ ]:


## Glimpse of the data:
games.head()


# The dataset consist of olympic data ( both summer and winter olympics) starting from 1896.A quick google search on summer and winter olympics states that summer olympics is a much bigger event ( called The Olympics) started in 1896 and was first held in Athens,Greece whereas the winter olympics started from 1924 first held in Chamonix,France.
# 

# In[ ]:


noc.head()


# The dataset consist of listings of participants in the olympics with gender,age,height,weight,game participated and the medal won.The NOC refers to National Olympics Committe's code for each of the region.Lets do a quality check first to see the number of missing values in the dataset.

# In[ ]:


print(games.isnull().any())


# We find that Age,Height,Weight and Medal has missing information in them.

# #### Year and Team Count:

# Lets see the trend of the team participation over the olympic run since 1896.

# In[ ]:


team=games.groupby(['Year'])['Team'].nunique().reset_index()
team.rename({'Team':'Team_Count'},inplace=True,axis=1)
team.head()


# In[ ]:


trace = go.Scatter(
                x=team['Year'],
                y=team['Team_Count'],
                name = "Team Participation in Olympics",
                line = dict(color = '#17BECF'),
                opacity = 0.8,
                mode="lines+markers"
                )

data = [trace]

layout = dict(
    title = "Team Participation in Olympics(Both Summer and Winter )",
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Team Participation in Olympics")


# The number of teams to represent olympics has hovered around 110 to 260 mark.There were 292 teams representing the olympics during the year 2008 while the year 1906 has seen the lowest representation - 52 teams.

# ###  Distribution of Age,Height and Weight:

# In[ ]:


plt.figure(figsize=(8,8))
plt.subplot(311)
ax=sns.distplot(games['Age'].dropna(),color='blue',kde=True)
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('Age Distribution of Sportspersons',fontsize=16,fontweight=200)
plt.subplot(312)
ax1=sns.distplot(games['Height'].dropna(),color='Red',kde=True)
ax1.set_xlabel('Height')
ax1.set_ylabel('Density')
ax1.set_title('Height Distribution of Sportspersons',fontsize=16,fontweight=200)
plt.subplot(313)
ax2=sns.distplot(games['Weight'].dropna(),color='green',kde=True)
ax2.set_xlabel('Weight')
ax2.set_ylabel('Density')
ax2.set_title('Weight Distribution of Sportspersons',fontsize=16,fontweight=200)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)


# * It is seen that all the three distributions are normal.
# * The mean of age centered around 26 years.The distribution can be called as bimodal since there are two peaks - one round 22 years and the other around 30 years.
# * The height distribution is a perfect bell shaped curve with the mean at the center.The mean and median height is around 175 . 
# * It can be said that by looking at the weight distribution it is skewed to the right.The mean value is aroung 65.
# 
# Now let us look at the actual values to know if our interpretations are correct.

# In[ ]:


games.drop(['ID','Year'],axis=1).describe()


# Our assumptions are indeed correct.

# #### Olympic Cities :

# Let us know about the cities where the olympics have taken place since 1896.

# In[ ]:


Year=games.groupby('City').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Year.columns=['City','Years']
Year['Count']=[len(c) for c in Year['Years']]


# In[ ]:


Year.sort_values('Count',ascending=False)


# It is seen that Olympic events has taken place thrice in the city of Athina,London whereas twice the games have been held at Sankt Moritz,Paris,Stockholm,Los Angeles,Lake Placid,Innsbruck.

# #### How many games ?

# Let us see how many sports have been included over the years in olympics.

# In[ ]:


sports=games.groupby('Year').Sport.nunique().to_frame().reset_index()


# In[ ]:


sports.columns=['Year','Count of Sport']


# In[ ]:


trace1 = go.Scatter(
                x=sports['Year'],
                y=sports['Count of Sport'],
                name = "Sports in Olympics",
                line = dict(color = '#17BECD'),
                opacity = 0.8,
                mode="lines+markers"
                )

data1 = [trace1]

layout1 = dict(
    title = "Representation of Sports (Count) in Olympics(Summer and Winter)",
)

fig = dict(data=data1, layout=layout1)
py.iplot(fig, filename = "Sport Count")


# Though it is easy to conclude from the graph that the count of sport is fluctuation it will give more sense if we consider Summer and Winter olympic sporting separately.

# #### Summer Olympic Sports:

# In[ ]:


sports=games.groupby(['Year','Season']).Sport.nunique().to_frame().reset_index()


# In[ ]:


plt.figure(figsize=(10,10))
ax=sns.pointplot(x=sports['Year'],y=sports['Sport'],hue=sports['Season'],dodge=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Year',fontsize=10)
ax.set_ylabel('Count',fontsize=10)
ax.set_title('Sports in Olympics',fontsize=16)


# #### Sport in Olympics:

# Having understood the count of sport in olympics lets see how many sports have been there since inception and which are relatively new sports.We will use an analysis similar to olympic cities.

# In[ ]:


### Sport in Summer Olympics:
summer_olympic=games[games['Season']=='Summer']
Sport_Count=summer_olympic.groupby('Sport').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Sport_Count.columns=['Sport','Years']
Sport_Count['Count']=[len(c) for c in Sport_Count['Years']]


# In[ ]:


Sport_Count['Years']=pd.Series(Sport_Count['Years'])
Sport_Count['Years']=Sport_Count['Years'].apply(lambda x:sorted(x))  ### Sort Year in ascending order inside the Year column.


# In[ ]:


Sport_Count.sort_values('Count',ascending=False,inplace=True)
Sport_Count


# Gymnastics,Swimming,Fencing,Athletics,Cycling have been there since inception from 1896 whereas Rugby Sevens is a relatively new sport included in 2016 olympics.
# 
# Lets carry out similar analysis for winter olympics.

# In[ ]:


### Sport in Winter Olympics:
Winter_olympic=games[games['Season']=='Winter']
Winter_Count=Winter_olympic.groupby('Sport').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Winter_Count.columns=['Sport','Years']
Winter_Count['Count']=[len(c) for c in Winter_Count['Years']]


# In[ ]:


#Winter_Count['Years']=pd.Series(Winter_Count['Years'])
Winter_Count['Years']=Winter_Count['Years'].apply(lambda x:sorted(x))


# In[ ]:


Winter_Count.sort_values('Count',ascending=False)


# * Ice Hockey,Figure Skating,Ski Jumping,Nordic Combined,Speed Skating ,Cross Country Skiing and Bobsleigh have been there since inception whereas snowboarding is a new sport which has been there since 1998.
# * Thre are two sports - Alpinish amd Military Ski patrol which has been played in winter olympics only once .

# #### Overall Medals Tally:

# Let us visualise the  all time medals tally of countries 

# In[ ]:


game_noc=pd.merge(games,noc,how='left',on='NOC')
game_noc.drop_duplicates(inplace=True,keep=False)


# In[ ]:


medal=game_noc.groupby(['region','Medal'])['Medal'].count()
medal=medal.unstack(level=-1,fill_value=0).reset_index()
medal.head()


# In[ ]:


medal['Total']=medal['Bronze']+medal['Gold']+medal['Silver']
total_games=game_noc.groupby('region')['Sport'].nunique().to_frame().reset_index()
total_games.rename({'Sport':'TotalGames'},inplace=True,axis=1)
#total_games.head()


# In[ ]:


medal=pd.merge(medal,total_games,how='left',on='region')
medal.sort_values('Total',ascending=False,inplace=True)


# In[ ]:


medal=medal[['region','TotalGames','Gold','Silver','Bronze','Total']]  ### Reordering the columns
medal.head(10)


# The above table shows the overall medals tally of the countries participated in the Olympics from 1896 to 2016.USA leads the overall medals tally with 5637 medals followed by Russia and Germany.The difference between the medal count for first and second place is too high -aroung 2000 medals.Lets visualise this with the bar plot.

# In[ ]:


plt.figure(figsize=(10,10))
ax=sns.barplot(medal['region'].head(10),medal['Total'].head(10),palette=sns.color_palette('Set1',10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Region',fontsize=10)
ax.set_ylabel('Total Medals',fontsize=10)
ax.set_title('Total Medal Count of Top 10 Countries in Olympics')


# #### Medals Analysis for USA

# Let us now see the Olympic Champions as well as the sports who have contributed to the medals tally for USA.

# In[ ]:


USA_Gold=game_noc[(game_noc['region']=='USA') & (game_noc['Medal']=='Gold')]
champ=USA_Gold.groupby('Sport').size().to_frame().reset_index()
champ.columns=['Sport','Count']


# In[ ]:


champ.sort_values(by='Count',ascending=False,inplace=True)


# In[ ]:


plt.figure(figsize=(10,10))
ax=sns.barplot(champ['Sport'].head(10),champ['Count'].head(10),palette=sns.color_palette('viridis_r',10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Count',fontsize=10)
ax.set_title('Games where USA has won Gold maximum times')


# USA has won maximum gold from Swimming,Athletics,Basketball.
# 
# Lets see who has won maximum gold and in which sport for USA.

# In[ ]:


champ=USA_Gold.groupby(['Name','Sport']).size().to_frame().reset_index()


# In[ ]:


champ.columns=['Name','Sport','Golds']
#champ.head()
champ.sort_values(by='Golds',ascending=False,inplace=True)
champ.head(10)


# Michael Phelps has won **Gold** 23 times followed by Raymold Clarence Ray Ewry 10 times in Athletics.

# Lets see how many events has Phelps participated.

# In[ ]:


Phelps=game_noc[(game_noc['Name']=='Michael Fred Phelps, II' ) & (game_noc['Medal']=='Gold')]
print("Swimming Event where Phelps has won Gold\n",Phelps['Event'].unique)


# How many olympic events has Phelps participated ?

# In[ ]:


game_noc[game_noc['Name']=='Michael Fred Phelps, II'].Year.nunique()


# Thus we see that Phelps has raked up 23 gold's in 5 years of his olympic stint.

# #### Gold Hunters

# In[ ]:


gold=game_noc.loc[game_noc['Medal']=='Gold'].groupby(['Name','Sport','region']).size().to_frame().reset_index()
gold.columns=['Name','Sport','region','count']
gold.sort_values('count',ascending=False,inplace=True)
gold.head(10)


# Phelps tops this list too and leads with a record breaking 23 gold followed by Raymold with 10 medals.

# #### Height,Weight and Age according to sports:

# The following block of analysis is inspired by comments from Andreas Stockl .Thanks Andreas.

# In[ ]:


### Considering only sports that were played from inception so that we have a good comparion with lot of data !!!!.
sport_box=game_noc[game_noc['Sport'].isin(Sport_Count.Sport[:10])]


# In[ ]:


plt.figure(figsize=(10,8))
plt.subplot(311)
ax=sns.boxplot(x='Sport',y='Age',data=sport_box,palette=sns.color_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Age',fontsize=10)
ax.set_title('Age distribution across sports',fontsize=16)
plt.subplot(312)
ax=sns.boxplot(x='Sport',y='Height',data=sport_box,palette=sns.color_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Height',fontsize=10)
ax.set_title('Height distribution across sports',fontsize=16)
plt.subplot(313)
ax=sns.boxplot(x='Sport',y='Weight',data=sport_box,palette=sns.color_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Weight',fontsize=10)
ax.set_title('Weight distribution across sports',fontsize=16)
plt.subplots_adjust(wspace = 1, hspace = 1,top = 1.3)


# * There is a difference between height,weight and age for every sports.
# 
# * In the age chart,we see that there are many outliers and there were persons of age more than 60 years.Really ???
# 
# * Rowing has sportspersons with less than 20 years of years.
# 
# * The median height for Water polo and Rowing is higher followed by Athletics and swimming.
# 
# * The maximum height has been for Wrestling !!!
# 
# * For weight,Athletics seems to have many outliers.while Wrestling has sportspersons with weight of more than 150 kgs which is perfectly understood.

# #### Conclusion:

# Woah !!! What a fun it has been analysing this dataset.The data provided a blend of both numeric and factorial variables that could be used to find the relationships between the variables and narrow down our answers.Though I stop here ,there are many areas which could be drilled down futher.
# 
# I hope that my kernel provided an interesting viewpoint on this dataset and you were able to pick up few ideas from this for your analysis.Thank you for reading.
# 
# **If you find it interesting,pls comment/upvote.**Thanks again...
