#!/usr/bin/env python
# coding: utf-8

# This kernel is mainly focused on exploring the given Video Games Dataset. \
# It started with no particular purpose and subsequent analysis of the Sport Genre was done since it seemed to be most interesting with its dramatic changes ver the time.
# 
# Any corrections or comments would be highly appreciated)*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#importing data
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')


# # Preproccesing

# In[ ]:


df.info()


# There are 11 columns\
# Types of data:
# * 1 integer - Rank
# * 4 objects(str) - The names of the games, platform, publisher and genre
# * 6 floats(one of the is Year, which will be converted into int later on)

# ## Looking for the empty values

# In[ ]:


df.isnull().sum()


# In[ ]:


#use missingno to graphicaly observe the missing data(optional)
msno.matrix(df)


# There are 2 columns with empty values: Year(271 missing) and Publisher(58 missing), total=329. \
# Considering the total size of 16598 rows, missing values cover less then 2% of all the dataset, so we can drop them without any significant loss

# ## Dropping empty values

# In[ ]:


#since the % of empty data is small we drop them
empty_data = np.where(pd.isnull(df))
df = df.drop(empty_data[0])
df.isnull().sum()


# Dropping Rank since it has no prediction value and it's seems like was generated after sorting the data based on Sales

# In[ ]:


df = df.drop(['Rank'], axis=1).reset_index()
df = df.drop(['index'], axis=1)


# For the future manipulation converting float type Year column into integer

# In[ ]:


df['Year']=df['Year'].astype(int)


# In[ ]:


#Saving changes in df into df1 in order to be able to restore the changes when needed
df1=df.copy()


# In[ ]:


df1


# # Analysis

# ## Illustrating graphically Sales per Genre

# In[ ]:


genre_sum =df1.groupby(['Genre']).agg({'sum'}).reset_index()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(genre_sum['Genre'],
        genre_sum['NA_Sales'],
        color='r',
        label='NA_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['EU_Sales'],
        color='b',
        label='EU_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['JP_Sales'],
        color='gainsboro',
        label='JP_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['Other_Sales'],
        color='k',
        label='Other_Sales');

ax.set_title('Sales per Genre except JP')
plt.xlabel('Genre')
plt.ylabel('Sales (M)')
ax.legend();


# The reason for not including Japan Sales is that now with the rest 3 we can obserbe similar trends and patterns. \
# The only difference is in Platform Sales where the North America Sales deviate from other 2 with its increase relative to other genres.

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(genre_sum['Genre'],
        genre_sum['NA_Sales'],
        color='grey',
        label='NA_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['EU_Sales'],
        color='gainsboro',
        label='EU_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['JP_Sales'],
        color='r',
        label='JP_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['Other_Sales'],
        color='gainsboro',
        label='Other_Sales');

ax.set_title('Sales per Genre JP')
plt.xlabel('Genre')
plt.ylabel('Sales (M)')
ax.legend();


# From the previous graph we can see that North America, Europe and Other sales are roughly identical, in their trends. \
# We can easily declare that Action genre is the most popular. The top genres in NA, EU and Other are:\
# 1) Action\
# 2) Sports\
# 3) Shooter\
# 4) Platform/Misc\
# 5) Role-Playing\
# When in JP, the most sold game genre is Role-Playing which is greater then even North America Sales(which sales were dominating in all other genres in all other markets)\
# Additionally JP has the least sales for the Shooter and Racing genres, as for the rest genres the sales distribution is quite the same as the rest of the world.

# ## Sales of each market by year

# In[ ]:


year_sum =df1.groupby(['Year']).agg({'sum'}).reset_index()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(year_sum['Year'],
        year_sum['NA_Sales'],
        color='r',
        label='NA_Sales');
ax.plot(year_sum['Year'],
        year_sum['EU_Sales'],
        color='b',
        label='EU_Sales');
ax.plot(year_sum['Year'],
        year_sum['JP_Sales'],
        color='g',
        label='JP_Sales');
ax.plot(year_sum['Year'],
        year_sum['Other_Sales'],
        color='k',
        label='Other_Sales');

ax.set_title('Sales per Realese Year')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();


# The most sales within time range 1980-2016 was 2009.
# Also we can tell that averall trend is that more and more people were buying video games untill 2009, then all markets had significant decreas.(it maybe due to the incoplete data)\
# In the next graphs will try to analize the reason of a drastic increase and later decrease of sales*

# In order to get understanding of rise and fall of the industry. \
# Let's compare these sales with the number of games published in 1980-2020.\
# data was taken from https://en.wikipedia.org/wiki/Category:Video_games_by_year

# In[ ]:


data_games = {'Year':list(range(1980, 2021))}
games_count = pd.DataFrame(data_games, columns = ['Year'])
games_published=[119,154,280,382,324,306, 381,439,397,483,590,637,668,370,735,793,672,645,655,695,700,691,709,718,679,792,857,899,983,939,814,798,779,660,606,586,570,515,390,342,83]
gc=games_count['Year']
for i in gc:
    games_count['Games_published']=games_published

games_count['Sales']=year_sum['Global_Sales']


# Lest normalize the sales and number of published games to be able to compare them next to each other

# In[ ]:


games_count = games_count.dropna()
games_count
games_count["Games_published"] = games_count["Games_published"] / games_count["Games_published"].max()
games_count["Sales"] = games_count["Sales"] / games_count["Sales"].max()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(games_count['Year'],
        games_count['Sales'],
        color='g',
        label='Sales');
ax.plot(games_count['Year'],
        games_count['Games_published'],
        color='r',
        label='Published Games');
ax.set_title('Sales vs Realese per Year')
plt.xlabel('Release Year')
plt.ylabel('Q')
ax.legend();


# Even with this rough next to each other comparison we can see the certain correlation(not necessarilly causation), but the reason for the peak in 2009 is still unknown, since there are more underlying factors(economical, technological etc.) to be considered for better understainding and prediction. 

# ## Top sold video games

# We will take first two words in the name in order to later group the games with their sequels, than we can pivot sales around each game to see how much $ each of game series earned

# In[ ]:


df2 = df1.copy()
df2=df2.sort_values(by='Global_Sales', ascending = False)

df3 = df2.copy()

short_names=df3['Name'].apply(lambda x: ' '.join(x.split()[:3]))
df3['Name']=short_names
for i in df3['Name']:
    df3['count'] = 1
    
df3=df3.groupby(['Name']).agg({'sum'}).copy()

df4=df3.copy()


# In[ ]:


columns=['Platform', 'Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','count']
df4.columns=columns
df4['Game Tilte']=df4.index
df4['Global_Sales']=df4['Global_Sales'].round(decimals=2)


# In[ ]:


large=df4.nlargest(10,['Global_Sales'],keep='first') 
fig = plt.figure(figsize=(15,6))
ax = fig.add_axes([0,0,1,1])
game_title = large['Game Tilte']
sales = large['Global_Sales']
ax.bar(game_title,sales)
xlabel=('Games Title'),
ylabel=('Sales (M)')
ax.set_title('Top 10 sold Game Series')
# Make some labels.
rects = ax.patches
labels = sales

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, label,
            ha='center', va='bottom')

plt.show()


# The graph above shows us the total sales of the top sold Game Series.\
# From the first glance we can see very high sales of Call of Duty relative to others. \
# After a close examination we can see that there are duplicates of the same game that were not combined:\
# 1) Super Mario series \
# 2) Need for Speed\
# 3) Grand Theft Auto\
# Also we have a 3rd position held by "The Legenf of". Since it is very popular begining of the game title we will closely exemine if it belong to 1 game series.

# In[ ]:


legend_cols = [col for col in df['Name'] if 'The Legend of' in col]
print(legend_cols)


# Too much titles starting from 'The Legend of.." , we have to it drop to not get confused.\
# Also we are going to combine the values of the duplicating game series.

# In[ ]:


large=large.drop(large.index[2])
large = large.rename(index={'Call of Duty:': 'Call of Duty', 'Grand Theft Auto:':'Grand Theft Auto', 'Super Mario Bros.':'Super Mario',                             'New Super Mario':'Super Mario', 'Need for Speed:':'Need for Speed' })
large2 =large.groupby(['Name']).agg({'sum'})
large3 = large2.copy()
large3.columns=['Platform', 'Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','count', 'Game_Title']
large3 = large3.sort_values(by='Global_Sales', ascending=False)
large3['Game_Title'] = large3.index


# In[ ]:


fig = plt.figure(figsize=(15,6))
ax = fig.add_axes([0,0,1,1])
game_title = large3['Game_Title']
sales = large3['Global_Sales']
ax.bar(game_title,sales)
xlabel=('Games Title'),
ylabel=('Sales (M)')
ax.set_title('Top 6 sold Game Series')
# Make some labels.
rects = ax.patches
labels = sales

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, label,
            ha='center', va='bottom')

plt.show()


# Here we can see the 6 top sold game series.\
# Next, let's see the contribution of each market on these sales.

# In[ ]:


fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(large3['Game_Title'],
        large3['NA_Sales'],
        color='r',
        label='NA_Sales');
ax.plot(large3['Game_Title'],
        large3['EU_Sales'],
        color='c',
        label='EU_Sales');
ax.plot(large3['Game_Title'],
        large3['JP_Sales'],
        color='b',
        label='JP_Sales');
ax.plot(large3['Game_Title'],
        large3['Other_Sales'],
        color='k',
        label='Other_Sales');

ax.set_title('Game Series Sales per Market')
plt.xlabel('Game Series Title')
plt.ylabel('Sales (M)')
ax.legend();


# As we expected the North America market contributes the most among others.\
# Pretty interesting that in Super Mario series Japan market is close to the European, which is in fact the peak of its sales comparing to other games series.

# Now there's a reasonable argument that the total sales depend on the quantity of games published. So in order to measure more presizely we can look at average sales per 1 game published.

# In[ ]:


large3['Sales_per1'] = large3['Global_Sales']/large3['count']
large4= large3.sort_values(by='Sales_per1', ascending=False).copy()
fig = plt.figure(figsize=(15,6))
ax = fig.add_axes([0,0,1,1])
game_title = large4['Game_Title']
sales = large4['Sales_per1']
ax.bar(game_title,sales)
xlabel=('Games Title'),
ylabel=('Sales (M)')
ax.set_title('Top Average Sales per 1 Game from series')
# Make some labels.
rects = ax.patches
labels = sales

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
            ha='center', va='bottom')

plt.show()


# Now we can see how much average sales was generated by 1 game from series. \
# The highes is Wii Sports that had 82.74 mln global sales. Next goes Super Mario and Super Smash Bros.
# 
# Even after so many manipulation to clear up the results we should consider the fact that Super Mario Series were firstly published earlier than all the others so it had "more" time to generate the sales. \
# Decide yourself if it's a significant measure to include in analysis or not. Even if we include the time since first publish in to the equation the Wii Sports will stay the same since all the sales were genereted in one year(see in further analysis)

# ## Our next analysis is on Yearly sales per Genre

# Creating subsets for each genre

# In[ ]:


Shooter = df1[df1.Genre == 'Shooter']
Shooter =Shooter.groupby(['Year']).agg({'sum'}).reset_index()
Shooter=Shooter[['Year','Name' ,'Global_Sales']]

Misc = df1[df1.Genre == 'Misc']
Misc =Misc.groupby(['Year']).agg({'sum'}).reset_index()
Misc=Misc[['Year','Name' , 'Global_Sales']]

Action = df1[df1.Genre == 'Action']
Action =Action.groupby(['Year']).agg({'sum'}).reset_index()
Action=Action[['Year','Name' , 'Global_Sales']]

Sports = df1[df1.Genre == 'Sports']
Sports =Sports.groupby(['Year']).agg({'sum'}).reset_index()
Sports=Sports[['Year','Name' , 'Global_Sales']]

Fighting = df1[df1.Genre == 'Fighting']
Fighting =Fighting.groupby(['Year']).agg({'sum'}).reset_index()
Fighting=Fighting[['Year','Name' , 'Global_Sales']]

Puzzle = df1[df1.Genre == 'Puzzle']
Puzzle =Puzzle.groupby(['Year']).agg({'sum'}).reset_index()
Puzzle=Puzzle[['Year','Name' , 'Global_Sales']]

Racing = df1[df1.Genre == 'Racing']
Racing =Racing.groupby(['Year']).agg({'sum'}).reset_index()
Racing=Racing[['Year','Name' , 'Global_Sales']]

Platform = df1[df1.Genre == 'Platform']
Platform =Platform.groupby(['Year']).agg({'sum'}).reset_index()
Platform=Platform[['Year','Name' , 'Global_Sales']]

Simulation = df1[df1.Genre == 'Simulation']
Simulation =Simulation.groupby(['Year']).agg({'sum'}).reset_index()
Simulation=Simulation[['Year','Name' , 'Global_Sales']]

Adventure = df1[df1.Genre == 'Adventure']
Adventure =Adventure.groupby(['Year']).agg({'sum'}).reset_index()
Adventure=Adventure[['Year','Name' , 'Global_Sales']]

Role_Playing = df1[df1.Genre == 'Role-Playing']
Role_Playing =Role_Playing.groupby(['Year']).agg({'sum'}).reset_index()
Role_Playing=Role_Playing[['Year','Name' , 'Global_Sales']]

Strategy = df1[df1.Genre == 'Strategy']
Strategy =Strategy.groupby(['Year']).agg({'sum'}).reset_index()
Strategy=Strategy[['Year','Name' , 'Global_Sales']]


# Plotting the Sales per Genre by Releas Year

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Shooter['Year'],
        Shooter['Global_Sales'],
        color='r',
        label='Shooter');
ax.plot(Misc['Year'],
        Misc['Global_Sales'],
        color='b',
        label='Misc');
ax.plot(Action['Year'],
        Action['Global_Sales'],
        color='g',
        label='Action');
ax.plot(Sports['Year'],
        Sports['Global_Sales'],
        color='c',
        label='Sports');
ax.plot(Strategy['Year'],
        Strategy['Global_Sales'],
        color='m',
        label='Strategy');
ax.plot(Fighting['Year'],
        Fighting['Global_Sales'],
        color='y',
        label='Fighting');
ax.plot(Puzzle['Year'],
        Puzzle['Global_Sales'],
        color='k',
        label='Puzzle');
ax.plot(Racing['Year'],
        Racing['Global_Sales'],
        color='tab:orange',
        label='Racing');
ax.plot(Platform['Year'],
        Platform['Global_Sales'],
        color='tab:green',
        label='Platform');
ax.plot(Simulation['Year'],
        Simulation['Global_Sales'],
        color='tab:brown',
        label='Simulation');
ax.plot(Adventure['Year'],
        Adventure['Global_Sales'],
        color='tab:purple',
        label='Adventure');
ax.plot(Role_Playing['Year'],
        Role_Playing['Global_Sales'],
        color='tab:olive',
        label='Role_Playing');
ax.set_title('Sales per Genre by Realese Year')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();


# As you see the graph included all genres is pretty messy and hard to analyse, so I'll take just one case, **Sports genre**, since I find it more interesting.\
# You can tweak colors and highlight genres in which you are intersted, by yourself*

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Shooter['Year'],
        Shooter['Global_Sales'],
        color='whitesmoke',
        label='Shooter');
ax.plot(Misc['Year'],
        Misc['Global_Sales'],
        color='whitesmoke',
        label='Misc');
ax.plot(Action['Year'],
        Action['Global_Sales'],
        color='whitesmoke',
        label='Action');
ax.plot(Sports['Year'],
        Sports['Global_Sales'],
        color='c',
        label='Sports');
ax.plot(Strategy['Year'],
        Strategy['Global_Sales'],
        color='whitesmoke',
        label='Strategy');
ax.plot(Fighting['Year'],
        Fighting['Global_Sales'],
        color='whitesmoke',
        label='Fighting');
ax.plot(Puzzle['Year'],
        Puzzle['Global_Sales'],
        color='whitesmoke',
        label='Puzzle');
ax.plot(Racing['Year'],
        Racing['Global_Sales'],
        color='whitesmoke',
        label='Racing');
ax.plot(Platform['Year'],
        Platform['Global_Sales'],
        color='whitesmoke',
        label='Platform');
ax.plot(Simulation['Year'],
        Simulation['Global_Sales'],
        color='whitesmoke',
        label='Simulation');
ax.plot(Adventure['Year'],
        Adventure['Global_Sales'],
        color='whitesmoke',
        label='Adventure');
ax.plot(Role_Playing['Year'],
        Role_Playing['Global_Sales'],
        color='whitesmoke',
        label='Role_Playing');
ax.set_title('Sales per Genre')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();


# In case of Sports genre we can observe very drastic increases and decreases. Let's exemine them closer.\
# Two peaks in 2006 and 2009 is our main interest.

# In[ ]:


Sports1=Sports.copy()
Sports1.columns=['Year', 'Name', 'Global_Sales']
Sports1=Sports1.sort_values(by='Global_Sales', ascending=False)
Sports1.nlargest(5,'Global_Sales')


# Table above shows 5 top grossing years in Sports genre. Two peaks as we expected: 2009: 138.52 mln, 2006: 136.16 mln.
# Now let's exemine each of these years closer:

# In[ ]:


df5=df1.copy()

#top 5 sold games in Sprts genre in 2006.
df5=df5[df5.Genre == 'Sports']
y1 = df5.loc[df['Year'] == 2006]
y2006=y1.nlargest(5,['Global_Sales'],keep='first') 
y2006


# Table above is representing top 5 games of Sports genre in 2006.\
# From the first glance we notice that Wii Sports is responsible for the 82.74 mln out of 136.16 mln, which is arounf 60% of all sales in that year - quite significant.\
# My explanation is that 2006 is a year of Wii platform introduction which is the main driver, so basically technological development factor played a major role in Sports Genre sales increase.

# In[ ]:


y2 = df5.loc[df['Year'] == 2009]
y2009=y2.nlargest(5,['Global_Sales'],keep='first') 
y2009


# Table above is representing top 5 games of Sports genre in 2009.
# This time we see that Nintendo's Wii Sports had only 33 mln in global sales, which is around 24% of total sales. \
# If we agregate Wii Sports Resort and Wii Fit Plus total sales would be 55 mln, 39% of total.\
# Based on the previous statements we can say that Wii Sports of Nintendo Alone is not a main driving factor.
# 
# To go further on our "technological factor drive" hypothesis development, let's see how much Sport Genre games for Wii platform earned in 2006 and 2009.

# In[ ]:


wii1 = y1.loc[df['Platform'] == 'Wii']
wii1_sum = wii1['Global_Sales'].sum()
wii1_sum


# In 2006 total sales for the Wii platform games is 84.03, from which 82.74 is Wii Sports. "Technological factor drive" hypothesis is true for the year 2006, when the Wii Sprts platform was introduced.

# In[ ]:


wii2 = y2.loc[df['Platform'] == 'Wii']
wii2_sum = wii2['Global_Sales'].sum()
wii2_sum


# In 2009 total sales for the Wii platform games is 88.17, from which 55 is Wii Sports. "Technological factor drive" hypothesis is true for the year 2006, but the Wii Sports is not the only contributor.\
# Let's see what are the other games made for the Wi platform in 2009.

# In[ ]:


wii2.info()


# In[ ]:


wii2.head(10)


# We see that in total there were 67 games published for the Wii platform including 2 Wii Sports games.\
# Conclusion: \
# The "Technological factor drive" hypothesis is true for both 2006 and 2009, since the major factor was the introduction of Wii platform and Wii balance. 

# ## Demand shift

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Shooter['Year'],
        Shooter['Global_Sales'],
        color='r',
        label='Shooter');
ax.plot(Misc['Year'],
        Misc['Global_Sales'],
        color='b',
        label='Misc');
ax.plot(Action['Year'],
        Action['Global_Sales'],
        color='indigo',
        label='Action');
ax.plot(Sports['Year'],
        Sports['Global_Sales'],
        color='c',
        label='Sports');
ax.plot(Strategy['Year'],
        Strategy['Global_Sales'],
        color='whitesmoke',
        label='Strategy');
ax.plot(Fighting['Year'],
        Fighting['Global_Sales'],
        color='whitesmoke',
        label='Fighting');
ax.plot(Puzzle['Year'],
        Puzzle['Global_Sales'],
        color='whitesmoke',
        label='Puzzle');
ax.plot(Racing['Year'],
        Racing['Global_Sales'],
        color='lightgrey',
        label='Racing');
ax.plot(Platform['Year'],
        Platform['Global_Sales'],
        color='whitesmoke',
        label='Platform');
ax.plot(Simulation['Year'],
        Simulation['Global_Sales'],
        color='lightgrey',
        label='Simulation');
ax.plot(Adventure['Year'],
        Adventure['Global_Sales'],
        color='whitesmoke',
        label='Adventure');
ax.plot(Role_Playing['Year'],
        Role_Playing['Global_Sales'],
        color='whitesmoke',
        label='Role_Playing');
ax.set_title('Sales per Genre per Realese Year')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();


# Based on the messy chart above we can track certain sales/demand shift from genre to genre.\
# 2000 - Increase of sales/demand: Action and Sport, decrease: Shooter.\
# 2006 - Sport on its first peak, decrease of sales/demand: Action, Shooter, Simulation, Puzzle.\
# 2007-2008 - Peak of the: Fight, Misc, and Racing, decrease: Sport.\
# 2010 - Peak of the: Sport ,Action, decrease: Misc, Role-Play, Racing.\
# 2011-2012 - Peak: Shooter,Misc, increase: Action, decrease: Sport.\
# 2015 - Increase: Sport, the same level: Shooter, decrease: all the rest.\
# \
# \
# The sales can't go upwards al the time, since all game developers sharing the same market, so the sales shifting from one to another genre.
# Looking in overall tendency we can observe that overall peak of the Video Games industry was in 2005-2010. Seince then the overall sales per genre decreases, we can *speculate* that it was the biggest cappacity of the market, meaning the max sales possible. E.g. 2007 decrease of Sport genre was "compensated" by increase of Action and some other genre. \
# Also we should consider the fact that the games' publishing year is also the main driver of the demand shift*

# The graph above is showing a certain recesion in the Video Games industry, but do not let it confuse you. The industry was developing and having profits since 1980 until 2018. (source:https://en.wikipedia.org/wiki/Video_game_industry)
# The reason for the difference with our plotted graph maybe the fact that the dataframe do not include other platforms as mobile games, one of the fast growing market. 
# The shift to the mobile games also can partially explain the decrease in other platforms, but it doesn't mean that Publishers in loss, since the same Publishers produce mobile games.
