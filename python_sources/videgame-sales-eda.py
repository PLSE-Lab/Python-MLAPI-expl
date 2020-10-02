#!/usr/bin/env python
# coding: utf-8

# # Video Game Sales Exploratory Analysis
# 
# Data available at: https://www.kaggle.com/gregorut/videogamesales
# 
# 
# This notebook still is a work in progress.
# 
# In this notebook i'm trying to think as a decision taker of a game dev studio, just analyzing the data to gather knowledge and aid in a decision
# 
# You can contact-me on my <a href='https://www.linkedin.com/in/ramonrcn/'>LinkedIn profile</a>

# In[ ]:


#importing libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Reading and basic pre-processing

# In[ ]:


#reading the data

vg = pd.read_csv('../input/vgsales.csv')
vg.info()


# In[ ]:


vg.head()


# In[ ]:


vg.tail()


# In[ ]:


vg.describe(include = 'all')


# One thing wrong here: Max year 2020. The dataset has registers up to 2017. We'll address it later

# In[ ]:


# Checking for NaN's
vg.isnull().sum()


# In[ ]:


# Deleting the entries with NaN's
vg.dropna(subset = ['Year','Publisher'], how = 'any', axis = 0, inplace = True)

# Setting the year as int
vg.Year = vg.Year.astype(int)

#Checking the Df info again
vg.info()


# In[ ]:


# Fixing one registry, the game came out in 2009, and the data go up to early 2017
## After a quick filter and google search, i found out that the game came out in '09
## vg.loc[vg.Year == 2020]

vg.Year.replace(2020, 2009, inplace= True)


# ## Which Platform choose? and what genre?

# If i was representing a game development studio, I wonder what platform has the biggest marketshare, and what genres could perform better in each platform. So let's get to it.

# In[ ]:


# Consolidating the data in another DataFrame

'''This dataframe contains all the sales data ordered by platform and global sales. It already meets my specification of knowing
what genre does best at each platform. But its tedious to read and takes too long to absorb the information'''

cons = vg.groupby(['Platform','Genre'], as_index = True).sum()
cons.drop(['Rank','Year'], axis= 1, inplace = True)
cons.sort_values(by = ['Platform','Global_Sales'], ascending = False, inplace = True)
cons.head(24)


# In[ ]:


# Gathering the same data as the above DF, but in a more 'plot friendly shape'

cons2 = vg.groupby(['Platform','Genre'], as_index = False).sum()
cons2.drop(['Rank','Year'], axis= 1, inplace = True)
cons2.sort_values(by = ['Platform','Global_Sales'], ascending = False, inplace = True)

# Filtering the games that sold more than 4 mil. copies, to generate a cleaner plot
cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False)


# In[ ]:


# Setting the colors used in the plots below, fell free to try any other color palette

#palette=sns.diverging_palette(128,240,79,32,1,n=12)
#palette= sns.color_palette("YlGnBu", 12)

clrz = ['#d70909',"#f26522",'#0000ff','#FFE00E','#a864a8','#790000','#005826','#00bff3','#636363','#8dc63f','#440e62','#ec008c']


# In[ ]:


# Ploting all the data from the DF above

with sns.plotting_context('notebook'):
    
    sns.catplot(data = cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False),
                x= 'Platform', y= 'Global_Sales', hue= 'Genre', ci = None, kind= 'swarm', 
                dodge = False, alpha = .8, aspect=2.5, marker = 'o', palette=sns.color_palette(clrz))
    
    sns.despine(left= True, bottom=True)
    plt.title('Average Game Sales By Genre and Platform (in milion units)')
    plt.grid(axis='both',which='major')
    plt.show()


# This plot gives a good idea of the average sales number, but is kinda hard to see, so i plotted the graph below

# In[ ]:


# Plotting a better graph

#This is an overlayed plot
f, ax = plt.subplots(ncols=1,nrows=1,sharey=True,figsize=(15, 10),dpi = 300)

with sns.plotting_context('notebook'):
    
    g = sns.barplot(data=cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False), x='Platform',
                y='Global_Sales', hue ='Genre', ci=None, dodge=False, alpha= .15,palette=sns.color_palette(clrz), ax=ax)
       
    g.set_xticklabels('')
    g.set_xlabel('')  
    
    ax2 = ax # Using the same axis (The ax.twinx method upsets the Y axis, and is a pain to realign later)
    
    g = sns.stripplot(data = cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False),
                x= 'Platform', y= 'Global_Sales', hue= 'Genre', palette=sns.color_palette(clrz),  
                dodge = True, alpha = 1, marker = 'D',ax=ax2  )
    sns.despine(left= True, bottom=True)
    plt.title('Average Game Sales By Genre and Platform (in milion units)')
    plt.grid(axis='both',which='major')
    plt.legend(ncol=2, frameon=True, loc='upper right')
    plt.show()


# Ok, so we already know what genres do better in which platform. What about the regional preference?
# What games and genres did well in each part of the world?

# <a id='SalesGenreOverview'></a>

# ## Sales by Genre and Region overview

# In[ ]:


# Sales by genre and region plot

regional = vg.groupby('Genre').sum().sort_values('Global_Sales', ascending = False).drop(['Rank','Year'], axis=1)
regional.plot(kind = 'bar', figsize = (15,8), rot= 0, fontsize = 12, grid= True, width=0.8)
plt.title('Sales by genre and region (in milion units)')
plt.show()
print()
regional


# ## Top 10 selling games by region

# In[ ]:


NA = vg.sort_values('NA_Sales',ascending = False).head(10)
EU = vg.sort_values('EU_Sales',ascending = False).head(10)
JP = vg.sort_values('JP_Sales',ascending = False).head(10)
Other = vg.sort_values('Other_Sales',ascending = False).head(10)
Global = vg.sort_values('Global_Sales',ascending = False).head(10)

#top10 = pd.concat([NA,EU,JP,Other,Global], axis = 0, ignore_index = True)


# ### North America

# In[ ]:


NA


# In[ ]:


NAg = NA.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='NA_Sales', ascending = False).reset_index()


# In[ ]:


sns.barplot(data=NAg, y='Genre',x='NA_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in North America by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
NAg


# ### Europe

# In[ ]:


EU


# In[ ]:


EUg = EU.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='EU_Sales', ascending = False).reset_index()


# In[ ]:


sns.barplot(data=EUg, y='Genre',x='EU_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in Europe by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
EUg


# ### Japan

# In[ ]:


JP


# In[ ]:


JPg = JP.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='JP_Sales', ascending = False).reset_index()


# In[ ]:


sns.barplot(data=JPg, y='Genre',x='JP_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in Japan by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
JPg


# This kinda surprised me to be honest, each region has some preference over the others, let's take the Wii Sports as an exemple:
# I remember when the Wii came out in '06, it was a huge hit, instant success all over the world, Except in Japan! it's homeland.
# 
# In fact, this shows how the japanese gamers prefer the portable platforms, like the DS, PSP and such. Other Japanese preference is the Pocket Monsters, Pokemon dominates the top 10 listing in Japan, and score a 5th place in most sold game worldwide.

# ### Other Regions

# In[ ]:


Other


# In[ ]:


Otherg = Other.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='Other_Sales', ascending = False).reset_index()


# In[ ]:


sns.barplot(data=Otherg, y='Genre',x='Other_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in Other regions by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
Otherg


# ### Global Sales

# In[ ]:


Global


# In[ ]:


Globalg = Global.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='Global_Sales', ascending = False).reset_index()


# In[ ]:


sns.barplot(data=Globalg, y='Genre',x='Global_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold Globally by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
Globalg


# Overall we can see that each region has their preferences in game genres and platforms  [as we can see in this plot ](#SalesGenreOverview) based on that info. we can already select some genres and regions to either make a deeper study, or analisys to take the decision of developing or not a new title with certain genre and target region

# ## Number of sales by Publisher

# In[ ]:


pub = vg.groupby('Publisher').sum().sort_values('Global_Sales', ascending = False).drop(['Rank','Year'], axis=1)

f = plt.figure(figsize=(15, 10),dpi = 300)
sns.barplot(data= pub.reset_index().head(20), x= 'Global_Sales', y='Publisher')
plt.grid(axis='both')
plt.title('Number of Sales by Publisher (In million units)')
plt.show()


# Well, Nintendo is'nt called 'Big N' for nothing as we can clearly see in the above plot

# ## Number of sales by Publisher and region

# In[ ]:


# 'massaging' the data to generate the plot below
pub2 = pub.sort_values('Global_Sales', ascending = False).reset_index().head(20).melt(
    id_vars='Publisher', value_vars=pub.columns, var_name= 'Region',value_name='Sales')

# Replacing the region tags
pub2.Region.replace(to_replace='NA_Sales',value='North America',inplace=True)
pub2.Region.replace(to_replace='EU_Sales',value='Europe',inplace=True)
pub2.Region.replace(to_replace='JP_Sales',value='Japan',inplace=True)
pub2.Region.replace(to_replace='Other_Sales',value='Other Regions',inplace=True)
pub2.Region.replace(to_replace='Global_Sales',value='Global Sales',inplace=True)

# Generating the plot
with sns.plotting_context('notebook'):    
    c = sns.catplot(data= pub2, y='Publisher', x='Sales', col='Region', col_wrap = 3, kind='bar',sharex=False,)
    c.fig.set_dpi(150)
    c.set_titles('Number of Sales in {col_name}')


# ## Best genre by publisher

# Each publisher has its strong genres, like Nintendo does really good with Platform, Capcom with Fighting and so on. This is kinda common knowledge, but let's check properly with the data available

# In[ ]:


#Wrangling the data to plot the info
GenPub = vg.groupby(['Publisher','Genre']).sum().sort_values('Global_Sales', ascending = False).drop(['Rank','Year'], axis=1).reset_index()

GenPub.rename({'NA_Sales':'North America','EU_Sales':'Europe','JP_Sales':'Japan','Other_Sales':'Other Regions','Global_Sales':'Global'},
              axis= 'columns', inplace= True)

GenPub.groupby(['Publisher','Genre'],as_index=False).sum().sort_values('Global', ascending = False, inplace = True)

GenPub = GenPub.melt(id_vars=['Publisher','Genre'], var_name='Region', value_name='Sales')


# In[ ]:


# Top 10 Publishers by genre
sns.set_context('notebook')

f, ax = plt.subplots(figsize=(10, 7),dpi = 200,)
f = sns.barplot(data= GenPub.loc[GenPub.Sales >= 60], x='Sales', y='Publisher', hue='Genre', estimator=max, errwidth=0) 
ax.set_xticks([50,150,250,350,450],minor=True)
plt.legend(loc=4)
plt.title('Top 15 publishers by genre')
plt.xlabel('Sales in milion units')
plt.grid(which='both', axis='both', alpha=0.3)

# Overlay dots at the bars
#sns.stripplot(data= GenPub.loc[GenPub.Sales >= 60], x='Sales', y='Publisher', hue='Genre', ax=ax, dodge = True)


# ## Most sold games
# The videogame industry has come a long way, and in it's history, some games became classics, selling milions of copies worldwide.
# Let's check the top selling game of the year

# In[ ]:


# Gathering the data
msold = vg.groupby(['Year','Name','Platform'], axis= 0, as_index=True).sum().sort_values(by=['Global_Sales'], ascending=False)#.reset_index()
msold.head()


# In[ ]:


# Rearanging the data
msold1 = msold.sort_values(by='Year',kind='mergesort').reset_index()
msold1.head()


# Now that we got the games sorted by year and most sold game, i wanted to pick just the most sold game for every year. Since that for every year the first row is the most sold game, the line below solved my problem

# In[ ]:


msold1.loc[msold1.Year == 1981].head(1)


# Threw it in a For loop, and got the result that i wanted

# In[ ]:


ano = 1980
most_sold = pd.DataFrame()
for val in msold1.iterrows():
    most_sold = msold1.loc[msold1.Year == ano].head(1).append(most_sold)
    #print(x)
    ano+= 1
    val = ano
    if (ano == 2017):
        break


# In[ ]:


# Most sold games by year
most_sold.sort_values(by='Year',ascending = True)


# ### Games sales by year

# In[ ]:


sby = vg.groupby('Year').sum().drop(['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1).reset_index()
f = plt.figure(figsize=(15, 10),dpi = 300)
sns.barplot(data=sby,y='Global_Sales', x='Year', estimator = max)
plt.xticks(rotation=90)
plt.ylabel('Global Sales in Milion Units')
plt.title('Game Sales by Year')
plt.grid(axis='both')
plt.show()


# ### Most sold genres by year

# In[ ]:


# Gathering the data and ordering by Year and global sales

gby = vg.groupby(['Year','Genre'], as_index= False).sum().drop(['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1)
gby.sort_values(by=['Year','Global_Sales'], ascending=[True,False], inplace= True)


# In[ ]:


# This is kinda hard to see, but i think its the best plot that i could came up with, 
#please feel free to modify and represent the information in some other way!

f = sns.catplot(data=gby, y='Global_Sales', x='Year', hue = 'Genre', orient='v', kind= 'strip', aspect=2, height=8, 
                dodge=False, marker='D', palette=clrz)
plt.grid(axis='both',which='both')
plt.xticks(rotation=45)
f.ax.set_yticks([10,30,50,70,90,110,130], minor=True)
plt.ylabel('Global Sales in Milion Units')
plt.title('Yearly Global Sales by Genre', fontdict={'fontsize':15})
plt.show()


# ### Top 6 consoles (marketshare)

# OK, so to acurately represent this we would need actual console sales numbers. But since this data is lacking, let's assume that for every game sold, one console was sold too

# First lets check the overall most sold consoles, and than the consoles from the last gen and portables

# In[ ]:


# 6 Most sold consoles overall
cons = vg.groupby(['Platform'], as_index=False).sum().drop(['Rank','Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1).sort_values('Global_Sales', ascending=False).head(6)
cons


# In[ ]:


# 7th and 8th gen consoles
cons2 = vg.groupby(['Platform'], as_index=False).sum().drop(['Rank','Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1).sort_values('Global_Sales', ascending=False)

cons2.loc[(cons2.Platform == 'X360') | (cons2.Platform == 'PS3') | (cons2.Platform == 'Wii') | 
         (cons2.Platform == 'PS4') | (cons2.Platform == 'XOne') | (cons2.Platform == 'WiiU') | (cons2.Platform == 'DS') |
        (cons2.Platform == 'PSP') | (cons2.Platform == '3DS') | (cons2.Platform == 'PSV')]


# Now that we got the 12 most sold consoles (in our analysis), let's see their sales performance over the years. The oldest platform in this case is the OG Play Station that came out in 1994, so we'll look from '94 to '17

# In[ ]:


temp = vg.groupby(['Platform','Year'], as_index= False).sum().drop('Rank', axis=1)


# In[ ]:


cons12 = temp.loc[(temp.Platform == 'PS') | (temp.Platform == 'PS2') | (temp.Platform == 'PS3') | (temp.Platform == 'PS4') | 
        (temp.Platform == 'X360') | (temp.Platform == 'XOne') | (temp.Platform == 'Wii') | (temp.Platform == 'WiiU') | 
        (temp.Platform == 'DS') | (temp.Platform == 'PSP') | (temp.Platform == '3DS') | (temp.Platform == 'PSV')]


# When i ploted the graph the 1st time, i noticed something funny going on with the years displayed in the plot. It had a random '1985' somewhere, and here is how i tracked it down

# In[ ]:


cons12.describe()


# In[ ]:


cons12.loc[cons12.Year == 1985]


# In[ ]:


cons12.loc[cons12.Platform == 'DS']


# I'm not a big fan of simply deleting stuff from the Dataset, so i'll just add that 0.02 from the wrong 1985 to 2014

# In[ ]:


cons12.loc[(cons12['Platform'] == 'DS') & (cons12['Year'] == 1985)]


# In[ ]:


cons12.loc[(cons12['Platform'] == 'DS') & (cons12['Year'] == 2014)]


# The warnings below i find interesting to leave in because they taught me how to get to specific values in the dataset and change certain values.
# 
# I was trying to use the .loc method to select the specific values that i wanted to than change it, but it wasn't working, because mainly the .loc method returns a 'view' of that data, not the actual data itself so it throws a warning and does nothing.
# 
# Than i tried with the .iloc method and it took me a while but i figured it out reading the documentation available at the link shown by the warning.
# it's under the section: 'Why does assignment fail when using chained indexing?' and really worth the reading

# In[ ]:


# Acessing the 2014 register of the DS platform, through the slice 17:18, and then on the 4th column, assigning the 0.02 value
cons12.iloc[17:18,(4)] = 0.02


# In[ ]:


# Acessing the 2014 register of the DS platform, through the slice 17:18, and then on the last column, updating the value to 0.04
cons12.iloc[17:18,(-1)] = 0.04


# In[ ]:


# Deleting the 1985 entry
cons12.drop(labels=25, axis=0, inplace= True)

# And on the last line, the updated values
cons12.head(17)


# In[ ]:


# Most sold consoles from 6th, 7th and 8th generations
g = sns.catplot(data= cons12, x='Year', y='Global_Sales', col='Platform', col_wrap=2, kind='bar', 
            sharex=False, sharey= True, aspect= 2.5, height = 4.5, estimator= max)


# ### Running in the 90's

# I was born in 1993, love racing games and cars from the 90's era. So lets take a look on the racing games from that era

# In[ ]:


race = vg.loc[(vg.Genre == 'Racing') & (vg.Year < 2000) & (vg.Year > 1989)]
race.reset_index(inplace=True, drop=True)
race


# Didn't figured out what i want to know of the racing games yet, i guess that is to come on the next update of this notebook...

# In[ ]:





# In[ ]:





# ### To do list:
# - Most sold game, 1990 to 2017 [OK]
# - Games sales by year [OK]
# - Genre sales by year [OK]
# - Console marketshare (top 6) [OK]
# - Running in the 90's - Racing games from 1990 (maybe)
# 
# - Word clouds:
# -- Games
# -- Publisher
# 
# #### Idea to try:
# ML model to forecast sales given: Publisher, Genre, Region (choosable via dropdown list)

# In[ ]:




