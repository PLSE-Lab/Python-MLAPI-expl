#!/usr/bin/env python
# coding: utf-8

# # Summary 
# 
# #### Background
# The data set shows the top100 chess players and their ratings over the last 17 years. Chess has been the first sports that have developed its own rating system (ELO) [wiki ELO] in contrast to other sports where usually weighted tournament wins will be summed up. The elo rating system has been adopted later to many other sports.
# 
# Besides that the data set kept record of a major part of the professional carreer of a chess player. A carreer in chess could be longer than the included 17 years in this data set, however that timespan is longer than a usual career of other professional athlethes. An ELO rating has the nice property that a ranking is somewhat quantifiable rather than just ordered.
# The sum of all ELO points in a closed group of chess players is constant; this is a constructed property of mathematics of the rating system; obviously is that is not the case as players drop in and out the top 100 and other factors in the actual use the elo system in chess. 
# 
# #### Approach
# We explore first the data set technically and analyse its data quality. Afterwards we cleanse and enrich the data set. The next chapter performs describe statistics and visualisation on the data. Then we make an effort derive hidden structures in the dataset and prepare in for some machine learning. The following notebook should be considered as a <b><u>work-in-progress</u></b> data exploration, rather than well told and completed data story.
# 
# The author has personal interest in watching and following chess for many years, however find rarely the time to play.
# 
# #### Key Findings
# 
# * Chess is getting younger; at least very slightly
# * Rated Chess is not played more often. One could assume that as it is the case for other sports like Formula One racing or European Football
# * A phase a increase "rating inflation" happen through ~2008-2010

# In[5]:


from collections import OrderedDict
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.finance import candlestick2_ohlc
import seaborn as sns
from datetime import datetime 
from sklearn import linear_model
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-pastel')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
elo = pd.DataFrame.from_csv("../input/fide_historical.csv")
elo = elo.reset_index()


# ## Enriching data
# 
# * PubYear - Year part of ranking date
# * PubMonth - Month part of ranking date
# * PubRunMouth - Running month since start of the data set
# * TimeDelta - Length of the actual interval in month, 1st interval = 3 months
# * Age - rounded up (usual in sports)
# * Listindex - running number on each publication of a ranking list
# * Rating_Move = difference to last rating
# * Rating_Move_timedelta = difference to last rating (weight by publishing period)

# In[6]:


elo['PubYear']     = elo.ranking_date.dt.year                      # Enriching the dataset PubYear
elo['PubMonth']    = elo.ranking_date.dt.month                     # Enriching the dataset PubMonth
elo['PubRunMonth'] = ((elo['PubYear']-2000)*12+elo['PubMonth'])-6  # Enriching the datasat PubRunMonth

# first period assumed to be three months
interval = [3.0]          + [round((d1-d0)/pd.Timedelta(1, unit='M'))
            for d0,d1 in zip(elo.ranking_date.unique()[0:-1],
                             elo.ranking_date.unique()[1:])]
elo['timedelta']   = elo['ranking_date'].map(dict(zip(elo.ranking_date.unique(),interval)))
elo['age']         = elo['PubYear']-elo['birth_year']
elo['listindex']   = elo['ranking_date'].map(dict(zip(elo.ranking_date.unique(),
                                                      range(0,elo.ranking_date.nunique()))))
for p in elo.name.unique():
    elo.loc[elo.name==p,'rating_move']        =-elo.loc[elo.name==p,'rating'].shift(1)        +elo.loc[elo.name==p,'rating']
elo['rating_move_timedelta']=elo['rating_move']/elo['timedelta']


# ## Data Cleansing
# Player names are spelled incorrectly or inconsitenly in the official rating lists. Credits to [srlmayor]. 

# In[7]:


# Manually reviewed and replaced
replacement = {
    'Bologan, Victor'             : 'Bologan, Viktor',
    'Bruzon Batista, Lazaro'      : 'Bruzon, Lazaro',
    'Dominguez Perez, Lenier'     : 'Dominguez, Leinier' ,
    'Dominguez Perez, Leinier'    : 'Dominguez, Leinier',
    'Dominguez, Lenier'           : 'Dominguez, Leinier',
    'Dreev, Aleksey'              : 'Dreev, Alexey',
    'Iturrizaga Bonelli, Eduardo' : 'Iturrizaga, Eduardo',
    'Kasparov, Gary'              : 'Kasparov, Garry',
    'Mamedyarov, Shakhriyaz'      : 'Mamedyarov, Shakhriyar',
    'McShane, Luke J'             : 'McShane, Luke J.',
    'Polgar, Judit (GM)'          : 'Polgar, Judit',
    'Sadler, Matthew D'           : 'Sadler, Matthew',
    'Short, Nigel D'              : 'Short, Nigel D.'}
for k,v in replacement.items():
    elo.loc[elo.name==k,'name'] = v


# ## General Findings
# 
# The data set consits of 114 data points i.e. published ranking lists. First publication was July 2000, last June 2017. 
# Initial publication have been quarterly, then bi-monthly, later monthly. According to the data describing the data has been provided from official source [World Chess Federation: fide.com] - the governing body of the chess clubs and the organisator of the official chess world championship.
# 
# From the data quality it is obvious that chess have not reached the professionalism on official statistics as for example other major americans sports or european league football. 

# In[8]:


print('Day of month the data is released:\t%i' %elo.ranking_date.dt.day.unique()[0])
# always on the 27th; information can be discarded and regarded as a monthly dataset
NUMBER_DATAPOINTS_ = elo.ranking_date.nunique()
print('Number of rating list publications:\t%i' %NUMBER_DATAPOINTS_)
STARTDATE_=elo.ranking_date.min()
print('Date of first publication:\t\t%s'%STARTDATE_)
ENDDATE_=elo.ranking_date.max()
print('Date of first publication:\t\t%s'%ENDDATE_)


# In[9]:


ax=elo.groupby('PubYear')['PubMonth'].nunique().plot(kind='bar')
ax.set_xlabel('Year')
ax.set_ylabel('Rating publications')
ax.set_title('Rating publications per year');
 


# ## Findings on individual attributes
# ### "Title"
# 
# 'g' stands for Grandmaster. It's lifetime title aquires around 2500-2600 rating points on slightly different criterias. One can assume that the 100 top chessplayer will hold the grandmaster title nowadays. There are ~2000 living players (active and retired) carrying that title.
# 
# * Title is pointless. There are three outliers.
#  - 'wg' (women grandmaster) are treated differently, see Polgar vs. Hou
#  - 'm' (Smirnov) received the 'g' in 2003. [wiki Smirnov]
#  - 'f' (Afromeev) outlier is even noted in wikipedia [wiki Afromeev]

# In[10]:


print(elo[elo.title!='g']         .groupby(['name'])         ['name', 'title'].max().values)


# In[11]:


print(elo[elo.name=='Hou, Yifan']         .groupby(['name'])         [['name', 'title']].max().values) #'wg' counterexample


# # Descriptive analysis on the dataset
# 
# ## 'Number one'
# 
# The obvioulys prominent position is the number one spot. It is not related in any way to the chess world champions, who is in a tournament cycle and (almost) not related to the elo rating system.

# In[12]:


g=elo[elo['rank']==1][['ranking_date', 'name']]
clr={'Kasparov, Garry'   : 'b',
     'Topalov, Veselin'  : 'g',
     'Anand, Viswanathan': 'y',
     'Kramnik, Vladimir' : 'k',
     'Carlsen, Magnus'   : 'r'}


# In[13]:


fig, ax = plt.subplots(figsize=(17,2))
ax.set_xlim(g.ranking_date.iloc[0],g.ranking_date.iloc[113])
for i in range(len(g)-1):
    ax=plt.barh(left=g.ranking_date.iloc[i],
            height=1,
            width=(g.ranking_date.iloc[i+1]-g.ranking_date.iloc[i]).days,
            bottom=0.5,
            edgecolor='none',
            color=clr[g.name.iloc[i]],
            label=g.name.iloc[i], alpha=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.02, 1.0))
plt.yticks(visible=False)
plt.title('Number One player over time');


# In[14]:


fig, ax = plt.subplots(figsize=(17,6))
for n in clr.keys():
    elo[elo['name']==n][['ranking_date','rating']]        .rename(columns={'rating': n})        .plot(ax=ax, label=n, color=clr[n])
plt.ylabel('Rating')
plt.xlabel('Date')
ax.legend(loc='lower right')
plt.title('Ratings over time of the number one players');


# ## Current top 5
# 
# <font color='blue'>Can we break down these curves into something? </font>

# In[15]:


elo[elo.ranking_date==ENDDATE_].name
clr={'Carlsen, Magnus'       :'r',
     'So, Wesley'            :'b',
     'Kramnik, Vladimir'     :'k',
     'Caruana, Fabiano'      :'g',
     'Mamedyarov, Shakhriyar':'y'}


# In[16]:


fig, ax = plt.subplots(figsize=(17,6))
names =clr.keys()
for n in names:
    elo[elo['name']==n][['ranking_date','rating']]        .rename(columns={'rating': n})        .plot(ax=ax, label=n, color=clr[n])
plt.ylabel('Rating')
plt.xlabel('Date')
ax.legend(loc='lower right')
plt.title('Ratings over time of the current top 5 players');


# ### Candlesticks

# In[17]:


df=elo[elo['name']=='Carlsen, Magnus'][['ranking_date','rating']]
df['ryear'] = df.ranking_date.dt.year
df= df.groupby('ryear', as_index=True)  .agg({'rating': ['first','max', 'min','last']})  .loc[:,'rating']  .reset_index()
fig, ax = plt.subplots(1,1)
candlestick2_ohlc(ax, df['first'], df['max'], df['min'], df['last'], width=0.9)
ax.set_xticklabels(df.ryear)
ax.set_title('Candelstick chart per year (Magnus Carlsen)');


# ## Elo over time
# 
# The data set will show the most upper end of normal elo distribution. Elo ratings have a slight inflation over time. The effect seems to be visible just by plotting the data in histograms on the same scale.
# Plotting the average (of a subset) visualises it even better.
# 
# <font color='blue'>The shape requires some attention. I have no clue, why it is a little bit S-shape. Kaggle's TheTS has it as well in his analysis</font>
# 
# We plot the average rating of three different subset (Top10,Top50,Top100) and some derivated meaures (moving average on 10 periods and its derivation).

# In[18]:


#Show six equal distribted dates from the observation period
dates = [list(elo.ranking_date.unique())[int(i)]
         for i in np.linspace(0,elo.ranking_date.nunique()-1,6).round()]

row, col = 3,2
fig, ax = plt.subplots(col,row, figsize=(17,6))
fig.tight_layout(h_pad=2.0)

for c in range(col):
    for r in range(row):
        d=dates[c*3+r]
        ax[c,r].set_xlim(2650,2900)
        ax[c,r].set_ylim(0,22)
        elo[elo.ranking_date==d].rating.hist(bins=20, edgecolor='w',ax=ax[c,r])
        ts = pd.to_datetime(str(d)) 
        ax[c,r].set_title(ts.strftime('%b-%Y'))
fig.subplots_adjust(top=0.90)        
fig.suptitle('Distrubtion of the Top100 over time');       


# In[19]:


fig, ax = plt.subplots(3,2, figsize=(10,8), sharex=True)
fig.tight_layout()

top = (('Top10', 10, 'lightblue'),
       ('Top50', 50, 'slateblue'),
       ('Top100', 100, 'darkblue'))
for t, i in zip(top, range(len(top))):
    g=elo[elo['rank']<=t[1]]        .rename(columns={'rating': t[0]})        .groupby(['ranking_date'])        [t[0]].mean()        .to_frame()
    g['avg']=g[t[0]].rolling(window=10).mean()
    g['avgdelta']=g['avg']-g['avg'].shift(1)
    ax[i,0].plot(g[t[0]],color=t[2], label=t[0])
    ax[i,0].plot(g['avg'],color=t[2], ls='--', label='MA10')
    ax[i,1].plot(g['avgdelta'], label='1st deriv. MA10', color= t[2])
    ax[i,0].legend(loc='upper left')
    ax[i,1].legend(loc='upper left')
    ax[i,0].set_ylim(2630,2810)
    ax[i,1].set_ylim(-1,2.5)
    ax[i,1].axhline(0, c='k', lw=.5)
plt.xlabel('Date')
ax[0,0].set_title('Average rating over time');
ax[0,1].set_title('Changes in average rating over time');


# ## Largest rating moves
# 
# The question after the largest rating move cannot be answered in one single chart due to the changes in the publication interval; either you count by publication intervall or normalize down to one month. One can assume that even the large upward move contains draws and maybe losses, et vice versa, so both charts have its justification. (in a longer period there is more time to lose the points again)
# 

# In[20]:


top  = elo[['ranking_date', 'name','rating_move']]          .sort_values(['rating_move'], ascending=False)[:10]
flop = elo[['ranking_date', 'name', 'rating_move']]          .sort_values(['rating_move'])[:10]
g = pd.concat([top,flop]).sort_values(by='rating_move', ascending=False)

fig, ax = plt.subplots(2,1, figsize=(6,11), sharex=True)
fig.tight_layout()
rg=['r' if x<0 else 'g' for x in list(g.rating_move)]
ax[0].barh(np.arange(20),g.rating_move, label=g, color=rg)
ax[0].set_yticks(np.arange(20))
ax[0].set_yticklabels([str(n)+" ("+d.strftime('%b-%Y')+")" for n,d in zip(g.name, g.ranking_date)])
ax[0].set_title('Largest Rating Moves (per publishing period)');

top  = elo[['ranking_date', 'name','rating_move_timedelta']]          .sort_values(['rating_move_timedelta'], ascending=False)[:10]
flop = elo[['ranking_date', 'name', 'rating_move_timedelta']]          .sort_values(['rating_move_timedelta'])[:10]
g = pd.concat([top,flop]).sort_values(by='rating_move_timedelta', ascending=False)

rg=['r' if x<0 else 'g' for x in list(g.rating_move_timedelta)]
ax[1].barh(np.arange(20),g.rating_move_timedelta, label=g, color=rg)
ax[1].set_yticks(np.arange(20))
ax[1].set_yticklabels([str(n)+" ("+d.strftime('%b-%Y')+")" for n,d in zip(g.name, g.ranking_date)])
ax[1].set_title('Largest Rating Moves (per month)');


# ## Player statistics
# 
# Surprising there are only 334 unique players listed (in the Top 100 over the last **17 years**)
# 
# The second chart is not easy to read. 34 players are listed in the Top 100 only once (and then the dropped out again). 18 player have been listed each single pub date.

# In[21]:


print('Unique number of player listed in dataset: %i' %elo.name.nunique())


# In[22]:


fig, ax = plt.subplots(1,1, figsize=(17,5))
elo.groupby(['name'], as_index=False)    ['ranking_date'].count()    .groupby('ranking_date')    ['name'].count()    .reindex(np.linspace(1,NUMBER_DATAPOINTS_,NUMBER_DATAPOINTS_,dtype='int32'), fill_value=0 )    .plot(kind='bar')
ax.set_xlabel('Number of listing')
ax.set_ylabel('Number of players')
ax.set_xticks([0,1,2]+[i-1 for i in range(5,NUMBER_DATAPOINTS_,5)]+[112,113])
ax.set_xticklabels([1,2,3]+[i for i in range(5,NUMBER_DATAPOINTS_,5)]+[113,114])
ax.set_title('Frequency of a player listed');


# In[23]:


## ToDo get the 18 players for Linear Regression


# ## Average number of games per month
# 
# The is picture, I have expected. The majority are players, who are not that popular; we do the only two well know 2700 GMs (Giri, Yi) on the right. Both are relativily young.

# In[24]:


## ToDo add more stats here


# In[25]:


tab=elo.groupby(['name'])['games','timedelta'].sum()
tab['AvgPlayedPerMonth'] = tab['games']/tab['timedelta']

fig, ax = plt.subplots(1,1, figsize=(17,5))
tab.sort_values('AvgPlayedPerMonth', ascending=False)    ['AvgPlayedPerMonth'][0:50]    .plot(kind='bar')
ax.set_title('Top 50 players games per month');


# The chart is also somewhat expected. Promined outliner is Polgar, who has played rarely during the later years in her career.

# In[26]:


fig, ax = plt.subplots(1,1, figsize=(17,5))
tab.sort_values('timedelta', ascending=False)    ['AvgPlayedPerMonth'][0:50]    .plot(kind='bar')
ax.set_title('Games per month (filtered Top50 players by time top50');


# ## Games per month
# The number of games played per player over times is remarkable constant on average. It's also noteworthy that less chess is played in February.

# In[27]:


g=elo.groupby('ranking_date').agg({'games':'sum','timedelta':'first', 'PubYear': 'first', 'PubMonth':'first'})
g['games_month']=g['games']/g['timedelta']/100
sns.regplot(y=g['games_month'], x=np.linspace(0,len(g['games_month']),len(g['games_month'])), order=1);
plt.ylabel('Games per month per player')
plt.xlim(-2,117)
plt.title('Games per month over time');
#ToDo Add labels


# In[28]:


fig, ax = plt.subplots(1,1, figsize=(7,7))
ax = plt.subplot(projection='polar')
frac=1/12*2*np.pi
maxvalue=np.ceil(g.games_month.max())
color=cm.bone(np.linspace(0,1,25))#offset for darker color

for y,i in zip(range(2000,2018), range(0,18)):
    plt.plot((g[g.PubYear==y]['PubMonth'].values-1)*frac,
             g[g.PubYear==y]['games_month'].values,
             marker='o', lw=0, color=color[i], label=str(y))
ax.set_rmin(0)
ax.set_rmax(maxvalue)
ax.set_rticks([2.5,5,7.5,10])
ax.set_theta_offset(np.pi/2)
ax.set_thetagrids([30*i for i in range(12)])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(bbox_to_anchor=(1.25, 1.0))
plt.title('Average games played by month');
### DEAD END ###
# I believe I am seeing not much here. The pattern is an artifact due the different publication intervals.


# ## Age
# 
# The average age of a top 100 players shows an interest trend over time. The trends drops until 2011 and raises again. It is caused by younger players entering the list ~2008-2010, who are then aging slowly.

# In[29]:


top = (('Top10', 10, 'lightblue'),
       ('Top50', 50, 'slateblue'),
       ('Top100', 100, 'darkblue'))

fig, ax = plt.subplots(1,1, figsize=(7,5))
for t in top:
    elo[elo['rank']<=t[1]]        .rename(columns={'age': t[0]})        .groupby(['ranking_date'])        [t[0]].mean()        .plot(color=t[2])
plt.legend(loc='lower left')
plt.xlabel('Date')
ax.set_title('Average age over time');


# In[30]:


fig, ax = plt.subplots(3,2, figsize=(10,8), sharex=True)
fig.tight_layout()

top = (('Top10', 10, 'lightblue'),
       ('Top50', 50, 'slateblue'),
       ('Top100', 100, 'darkblue'))
for t, i in zip(top, range(len(top))):
    g=elo[elo['rank']<=t[1]]        .rename(columns={'age': t[0]})        .groupby(['ranking_date'])        [t[0]].mean()        .to_frame()
    g['avg']=g[t[0]].rolling(window=10).mean()
    g['avgdelta']=g['avg']-g['avg'].shift(1)
    ax[i,0].plot(g[t[0]],color=t[2], label=t[0])
    ax[i,0].plot(g['avg'],color=t[2], ls='--', label='MA10')
    ax[i,1].plot(g['avgdelta'], label='1st deriv. MA10', color= t[2])
    ax[i,0].legend(loc='upper left')
    ax[i,1].legend(loc='upper left')
    ax[i,0].set_ylim(28,34)
    ax[i,1].set_ylim(-0.5,0.5)
    ax[i,1].axhline(0, c='k', lw=.5)
plt.xlabel('Date')
ax[0,0].set_title('Average age over time');
ax[0,1].set_title('Changes in age over time');


# In[31]:


yeargroups=((0,9), (9,12), (12,17))

fig, axs = plt.subplots(1,3, figsize=(17,5), sharey=True)
for ax, yg in zip(axs, yeargroups):
    for y in range(yg[0],yg[1]):
        date=STARTDATE_+pd.DateOffset(years=y)
        elo[elo.ranking_date==date]['age'].plot(kind='kde', label=str(date.year), ax=ax)
    ax.legend()
    ax.set_xlim(-1,75)
    ax.set_ylim(0,0.07)
    ax.set_xlabel('Average age')
fig.suptitle('Age distribution at start of the year');    


# In[32]:


dates=[STARTDATE_+pd.DateOffset(years=y) for y in range(0,17)]
df=elo.loc[elo.ranking_date.isin(dates),['ranking_date','age']]
df["year"]=df.ranking_date.dt.year
           
pal = sns.cubehelix_palette(17, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="year", hue="year", aspect=8, size=0.9, palette=pal, xlim=(13,55),ylim=(0,0.07))
g.map(sns.kdeplot, "age", shade=True, alpha=1, lw=1.5)
g.map(sns.kdeplot, "age", color="w", lw=2) 
g.map(plt.axhline, y=0, lw=2, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .25, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

g.map(label, "age")
g.fig.subplots_adjust(hspace=0,top=0.98)
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True);
g.fig.suptitle('Age distribution at start of the year (alternate plot)');


# In[33]:


InOut = dict()

for i in range(0,NUMBER_DATAPOINTS_-1):
    name0 = set(elo[elo.listindex==i]['name'])
    name1 = set(elo[elo.listindex==i+1]['name'])
    a = elo[(elo.name.isin(name0-name1))&(elo.listindex==i)]        ['age'].mean()
    b = elo[(elo.name.isin(name1-name0))&(elo.listindex==i+1)]        ['age'].mean()
    InOut[i]=(a,b)

g=pd.DataFrame.from_dict(InOut).T
g['ranking_date']=list(elo.ranking_date.unique()[:-1])
g.columns = ['leaving', 'entering', 'ranking_date']   


# In[34]:


fig, ax  = plt.subplots(1,1,figsize=(17, 6))
l=np.linspace(0,NUMBER_DATAPOINTS_-1,NUMBER_DATAPOINTS_-1)
sns.regplot(y=g.entering, x=l, color='green', label='entering')
sns.regplot(y=g.leaving, x=l, color='red', label='leaving' )
plt.legend()
plt.xlim(-2,116)
plt.ylim(0, 45)
plt.ylabel('Age')
plt.title('Player entering and leaving the Top100 list');


# # Player statistics
# 
# We switch perspectives and analyse the characteristics of a player rather than a time series.
# 
# * Birth year: as given
# * Time Listed: Number of months listed in the Top100 (first period asssumed as 3 months)
# * Best Rank

# In[35]:


### DEAD END ###
# Failed to determine retired players
tab=elo.groupby('name').games.rolling(center=False,window=3).mean()
tab=tab.reset_index()
tab2=tab.groupby(['name']).games.last()
tab2.sort_values();


# In[36]:


player=elo.groupby(['name'])          .agg({'birth_year'  : min,
                'rank'        : min,
                'timedelta'   : sum,
                'games'       : sum,
                'ranking_date': min,
                'rating'      : max})\
          .reset_index()
LastRating=elo.groupby(['name'])              .agg({'ranking_date': max})              .reset_index()
player=pd.merge(how='inner', left=player, left_on='name', right=LastRating, right_on='name')
Country=elo.groupby(['name'])           .country.last()           .to_frame().reset_index()
player=pd.merge(left=player,  left_on='name', right=Country,right_on='name')
player.rename(columns = {'games':'TotalGames',
                         'timedelta': 'TimeListed',
                         'birth_year': 'BirthYear',
                         'rank':'BestRank',
                         'ranking_date_x':'FirstListed',
                         'ranking_date_y':'LastListed',
                         'country': 'Country',
                         'rating': 'Rating'},
              inplace=True)
#player['FirstListedNUM'] = round((player.FirstListed.subtract(pd.Timestamp(STARTDATE_)))/pd.Timedelta(1, unit='M'))
#player['LastListedNUM'] = round((player.LastListed.subtract(pd.Timestamp(STARTDATE_)))/pd.Timedelta(1, unit='M'))

# Regions
regions = {
    'Western Europe' :
        ['ENG','SWE','ESP','UKR','FRA','BEL','NOR','GER','SUI','GRE',
         'FIN','AUT','DEN','ISL','NED','IRL'],
    'Eastern Europe':
        ['HUN','POL','SLO','BUL','SVK','CRO','ROM','CZE','MKD','ROU','SRB','BIH'],
    'Former Soviet Union' :
        ['RUS','ARM','BLR','TJK','GEO','MDA','UZB','LAT','KAZ','AZE','LTU'],
    'America' : 
        ['USA','PAR','CAN','CUB','PER','ARG','BRA','VEN'],
    'Middle East & Africa' :
        ['EGY','TUR','ISR','UAE','MAR','IRI'],
    'Asia' :
        ['CHN','PHI','VIE','SGP','SCG','IND','INA']}
regionsmap = {}
for region, countrylist in regions.items():
    regionsmap.update({country: region for country in countrylist})
player['Region']=player['Country'].map(regionsmap)

#print(len(player)) #=334 as crosscheck
player.head()


# In[37]:


player.TimeListed.max()


# In[38]:


plt.scatter(y=player['TotalGames'], x=player['TimeListed'], color='lightblue');
plt.axvline(x=206, ymin=0, ymax=player['TotalGames'].max(), linewidth=1, color='red', alpha=0.5);
plt.xlim(-5, 210);
plt.title('Games played over time by player');


# In[39]:


plt.scatter( y=player['TotalGames'], x=player['BirthYear'], color='lightblue');
plt.xlim(player['BirthYear'].min()-5, player['BirthYear'].max()+5);
plt.ylim(-5, player['TotalGames'].max()+5);
plt.title('Games played by birth year by player');


# In[40]:


g=player.groupby(['Country', 'Region'], as_index=False)    ['name'].count()    .rename(columns={'name':'count'})    .sort_values('count', ascending=False)
g['Country2']=''
g.loc[g['count']>4,'Country2']=g['Country']
gg=g.groupby('Region', as_index=False)     ['count'].sum()
    
fig, ax = plt.subplots(1,2,figsize=(14,6))
ax[1].pie(x=gg['count'], labels=gg['Region']);
ax[0].pie(x=g['count'], labels=g['Country2']);
ax[0].set_title('Player by country')
ax[1].set_title('Player by region');


# In[41]:


fig, ax = plt.subplots(1,1, figsize=(9,5))
ax = sns.violinplot(x="Region", y="Rating", data=player)
plt.title('Players by region and Max Rating');


# ## Correlation
# Correlations are roughly in line with what one could expect.

# In[42]:


sns.heatmap(player.corr(), annot=True)
plt.title('Correlation of player attributes');


# ## t-SNE
# 
# t-SNE is a way to represent a higher dimensional dataset on a two-dimensional plane.
# 
# Kasparow is an outlier, having played a small number of games and very high rating; the majority of his carreer lies before 2000. The current ranking leader having well exposed spot over the data scatter. In another dimension 
# Tiviakov is a prominent exposed spot havin played a high number of games, however just reached pos 20 on a top 100 table.

# In[43]:


#ToDo: How can I fix the random_state ? 
tsne = TSNE(n_components=2, init='pca', random_state=1)
X_tsne = tsne.fit_transform(player[['BestRank', 'TotalGames', 'BirthYear',
                                    'TimeListed', 'Rating']])
clr={'Carlsen, Magnus':'r',
     'So, Wesley':'b',
     'Kramnik, Vladimir':'darkgrey',
     'Caruana, Fabiano':'g',
     'Aronian, Levon':'y',
     'Polgar, Judit': 'pink',
     'Kasparov, Garry': 'k',
     'Tiviakov, Sergei': 'cyan'}

fig, ax  = plt.subplots(1,1,figsize=(17, 6))
ax.scatter(X_tsne[:, 0], X_tsne[:, 1])
for name in clr:
    i=player[player.name==name].index[0]
    plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=clr[name]) 
    plt.text(s=name, x=X_tsne[i, 0]+10, y=X_tsne[i, 1]+10)
plt.title('t-SNE, hightlight prominent players');   


# In[44]:


df=pd.DataFrame(X_tsne, columns=[['X','Y']])
t=df[df['X']==df.X.max()].index
player.loc[t]


# ## Linear Regression
# 
# It is hard to find a linear regression model as players are at different stages of their career. 
# This is illustrated on an hand picked examples. The top row show four GMs current topping the rating list, however the regression is dominated by their rating progression. The second row show four GMs playing at the top during the complete given time series, they might have peaked earlier, but the regressions coefficient is still positive due to rating inflation. The bottom row shows four GMs dropped out of the leaders circles.

# In[ ]:


def drawRegression(name, ax):
    
    elo_Y = elo[elo.name==name]['rating'].values
    elo_X = elo[elo.name==name]['PubRunMonth']          - min(elo[elo.name==name]['PubRunMonth'])
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(elo_X.values.reshape(-1, 1), elo_Y)
    
    ax.scatter(elo_X,elo_Y,  color='lightblue')
    x=np.linspace(0,max(elo_X),max(elo_X));
    f=regr.coef_*x+regr.intercept_
    ax.plot(f,color='darkblue', alpha=0.5)
    
    ax.set_ylim(2600, 2900)
    ax.set_xlim(-5,210)
    ax.set_title(name)


# In[ ]:


names=['Carlsen, Magnus'   , 'So, Wesley'       ,'Vachier-Lagrave, Maxime','Nakamura, Hikaru',
       'Anand, Viswanathan', 'Kramnik, Vladimir','Adams, Michael'         ,'Gelfand, Boris',
       'Shirov, Alexei'    , 'Polgar, Judit'    ,'Karpov, Anatoly'        ,'Short, Nigel D.']
fig, ax = plt.subplots(3,4,figsize=(17,10));
for i in range(3):
    for j in range(4):
        drawRegression(names.pop(0),ax=ax[i,j])


# In[ ]:





# #### Change log - work in progress 
# 29/06/2017 First look at the dataset and the data structure<br>
# 30/06/2017 Verifying title; Number1 plot<br>
# 02/07/2017 Continue Number1 plot<br>
# 03/07/2017 Number1 plots<br>
# 04/07/2017 Elo over time plots, more name corrections<br>
# 05/07/2017 Elo averages, some stats, found chess unicode symbols &#9816; <br>
# 06/07/2017 Games played analysis<br>
# 07/07/2017 Player statistics<br>
# 09/07/2017 Prettifying, Correlation on player attributes<br>
# 10/07/2017 Commentary <br>
# 11/07/2017 Commentary, Start statistics<br>
# 16/07/2017 Regression model and adjustments <br>
# 17/07/2017 Regression model and adjustments <br>
# 20/07/2017 Regression model and adjustments <br>
# 25/07/2017 Regression write-up <br>
# 27/07/2017 Regression write-up <br>
# 30/07/2017 Player's t-SNE <br>
# 31/07/2017 Clean up for kaggle, Commentary <br>
# 01/08/2017 Commentary, prettyfying <br>
# 02/08/2017 Refactoring, Age <br>
# 06/08/2017 More on Age<br>
# 07/08/2017 Clean up <br>
# 08/08/2017 Frequency Chart<br>
# 20/08/1017 Rebuild published version<br>
# 12/10/2017 Corrections<br>
# 13/10/2017 Regions pie<br>
# 14/10/2017 Candlesticks<br>
# 16/10/2017 Polarplots <br>
# 17/10/2017 Prettify Polarplots <br>
# 18/10/2017 Minor corrections <br>
# 19/10/2017 Joy plot age distributions <br>
# 20/10/2017 Rating moves <br>
# 22/10/2017 General Prettifying and commentary <br>
# 
# #### Open points for future work:
# * World champions by age
# * Number of games vs age
# * Work with Dates
# 
# #### References:
# [wiki ELO]      https://en.wikipedia.org/wiki/Elo_rating_system<br>
# [wiki Smirnov]  https://en.wikipedia.org/wiki/Pavel_Smirnov <br>
# [wiki Afromeev] https://en.wikipedia.org/wiki/Vladimir_Afromeev <br>
# [srlmayor]      https://www.kaggle.com/srlmayor/exploring-top-100-chess-players-2000-2017<br>

# In[ ]:




