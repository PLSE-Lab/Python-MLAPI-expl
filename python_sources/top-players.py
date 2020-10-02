#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# In this kernel, we are going to look at the statistics of football matches in the last six years (between 2011-2017) in order to find interesting discoveries and facts.

# In[ ]:


#importing helping libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the data and clean it in order to be used 
# 1. First, we will join the two tables into one big table 

# In[ ]:


#Loading the dataset
events = pd.read_csv('../input/events.csv')
data_ginf = pd.read_csv('../input/ginf.csv')
df=data_ginf.merge(events,how='left')
df.head()


# In[ ]:


df.info()


# 2. We will convert those numbers to its corresponding names using the text file attached 

# In[ ]:


new=dict()
with open('../input/dictionary.txt','r') as f:
    data=f.read()
data=data.split('\n\n\n')
for i in range(len(data)):
    if data[i]:
        variable_name = data[i].split('\n')[0]
        values = data[i].split('\n')[1:]
        new[variable_name]={int(s.split('\t')[0]):s.split('\t')[1] for s in values}
        print(data[i])
for name in new:
    df[name]=df[name].map(new[name])


# 3. Finally, separate the data of the five leagues into 5 data frames in case we need them and let's begin our investigation and analysis 

# In[ ]:


bundesliga=df[df['country']=='germany']
ligue1=df[df['country']=='france']
laliga=df[df['country']=='spain']
premiereleague=df[df['country']=='england']
seriea=df[df['country']=='italy']
print('Bundes Liga data shape:',bundesliga.shape)
print('Ligue 1 data shape:',ligue1.shape)
print('La Liga data shape:',laliga.shape)
print('Premiere League:',premiereleague.shape)
print('Serie A data shape:',seriea.shape)


# # Who are the best strikers in Europe !? 

# ## Top Scorers
# The striker is one of the most important positions in the football team, Having a player that can put the ball in the net more than others can be the most important player in the football team, So how do we find them and identify them?  
# The easiest way is to make a table with player names and in front of them the goals they scored, in other words, using the **top scorers** table
# 

# In[ ]:


def top_scorers(data):
    goals=data.loc[data['is_goal']==1&(data['situation']!='Own goal')] #excluding own goals(we are looking for strikers who can score in the opponent's net)
    goals=goals.groupby('player')['is_goal'].sum().reset_index().rename(columns={'is_goal':'G'}).sort_values(by='G',ascending=False)
    goals=goals[['player','G']].set_index('player')
    return goals
player_tp=top_scorers(df)
print('G : Goals')
player_tp[:20]


# In[ ]:


def pointgraph(data,x,s):
    plt.figure(figsize=(12,8))
#     data=top_scorers(data)
    ax=sns.scatterplot(x=data[x],y=data.index,s=700,alpha=.7)
    for i,j in enumerate(data[x]):
        ax.text(j-2,i-0.2,int(j),color='white')
    plt.title(s)
    plt.tight_layout()
    plt.show()
pointgraph(player_tp[:20],'G','Top 20 Scorers')


# Lionel Messi, Cristiano Ronaldo, and Zlatan Ibrahimovic are, of course, very good strikers playing in very good teams 
# with no doubt, Here is the **top scorers** or the players who score the most goal, but are these the top strikers who can score whenever they saw a chance??
# I think then it's a bad idea to just check the top scorers' table to search for the strikers, it' just a wrong way   

# ## Goals Per Match 
# It's time to make change the comparison a bit, our first and most simple modification is counting the **goals per match**, just by dividing the number of goals over the number of matches played by that player The players that played fewer matches of quarter of the most player are excluded, in order to prevent player who played only one match with 2 goals to have 200% (minimum played 31 matches)

# In[ ]:


def GPM(data):
    x=data[data['situation']!='Own goal']
    y=x.groupby(['id_odsp','player'])['is_goal'].sum().reset_index().rename(columns={'id_odsp':'Matches','is_goal':'G'})
    xy=y.groupby('player').agg({'Matches':'count','G':"sum"})
    xy['GPM']=xy['G']/xy['Matches']
    xy=xy[xy['Matches']>xy['Matches'].max()*0.25]
#     print(xy['Matches'].max()*0.25)
    xy.sort_values(by='GPM',ascending=False)
    return xy.sort_values(by='GPM',ascending=False)

print('G : Goals')
print('GPM : Goals Per Match')
player_gpm=GPM(df)
player_gpm[:20]


# In[ ]:


def twin_barplot(data1,x1,y1,s1,data2,x2,y2,s2):
    plt.figure(figsize=(20,10))

    plt.subplot(121)
    ax=sns.barplot(x=x1,y=y1,data=data1)
    for i,j in enumerate(data1[x1][:20]):
        ax.text(0.5,i,j,weight='bold')
    plt.title(s1)
    plt.ylabel("")
    plt.subplot(122)
    plt.subplots_adjust(wspace=.5)
    ax=sns.barplot(x=x2,y=y2,data=data2)
    for i,j in enumerate(player_gpm[x2][:20]):
        ax.text(0.01,i,j,weight='bold')
    plt.title(s2)
twin_barplot(player_tp[:20],'G',player_tp.index[:20],'Goals',player_gpm[:20],'GPM',player_gpm.index[:20],'Goals Per Match')


# Messi still hold the lead  (good news for his fans) with about 0.04 difference between him and Cristiano, also Lewandowski dropped from the 4th to 6th position, instead Luis Suarez jumped from 8th to 3rd , New faces appeared here like Huntelaar and Neymar, they may be not involved in all matches so the number of Goals is not so high but regarding the small number of matches they participated in, maybe they have a chance to compete for the top 20.
# 

# ## Non-Penalty Goal Per Match 
# 
# On the other hand, The penalties considered the easiest  scoring opportunity, and penalty specialist is another talent that we aren't looking for now (some defenders are penalty specialists in doesn't need to be a striker to score a penalty), And because not all the players took the equal numbers of penalties (some players don't take any penalty at all), so it's something available for all players, So we are going to exclude the **penalty goals** from our calculations 

# In[ ]:


def NPGPM(data):
    x=data[(data['situation']!='Own goal')&(data['location']!='Penalty spot')]
    y=x.groupby(['id_odsp','player'])['is_goal'].sum().reset_index().rename(columns={'id_odsp':'Matches','is_goal':'NPG'})
#     print(y[y['player']=='sergio aguero'])
    xy=y.groupby('player').agg({'Matches':'count','NPG':"sum"})
    xy['NPGPM']=xy['NPG']/xy['Matches']
    xy=xy[xy['Matches']>31]
#     print(xy['Matches'].max()*0.25)
    
    return xy.sort_values(by='NPGPM',ascending=False)
print('NPG : Non-Penalty Goals')
print('NPGPM : Non-Penalty Goals Per Match')
player_npg=NPGPM(df)
player_npg[:20]


# ok, WOW!! Messi scored around 15% of his goals from penalties (30 penalties), Cristiano scored around 22% (43) and Zalatan scored around 23%(35), no wonder most of them fell down in the **NPGPM** table, Luiz Suarez jumped to the 2nd
# 
# Let's have a look at removing penalties from our old top scorers 

# In[ ]:


def double_bargraph(data,s):
#     print(data)
    ax=data.plot(kind='barh',figsize=(20,20),edgecolor='k',linewidth=1)
    plt.title(s)
    plt.legend(loc='best',prop={'size':40})
    for i,j in enumerate(data.iloc[:,1]):
        ax.text(0.5,i,j,weight='bold')
    for i,j in enumerate(data.iloc[:,0]):
        ax.text(0.5,i-0.2,j,weight='bold',color='white')
xx=pd.concat([player_tp,player_npg],axis=1).fillna(0)
double_bargraph(xx.sort_values(by='G',ascending=False)[['G','NPG']][:20],'Goals Vs. Non-Penalty Goals')


# **The main fault here that not all players have the same opportunities to score **

# ## Expected Goals

# We stated before that what we are looking for is a striker who can scores whenever he got a chance, so now we know How many attempts each player make on goal that average player can score?  
# To discover this value we should look for two qualities 
# 1. The players with the most attempts made on goal, it's clear that the opportunities and chances created to each player depends on the striker skills and his teammates skills, So it shouldn't be weird to find the most attempts created to the players playing in the strongest clubs in their league, In addition to playing in front of weak opponents 
# 2. The best players to use these chances in order to score a goal
# 
# <span style='color:red'>Note:</span> All calculations made on Expected goals exclude the penalties in order to be fair enough 
# 
# So First let's see the Distribution of goals and attempts and their mean (we will include only the players with 10 goals or more to remove non-striker players)
# 

# In[ ]:


def ExpG(data):
    x=data[(data['location']!='Penalty spot')&(data['event_type2']!='Own goal')&(data['event_type']=='Attempt')]
    y=x.groupby(['player','id_odsp']).agg({'is_goal':'sum','event_type':'count'}).reset_index()
    y['total']=y['is_goal']/y['event_type']
    y=y.groupby('player').agg({'is_goal':'sum','total':'mean','event_type':'sum','id_odsp':'count'})
    y['total2']=y['event_type']/y['id_odsp']
    y['GPM']=y['is_goal']/y['id_odsp']
    y=y[y['is_goal']>18]
    y.columns=['NPG','Avg GPA','Attempts','Matches','APM','GPM']
    return y
print('NPG : Non-Penalty Goals')
print('Avg GPA : Average Goal Per Attempt')
print('APM : Attempt Per Match')
print('GPM : Goal Per Match')

ExpG(df).sort_values(by='Attempts',ascending=False)[:20]


# **the most player made attempts on goal was Cristiano with 6.4 attempts per match (which is a way high) with total attempts 1138, Although AvgGPA is 0.14 which means that he has every match about 6 attempts on goal, avg scoring goals is 14% of those attempts he scored 0.87 goals per match 
# Most of this list are players playing in top teams, which feed our theory that playing in top teams means more goals, but what if we sort them with Avg GPA to know even if you have fewer attempts because you playing in a small team or any other reason, How much often will you score?**

# In[ ]:


def bar(data,x,y,s ):
    fig=plt.figure(figsize=(15,15))
    ax=sns.barplot(x=x,y=y,data=data)
    plt.title(s)
    for i,j in enumerate(data[x]):
        ax.text(0.01,i,j,weight='bold')
player_expg=ExpG(df).sort_values(by='Avg GPA',ascending=False)[:20]
bar(player_expg,'Avg GPA',player_expg.index,'Average Goals Per Match')        


# **Now the Controversial part, I myself panicked when I saw this table and doubted my calculations, but then I remembered we are searching for the best striker who could score from slight chance, so we asking this value not all values and attributes that make a modern forward like dribbling and positioning without ball, even assisting plus being striker, But all of this is out of our scope, so now our result make a little more sense.  
# Bas Dost is Dutch player who plays in Sporting FC, he is superb in Aerial Duels, Headed attempts and Finishing, He isn't key player like Messi or decisive player like Alexis Sanchez, but if he had the chance to soccer, it's hard for him to miss it , he has highest Avg GPA with about 0.4 between him and Javier Hernandez(Chicharito) , he played in small club in Europe so his number of goals and attempts lower than others like Higuain in Jevuents.  
# Also, Javier Hernandez or Chicharito, we all saw what he has done in World cup with his national team (Mexico) in front of Germany and South Korea, but he played for Leverkusen so he won't have the same support others in strong teams have   **

# 

# ### What we miss??
# * Of course, more analysis should be done, who scored the most difficult opportunities? and do they score the dead chances or very critical chances ?? 
# * The top of our table had a low number of attempts! That should make us search more in the answer to a question, why this happened? maybe the highest player has the chance to score through a small number of attempts are the Central Forwards (most of the players in this list are center forwards), But maybe the low number of attempts means that those player have a problem in positioning so he didn't have the ball in front of a goal regularly 
# * We are comparing players that play in different leagues with different styles   
# 
# So does it really differ !! let's see how many goals each league witnessed.

# In[ ]:


def GPL(data,colors,labels):
    plt.figure(figsize=(15,12))
    plt.xticks(list(range(10)))
    plt.xlabel('Goals Per Match')
#     plt.legend(loc='best',prop={'size':40})
    for d,c,s in zip(data,colors,labels):
        d=d.groupby('id_odsp')['is_goal'].sum()
        sns.kdeplot(d,shade=True,color=c,label=s)
        plt.axvline(d.mean(),linestyle='dashed',color=c,label=(s+' Mean'))
#FOR the honor of League winners this year, i changed the colors to be the color of the winner teams shirts
GPL([bundesliga,laliga,ligue1,seriea,premiereleague],['r','w','g','k','b'],['BundesLiga','LaLiga','Ligue1','SerieA','PremiereLeague'])


# **Ok there's a clear difference between all leagues and the premier league, The English league has less number of goals per match, Whether it's because of it has better defenders or their attackers are much worse, and I think it's the attackers are much worse**

# So let's check the top 5 in each league with the same steps we did before 
# ## 1- Premiere League
# <img src="https://www.premierleague.com/resources/ver/i/elements/premier-league-logo-header.png" width="300px">
# 

# In[ ]:


# top scorer in PL
def pointgraph(data,x,s):
    plt.figure(figsize=(12,8))
#     data=top_scorers(data)
    ax=sns.scatterplot(x=data[x],y=data.index,s=700,alpha=.7)
    for i,j in enumerate(data[x]):
        ax.text(j-.5,i-0.2,int(j),color='white') #we will overide the original function just to update the x position of text 
    plt.title(s)
    plt.tight_layout()
    plt.show()
def league_repr(data,n):
    tp=top_scorers(data)
    gpm=GPM(data)
    npgpm=NPGPM(data)
    xx=pd.concat([tp,npgpm],axis=1).fillna(0)
    expg=ExpG(data)
    
    pointgraph(tp[:n],'G','Top Scorers')
    twin_barplot(tp[:n],'G',tp.index[:n],'Goals',gpm[:n],'GPM',gpm.index[:n],'Goals Per Match')
    double_bargraph(xx[['G','NPG']].sort_values(by='G',ascending=False)[:n],'Goals Vs.Non-Penalty Goals')
    bar(expg.sort_values(by='Avg GPA',ascending=False)[:n],'Avg GPA',expg.sort_values(by='Avg GPA',ascending=False).index[:n],'Average Goals Per Attempt')
    print('sorted by number of attempts')
    display(expg.sort_values(by='Attempts',ascending=False)[:n])
# pl_tp=top_scorers(premiereleague)
# pl_gpm=GPM(premiereleague)
# pl_xx=pd.concat([top_scorers(premiereleague),NPGPM(premiereleague)],axis=1).fillna(0)
# pl_expg=ExpG(premiereleague)
# pointgraph(pl_tp[:20],'G','Top Scorers in Premiere League')
# twin_barplot(pl_tp[:20],'G',pl_tp.index[:20],'Goals',pl_gpm[:20],'GPM',pl_gpm.index[:20],'Goals Per Match')
# double_bargraph(pl_xx[['G','NPG']].sort_values(by='G',ascending=False)[:20],'Non-Penalty')
# bar(pl_expg.sort_values(by='Avg GPA',ascending=False)[:20],'Avg GPA',pl_expg.sort_values(by='Avg GPA',ascending=False).index[:20],'Average Goals Per Attempt')
# pl_expg.sort_values(by='Attempts',ascending=False).head(20)

league_repr(premiereleague,20)


# **2- Ligue 1**
# <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/Ligue_1_Logo.svg/1200px-Ligue_1_Logo.svg.png" width="200px">
# 

# In[ ]:



league_repr(ligue1,20)


# **3-Serie A**
# <img src="https://upload.wikimedia.org/wikipedia/en/0/02/Serie_A_logo_%282018%29.png" width='200px'>

# In[ ]:


league_repr(seriea,20)


# **4- La Liga**
# <img src="https://a2.espncdn.com/combiner/i?img=%2Fi%2Fleaguelogos%2Fsoccer%2F500%2F15.png" width='300px'>

# In[ ]:


league_repr(laliga,20)


# **5- Bundesliga**
# <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Bundesliga_logo_%282017%29.svg/1200px-Bundesliga_logo_%282017%29.svg.png" width='300px'>

# In[ ]:


league_repr(bundesliga,20)


# **Conclusion**
# I think now we can conclude that having more chances lower you on Avg GPA table, But more chances mean more goals, why do the top scorers don't make it for the final table? For the same reason, top scorers mean more goals, and more goals mean more chances and attempts on goals so they going to miss more attempts so that they dropped the table   
# And now the question that needs more investigation are those players outliers!? Should they score more than that to judge them !?   
# It's logical that not all the players play an equal amount of matches and participate in match equal amount of minutes, but does that the case in every league !? and if that not the best way to identify the real striker, what would be the best metrics? More Analysis is needed.  
# I will continue working on this kernel to find it and if anyone has an opinion to help, it will be appreciated   

# # Refrences 
# 1. https://statsbomb.com/2015/11/flavio/
# 2. https://towardsdatascience.com/a-simple-method-to-predict-player-performance-using-fantasy-football-data-8b2d3adb3a1a
# 3. https://talksport.com/football/234007/who-won-what-around-europe-201617-season-league-champions-and-cup-winners-revealed/
# 4. https://statsbomb.com/2014/01/you-might-think-messi-is-the-best-goalscorer-in-each-of-the-past-4-seasons-you-would-be-wrong/
# 5. http://11tegen11.net/2014/01/15/how-to-scout-goal-scoring-talent/

# In[ ]:




