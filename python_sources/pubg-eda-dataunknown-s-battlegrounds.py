#!/usr/bin/env python
# coding: utf-8

# # PUBG Prediction Analysis

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


pubg = pd.read_csv('../input/train_V2.csv')


# In[ ]:


pubg.head()


# In[ ]:


#Every column there is !
pubg.columns


# In[ ]:


#I am going to work on the first 1million rows of data since my kernel can't really handle 4.2m otherwise
pubg = pubg[:1000000]


# In[ ]:


pubg.shape


# ### Let's see if we have any missing data !

# In[ ]:


pubg.describe().iloc[0,:] 


# In[ ]:


get_ipython().run_line_magic('matplotlib', "inline # so that I don't have to type plt.show() everytime")


# ### Visualising missing data if any !

# In[ ]:


sns.heatmap(pubg.isnull(),yticklabels=False)
f = plt.gcf()
f.set_size_inches(10,8)


# Before starting with EDA or anything else, I am going to change the matchTypes to the three main ones which a pubg lover personally knows and loves !

# In[ ]:


pubg['matchType'].value_counts()


# In[ ]:


#converting them
pubg['matchType'].replace(['squad-fpp','duo-fpp','solo-fpp','normal-squad-fpp','crashfpp','normal-duo-fpp','flaretpp',
                           'normal-solo-fpp','flarefpp','normal-squad','crashtpp','normal-solo','normal-duo'],
                          ['squad','duo','solo','squad','others','duo','others','solo','others','squad',
                          'others','solo','duo'],inplace=True)
#others matchtype represent names of types which I can't personally distinguish as solo/squad/duo even after searching on kaggle


# In[ ]:


pubg['matchType'].value_counts()


# # 1. EDA (Exploratory Data Analysis)

# ## 1.1 Assists

# ![Assists](http://cdn3-www.gamerevolution.com/assets/uploads/2018/09/pubg-mobile-cant-play-with-friends.jpg)

# Assists : Number of enemy players this player damaged that were killed by teammates

# In[ ]:


pubg['assists'].value_counts().to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


sns.factorplot('assists','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(15,8)
plt.title('Assists vs winPlacePerc')


# Now personally speaking I don't really think assists should have been given importance in solos since you can't really knock someone is solos. I don't understand what it represents, maybe killing someone who has already been shot by someone else and has low health.

# Nevertheless, Having assists till 5-6 values is a good indication of having a good winPlacePercentage.

# But that decreases after 8-9 assists. This could potentially represent players who are good only at assists than  killing and winning ! Remember this is going to matter more in squads since assists might help your team to go forward in the game but if you are only good at assisting and not killing chances of victory are substantially reduced !

# ## 1.2 Boosts
# 

# Boosts : Number of boost items used. (Energy Drinks, PainKillers, Adrenaline Syringe)

# In[ ]:


pd.crosstab(pubg.boosts,pubg.matchType).style.background_gradient(cmap='summer_r')


# In[ ]:


x = pubg[pubg['boosts']<6].count()[0]/pubg.shape[0]*100
print('Percentage of players who used less than 6 boosts per game : ',x)


# Majority of the people use less than 6 boosts per game ! 

# In[ ]:


sns.barplot('boosts','winPlacePerc',hue='matchType',data=pubg)
plt.title('Boosts vs WinPlacePerc')
f = plt.gcf()
f.set_size_inches(15,8)


# From the graph and cross tab above, it is evident that players using more than 5-6 boosts per game are less in number but have higher chances of winning ! This is quite understandable in itself.

# ## 1.3 DamageDealt

# Damage Dealt : Total damage dealt. Note: Self inflicted damage is subtracted

# Damage dealt is a continuous feature. So plotting has to be in a similar fashion

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,6))
sns.distplot(pubg.damageDealt,kde=True,ax=ax[0])
plt.title('Density Graph of DamageDealt')
sns.scatterplot('damageDealt','winPlacePerc',hue='matchType',ax=ax[1],data=pubg)
plt.title('DamageDealt vs winPlacePerc')


# In[ ]:


pubg.groupby(['matchType'])['damageDealt'].mean()


# There isn't much difference between the means of damage dealt between the major three classes of matchTypes. But there is one thing which can be noted from the scatter plot above. See below :

# In[ ]:


pubg[(pubg['matchType']=='solo')&(pubg['winPlacePerc']==0)&(pubg['damageDealt']>500)].count()[0]


# So what I did in the above line is basically, all those players who have dealt more than 500 damage to enemies in solo match but still managed to have a winPlacePerc of 0.0. Is this badluck or a case of outliers ?

# In[ ]:


pubg[(pubg['matchType']=='solo')&(pubg['winPlacePerc']==0)&(pubg['damageDealt']>1500)].count()[0]


# Well now the above one is just sheer badluck !

# In[ ]:


for i in range(5):    
    plt.title('DamageDealt with winPlacePerc = '+str(0.4+(i/10))+' - '+str(0.4+(i+1)/10))
    pubg[(pubg['winPlacePerc']>=0.4+(i/10))&(pubg['winPlacePerc']<0.4+((i+1)/10))].damageDealt.plot.hist(color='red',edgecolor='black',bins=20)
    plt.show()


# The data above shows people who didn't really deal a lot of damage but still ended up getting a good place ! This well might be because of people hiding themselves for majority of the game (proners and campers) !

# ## 1.4 HeadShot Kills

# ![HeadshotKills](http://images2.minutemediacdn.com/image/upload/c_fill,w_912,h_516,f_auto,q_auto,g_auto/shape/cover/sport/5b2d10be3467ac14d0000006.jpeg)

# HeadShotKills : Number of enemy players killed with headshots.

# This should give us a better idea of how much probability a player has at winning than other similar attributes like kills since headshots are generally difficult to take and are taken by more skilled players

# In[ ]:


pubg['headshotKills'].value_counts().to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


sns.factorplot('headshotKills','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(10,8)
plt.title('Headshot Kills vs winPlacePerc')


# HeadshotKills is very informative. As can be seen from the plot above, the more the headshots people have, the better the chances of winning.

# #### Skill speaks for itself

# In[ ]:


x = pubg[(pubg['headshotKills']>5)&(pubg['winPlacePerc']>0.8)].count()[0]/pubg[pubg['headshotKills']>5].count()[0]*100
print('Percentage of people having more than 5 headshot kills and having winPlacePerc > 0.8 : ',x)


# ## 1.5 Heals

# ![FirstAid](https://d1nglqw9e0mrau.cloudfront.net/assets/images/game/icons/heal_firstaid-57fc8495.png)

# Heals : Number of healing items used.(First aids, Bandages,MediKits)

# In[ ]:


sns.factorplot('heals','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(25,15)
plt.title('Heals vs winPlacePerc')


# There are higher chances of ones winning if a person has used heals more than 0 times and understandably so ! Also solos have players using more healing items than duos or squads.

# ## 1.6 Kill Place
# 

# KillPlace : Ranking in match of number of enemy players killed. (Rank 1 means most players killed in that match)

# In[ ]:


sns.scatterplot('killPlace','winPlacePerc',data=pubg[pubg['matchType']=='solo'])
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('KillPlace vs winPlacePerc')


# There seems to be a negative correlation between the two features and understandably so since if I am at rank 1 in kills in a match, I probably have achieved a good rank overall !

# In[ ]:


sns.heatmap(pubg[['killPlace','winPlacePerc']].corr(),annot=True)


# The negative correlation can be seen from the above heatmap. I mean obviously the more kills I have, the lesser killPlace rank I will have, and hence better will be my chances of winning !

# ### Kill Points is a feature which is calculated by rankpoints feature. RankPoints is inconsistent and will be deprecated so I won't be using both of these features !

# In[ ]:


pubg.drop(['killPoints','rankPoints'],axis=1,inplace=True)


# In[ ]:


pubg.shape[1] #drop success !


# ## 1.7 Kill Streaks

# KillStreaks : Max number of enemy players killed in a short amount of time

# In[ ]:


sns.factorplot('killStreaks','winPlacePerc',hue='matchType',data=pubg)
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('KillStreaks vs winPlacePerc')


# There isn't much that can be said about killstreaks since we can't really see a pattern ! I mean, I would expect someone with a killstream of 5 or above to have better chances of winning ! But the data doesn't follow the same intuition ! A lot of data who had 8 as killstreak only has < 0.3 as winPlacePerc

# ## 1.8 Kills

# Kills : Number of enemy players killed.

# In[ ]:


f,ax = plt.subplots(2,2,figsize=(15,10))
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='solo'],ax=ax[0,0],palette='Blues')
ax[0,0].set_title('Solos')
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='duo'],ax=ax[0,1],palette='Blues_r')
ax[0,1].set_title('Duos')
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='squad'],ax=ax[1,0],palette='OrRd')
ax[1,0].set_title('Squads')
sns.barplot('kills','winPlacePerc',data=pubg[pubg['matchType']=='others'],ax=ax[1,1],palette= 'OrRd_r')
ax[1,1].set_title('Others')


# In all the matches, having higher kills generally corresponds to higher chances of winning ! There are some inconsistencies after 15 kills or higher in solos, duos, squads but that might be becuase players who want to fight don't really play to win, they just want to kill !

# In[ ]:


pubg['kills'].max()


# Someone killed 65 players in a single match !!!!!!!!! You gotta be kidding me ! Let's see who he is :

# In[ ]:


pubg['kills'].argmax()


# In[ ]:


pubg.iloc[334400].to_frame()


# DAMN DAWGGGG !!!!! You got a bright future in your scope !

# In[ ]:


sns.heatmap(pubg[['kills','winPlacePerc']].corr(),annot=True)


# It is quite understandable to say the more kills you have the more chances you have to win/survive.

# ## 1.9 LongestKill

# Longest Kill : The longest distance in meters a person has killed someone.
# 
# 

# Can be misleading sometimes since I can kill someone 1000m away who is knocked out ! But let's see

# In[ ]:


print('Longest Kill :',str(pubg['longestKill'].max()) + ' - AWM + 15x')
print('Shortest Kill :',str(pubg['longestKill'].min())+' - Pan Kill')


# In[ ]:


sns.scatterplot('longestKill','winPlacePerc',data=pubg)
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('LongestKill vs WinPlacePerc')


# Having high range kills would mean a better player and kinda understandably so!

# In[ ]:


x = pubg[(pubg['longestKill']>200)&(pubg['winPlacePerc']>0.8)].count()[0]/pubg[pubg['longestKill']>200].count()[0]*100
print('Percentage of people who had LongestKills > 200m and winPlacePerc > 0.8 :',x)


# So people who had killed others with distance more than 200m have good chances of survival/winning !

# ## 1.10 MatchDuration

# MatchDuration : Duration of match in seconds

# In[ ]:


sns.distplot(pubg.matchDuration)
f = plt.gcf()
f.set_size_inches(10,8)
plt.title('Match duration Density Graph')


# In[ ]:


1400/60,1800/60


# So by data, the fights usually start at around 23 minutes of match start and majority of them die in this period as well(the first peak of the graph). This can be fourth or fixth last circle ! Then at 30 minutes, due to the last/second last circle, people have no other alternative apart from a one on one (see the second peak of the graph)

# ## 1.11 Revives

# ![](https://cdn1.alphr.com/sites/alphr/files/styles/insert_main_wide_image/public/2018/01/pubg_guide_-_helping_a_team_mate.jpg?itok=kOzC7EQo)

# Revives : Number of times this player revived teammates.

# NOTE : This is going to be meaningless for solos !

# In[ ]:


pubg.groupby(['matchType'])['revives'].value_counts().to_frame().style.background_gradient(cmap='summer_r')


# Well it is easily understandable and can be verified by the above cross tab that squads are generally going to have moe revives than any other matchtype. Let's take a closer look at it !

# In[ ]:


sns.catplot('revives','winPlacePerc',data=pubg[pubg.matchType=='squad'],jitter=False)
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('Squad revives vs winPlacePerc')


# The more the revives a player performs, the better his chances becomes at winning. But after 6 revives per player, the winning percentage decreases or a specific pattern can't be observed. This may be because you are being knocked out more often than not would make you a NOOB !

# ## 1.12 RideDistance
# 

# ![](https://cdn1.alphr.com/sites/alphr/files/styles/insert_main_wide_image/public/2018/01/pubg_guide_-_vehicle_ride.jpg?itok=dO-nSud7)

# In[ ]:


print('The highest ride distance by a player :',pubg['rideDistance'].max())


# How can someone travel 33KM in a single match ??? Can be a potential outlier !

# In[ ]:


sns.scatterplot('rideDistance','winPlacePerc',data=pubg)
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('rideDistance vs winPlacePerc')


# In[ ]:


for i in range(5):    
    plt.title('RideDistance with winPlacePerc = '+str(0.4+(i/10))+' - '+str(0.4+(i+1)/10))
    pubg[(pubg['winPlacePerc']>=0.4+(i/10))&(pubg['winPlacePerc']<0.4+((i+1)/10))&(pubg['rideDistance']<10000)].rideDistance.plot.hist(color='red',edgecolor='black',bins=20)
    plt.show()


# Ride distance increases for the players with hgh winPlacePerc. The increase is not substantial but can be seen !

# In[ ]:


sns.heatmap(pubg[['rideDistance','winPlacePerc']].corr(),annot=True)


# There is still a positive correlation between the two !

# ## 1.13 RoadKills
# 

# RoadKills : Number of kills when in vehicle

# In[ ]:


pubg['roadKills'].value_counts()


# In[ ]:


sns.catplot('roadKills','winPlacePerc',kind='boxen',data=pubg)
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('RoadKills vs winPlacePerc')


# Having higher roadkills means better chances of winning ! I didn't expect something like this. I personally have killed a lot of people and destroyed a lot of vehicles when in a vehicle, makes me a pro player I guess !

# ## 1.15 Team Kills
# 

# ![](https://www.greenmangaming.com/blog/wp-content/uploads/2017/07/PUBG_blogbanner.jpg)

# Team Kills : Number of times player killed team mates

# In[ ]:


sns.violinplot('teamKills','winPlacePerc',hue='matchType',data=pubg[(pubg['matchType']=='squad')|(pubg['matchType']=='duo')],split=True)
f = plt.gcf()
f.set_size_inches(10,8)
plt.title('TeamKills vs winPlacePerc (duos and squads only)')


# Players who do have some team kills are comparitively less in numbers who have higher winPlacePerc ! This is something I am glad to see, because of the traitors and my brothers also who have betrayed me in an otherwise great match !

# ## 1.16 VehiclesDestroyed
# 

# VehiclesDestroyed : The number of vehicles destroyed by a single player !

# In[ ]:


sns.factorplot('vehicleDestroys','winPlacePerc',hue='matchType',data=pubg)
f=plt.gcf()
f.set_size_inches(10,8)
plt.title('VehiclesDestroyed vs winPlacePerc')


# Damn ! Someone who has destroyed even a single vehicle or more proves to be having a higher chance of survival and winPlacePercentage. Terminator !

# ## 1.17 WalkDistance

# Walkdistance : Distance walked by player in metres

# In[ ]:


#below represents people who haven't moved a single step but still had more than 0 kill
pubg[(pubg['walkDistance']==0)&(pubg['kills']>0)].count()[0]


# Cheaters ! Hackers ! Might even have one data row of my hacker brother !

# In[ ]:


pubg.drop(pubg[(pubg['walkDistance']==0)&(pubg['kills']>0)].index,inplace=True)


# In[ ]:


sns.scatterplot('walkDistance','winPlacePerc',data=pubg)
f = plt.gcf()
f.set_size_inches(8,6)
plt.title('WalkDistance vs winPlacePerc')


# In[ ]:


# Most of the data have walkdistance value < 10000m. So that's why I have put a limit on it in below code
pubg[(pubg['winPlacePerc']>0.9)&(pubg['walkDistance']<10000)].walkDistance.plot.hist(bins=50,edgecolor='black')
f = plt.gcf()
f.set_size_inches(12,9)
plt.title('WalkDistance Frequency')


# The above graph looks like a normal distribution. This checks with the fact that the circle forms randomly and by Central Limit Theoremm, half of the people actually have to walk towards the circle whereas other half are going to be inside ! This can be verified by the average distance walked. Let's see :

# In[ ]:


pubg['walkDistance'].mean()


# In[ ]:


print(pubg['walkDistance'].max())
print("How the hell can someone walk a fu****g 25 Km. Let's see who he actually is !")
pubg[pubg['walkDistance']>25000].iloc[0,:].to_frame()


# Usain Bolt Bolt Bolt !!!!!!

# ## 1.18 Weapons Acquired

# Weapon Acquired : Number of weapons picked up

# In[ ]:


sns.catplot('weaponsAcquired','winPlacePerc',data=pubg, jitter=False)
f=plt.gcf()
f.set_size_inches(20,15)
plt.title('Weapons Acquired vs winPlacePerc')


# There seems to be a positive correlation between weapons acquired and winPlacePerc. Let's see !

# In[ ]:


sns.heatmap(pubg[['weaponsAcquired','winPlacePerc']].corr(),annot=True)


# ## 1.19 Correlation Matrix
# 

# In[ ]:


sns.heatmap(pubg.corr(),annot=True,linewidths=0.5,cmap='RdYlGn')
f = plt.gcf()
f.set_size_inches(25,20)


# Correlation with winPlacePerc :

# 1. Boosts have a higher correlation with winPlacePerc even more than heals.
# 2. KillPlace and winPlacePerc have high negative correlation which is something we saw above !
# 3. DamageDealt,Heals,Kills,KillStreaks,LongestKill,RideDistance,WeaponsAcquired all of these have considerable correlation with winPlacePerc as well.
# 4. WalkDistance seems to have the highest correlation with winPlacePerc which can prove to be useful in a prediction model !

# # Tableau and Analysis

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1548504873335' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;26&#47;266JW6WP2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;266JW6WP2' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;26&#47;266JW6WP2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1548504873335');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1029px';vizElement.style.height='722px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                   \nvizElement.parentNode.insertBefore(scriptElement, vizElement);    \n</script>")


# The above is a tableau created dashboard. It is an interactive dashboard which you can interact with using hover or selecting a particular bubble. Do that to understand better !
# 
# The bigger the bubble is, the more the data it holds.
# 
# What is particularly interesting about this workbook is, when you analyse this workbook, the amount of kills/weapons aqcuired etc. are going to be generally more for winPlacePerc = 1.0 and less for winPlacePerc = 0, giving an indication of positive correlation within the variables except for killPlace in which as we saw above, a negative correlation can be observed !

# # 2. Feature Engineering

# ## 2.1 Health

# The first thing I am going to do is to combine heals and boosts since survival in a game is mostly governed by these two !

# In[ ]:


pubg['health'] = pubg['heals'] + pubg['boosts']


# Let's see how the combined correlation compares to that of individual heals and boosts :

# In[ ]:


sns.heatmap(pubg[['health','heals','boosts','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)


# Correlation of health with winPlacePerc is comaparitively lower than just with boosts. But this helps us with two things. Our model won't overfit with keeping just boosts as the prime feature winPlacePerc is based on and also heals and boosts combined together will make for a more robust prediction ! You can see afterwards if just keeping boosts instead of our new feature health has more impact on our predictions or not !

# ## 2.2 HeadshotSkill

# As I talked about in the above section, a player who is known to take more headshots will generally be a more skilled and a worthy winner indeed. Now headshot kills represent the number of enemies killed by headshots. This generally can be misleading since if I killed only one player that too by headshot, this might have happened out of sheer chance ! Also if I killed 5 people out of which only 1 was by headshot, I may not be skilled enough. So I am going to create a feature HeadshotSkill which would accurately depict my skill level based on data and is formed below :

# In[ ]:


def head(c):
    if(c[0]==0):
        return 0
    elif(c[1]==0):
        return 0
    else:
        return c[0]/c[1]
pubg['headshotSkill'] = pubg[['headshotKills','kills']].apply(head,axis=1)


# In[ ]:


pubg.describe()


# In[ ]:


sns.heatmap(pubg[['headshotKills','headshotSkill','kills','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)


# A correlation though not substantial can be seen !

# ## 2.3 DistanceCovered

# In[ ]:


pubg['distanceCovered'] = pubg['walkDistance']+pubg['rideDistance']+pubg['swimDistance']


# In[ ]:


sns.heatmap(pubg[['distanceCovered','walkDistance','rideDistance','swimDistance','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)


# ## 2.4 ActualKills

# I personally despise team kills a lot and obviously if a match is based on team vs team, having lesser membes is a disadvantage !

# In[ ]:


pubg['actualKills'] = pubg['kills'] - pubg['teamKills']


# In[ ]:


sns.heatmap(pubg[['kills','actualKills','teamKills','winPlacePerc']].corr(),annot=True)
f=plt.gcf()
f.set_size_inches(8,6)


# In[ ]:




