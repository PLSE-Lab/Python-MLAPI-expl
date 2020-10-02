#!/usr/bin/env python
# coding: utf-8

# <h1>NFL 1st and future- Analytics</h1>
# <img src='https://1ycbx02rgnsa1i87hd1i7v1r-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/nfl.png'>
# <h2>About NFL</h2>
# <p>The National Football League is America's most popular sports league, comprised of 32 franchises that compete each year to win the Super Bowl, the world's biggest annual sporting event. Founded in 1920, the NFL developed the model for the successful modern sports league, including national and international distribution, extensive revenue sharing, competitive excellence, and strong franchises across the country.
# 
# The NFL is committed to advancing progress in the diagnosis, prevention and treatment of sports-related injuries. The NFL's ongoing health and safety efforts include support for independent medical research and engineering advancements and a commitment to work to better protect players and make the game safer, including enhancements to medical protocols and improvements to how our game is taught and played.
# 
# As more is learned, the league evaluates and changes rules to evolve the game and try to improve protections for players. Since 2002 alone, the NFL has made 50 rules changes intended to eliminate potentially dangerous tactics and reduce the risk of injuries.
# 
# For more information about the NFL's health and safety efforts, please visit <a href='www.PlaySmartPlaySafe.com'>the NFL's health and safety efforts</a></p>

# <h2>The Challenge</h2>
# <p>In the NFL, 12 stadiums have fields with synthetic turf. Recent investigations of lower limb injuries among football athletes have indicated significantly higher injury rates on synthetic turf compared with natural turf (Mack et al., 2018; Loughran et al., 2019). In conjunction with the epidemiologic investigations, biomechanical studies of football cleat-surface interactions have shown that synthetic turf surfaces do not release cleats as readily as natural turf and may contribute to the incidence of non-contact lower limb injuries (Kent et al., 2015). Given these differences in cleat-turf interactions, it has yet to be determined whether player movement patterns and other measures of player performance differ across playing surfaces and how these may contribute to the incidence of lower limb injury.
# 
# Now, the NFL is challenging Kagglers to help them examine the effects that playing on synthetic turf versus natural turf can have on player movements and the factors that may contribute to lower extremity injuries. NFL player tracking, also known as Next Gen Stats, is the capture of real time location data, speed and acceleration for every player, every play on every inch of the field. As part of this challenge, the NFL has provided full player tracking of on-field position for 250 players over two regular season schedules. One hundred of the athletes in the study data set sustained one or more injuries during the study period that were identified as a non-contact injury of a type that may have turf interaction as a contributing factor to injury. The remaining 150 athletes serve as a representative sample of the larger NFL population that did not sustain a non-contact lower-limb injury during the study period. Details of the surface type and environmental parameters that may influence performance and outcome are also provided.</p>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


injury_rec=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
player_trackdata=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
play=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')


# <h2>EDA of Injury Record Data</h2>
# <p>first We are going to ask ourselves a questions and answer it from the datasets</p>
# <ul>
#     <li>Making a pie chart of the fields to Know which field has the greater percentage</li>
#     <li>I will determine the severity of the injury through the number of missing days in Injury record dataset</li>
#     <li>Making a pie chart of injury type</li>
#     <li>Comparing between the injuries that happened in Synthetic field and natural field</li>
#     <li>Comparing between the injury-types that happened in Synthetic field and natural field</li>
#     <li>let's see the injury type in each body part in synthetic and natural field</li>
# <ul>

# In[ ]:


plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
injury_rec['Surface'].value_counts().plot.pie(autopct='%.1f%%',
                                              shadow=True)
plt.title('Comparing injuries in different fields')
plt.ylabel('')
plt.show()


# In[ ]:


injury_rec['injury_type']=injury_rec[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)
def change_vals(i):
    if i==1:
        return 'Light Injury'
    elif i==2:
        return 'Medium Injury'
    elif i==3:
        return 'Almost Serious Injury'
    else:
        return 'Serious Injury'
injury_rec['injury_type']=injury_rec['injury_type'].apply(change_vals)


# In[ ]:


injury_rec['injury_type'].value_counts(normalize=True).plot.pie(autopct='%.1f%%',
                                                                shadow=True,
                                                               colors=['gold', 'yellowgreen', 'lightcoral', 'lightskyblue'])
plt.title('Injury-Types of NFL players')
plt.ylabel('')
plt.show()


# <p>from the chart above we can see that most of injuries takes 7 days to return to games</p> 

# In[ ]:


sns.countplot(x='Surface',hue='BodyPart',data=injury_rec)
plt.legend(loc='best')
plt.show()


# <p>In Synthetic Surface there are alot of ankle injuries and toes in comparison with Natural Surface, there are no heel injuries in Synthetic, equally knee injuries</p>

# In[ ]:


sns.countplot(x='Surface',hue='injury_type',data=injury_rec)
plt.legend(loc='best')
plt.show()


# <p>we can see that there are alot of injuries happened in synthetic are Serious and almost serious comparing with natural</p>

# <h3> In Synthetic Surface</h3>

# In[ ]:


body_part=injury_rec['BodyPart'].unique().tolist()
fig=plt.figure(figsize=(15,3))
for i in range(len(body_part)-1):
    ax=fig.add_subplot(1,5,i+1)
    c=injury_rec[(injury_rec['BodyPart']==body_part[i])&(injury_rec['Surface']=='Synthetic')]['injury_type'].value_counts(normalize=True)*100
    c_lst=c.tolist()
    ax.bar(c.index,c_lst,width=0.4)
    ax.set_title('{} in Synthetic'.format(body_part[i]))
    ax.set_ylabel('')
    xlabels=[i for i in c.index]
    ax.set_xticklabels(xlabels, rotation=90)
plt.show()


# <h3>In Natural Surface</h3>

# In[ ]:


body_part=injury_rec['BodyPart'].unique().tolist()
fig=plt.figure(figsize=(15,3))
for i in range(len(body_part)):
    ax=fig.add_subplot(1,5,i+1)
    c=injury_rec[(injury_rec['BodyPart']==body_part[i])&(injury_rec['Surface']=='Natural')]['injury_type'].value_counts(normalize=True)*100
    c_lst=c.tolist()
    ax.bar(c.index,c_lst,width=0.4,color='red')
    ax.set_title('{} in Natural'.format(body_part[i]))
    ax.set_ylabel('')
    xlabels=[i for i in c.index]
    ax.set_xticklabels(xlabels, rotation=90)
plt.show()


# <h2>EDA of Play List data</h2>
# <ul>
#     <li>Use the playkey to See the Temperature, Weather, position of players who have been 
#     injured</li>
#     <li>Examine which position more exposed to injuries</li>
#     <li>Examine which position more exposed to injuries</li>
#     <li>Box plot of temperature</li>
#     <li>let's see if the weather is effective </li>
# </ul>

# In[ ]:


injured_full_data=injury_rec.merge(play)
len(injured_full_data)


# <p>77 link between injury record data and playlist data</p>

# In[ ]:


injured_full_data['PlayType'].value_counts().plot.barh(figsize=(15,5))
plt.title('play-type of injured players')
plt.xlabel('frequency')
plt.ylabel('play-type')
plt.show()


# <p>most of injuries happen in Pass and Rush</p>

# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
injured_full_data['Position'].value_counts().plot.barh()
plt.title('Position of injured players')
plt.xlabel('frequency')
plt.ylabel('')
plt.subplot(1,2,2)
play['Position'].value_counts().plot.barh()
plt.title('Position of all players')
plt.xlabel('frequency')
plt.ylabel('')
plt.show()


# <p>there are positions not exist in injured players, (WR,OLB,CB) positions are the most positions exposed to injury</p>  

# In[ ]:


injured_full_data['Temperature'].plot.kde(label='injured players')
play['Temperature'].plot.kde(color='red',label='all players')
plt.axvline(injured_full_data['Temperature'].mean(),c='green',label='inj_mean')
plt.axvline(play['Temperature'].mean(),c='yellow',label='all_mean')
plt.legend(loc='best')
plt.show()


# In[ ]:


injured_full_data['Weather'].value_counts().plot.barh(figsize=(15,10))
plt.title('Weather')
plt.xlabel('frequency')
plt.ylabel('Weather-type')
plt.show()


# <p>most of injuries happened in (Cloudy,Partly Cloudy,Sunny)</p>

# In[ ]:


plt.figure(figsize=(6,16))
plt.subplot(2,1,1)
injured_full_data[injured_full_data['Surface']=='Synthetic']['Weather'].value_counts().plot.barh(figsize=(15,10),color='red')
plt.title('Weather in Synthetic')
plt.ylabel('')
plt.subplot(2,1,2)
injured_full_data[injured_full_data['Surface']=='Natural']['Weather'].value_counts().plot.barh(figsize=(15,10))
plt.title('Weather in Natural')
plt.xlabel('frequency')
plt.ylabel('')
plt.show()


# <h2>EDA of Player Track-data</h2>

# In[ ]:


player_trackdata['s'].plot.hist(bins=20)
plt.axvline(player_trackdata['s'].mean(),color='red',label='mean')
plt.axvline(player_trackdata['s'].median(),color='green',label='median')
plt.axvline(player_trackdata['s'].std(),color='blue',label='std')
plt.legend()
plt.show()


# <h3>Plot the distribution of speed of injured players</h3>

# In[ ]:


inj_players=injury_rec['PlayKey'].tolist()
player_trackdata.query('PlayKey in @inj_players')['s'].plot.hist(bins=20)
plt.title('Distribution of speed of injured players')
plt.xlabel('speed')
plt.ylabel('frequency')
plt.axvline(player_trackdata.query('PlayKey in @inj_players')['s'].mean(),label='mean',color='red')
plt.axvline(player_trackdata.query('PlayKey in @inj_players')['s'].std(),label='std',color='blue')
plt.legend()
plt.show()


# <h3>Plot the distribution of Orientation of injured players</h3>

# In[ ]:


player_trackdata.query('PlayKey in @inj_players')['o'].plot.hist(bins=30)
plt.title('Distribution of Orientation of injured players')
plt.xlabel('Orientation')
plt.ylabel('frequency')
plt.axvline(player_trackdata.query('PlayKey in @inj_players')['o'].mean(),label='mean',color='red')
plt.axvline(player_trackdata.query('PlayKey in @inj_players')['o'].std(),label='std',color='blue')
plt.legend(loc='best')
plt.show()


# <h3>Plot the distribution of Direction of injured players</h3>

# In[ ]:


player_trackdata.query('PlayKey in @inj_players')['dir'].plot.hist(bins=30)
plt.title('Distribution of Direction of injured players')
plt.xlabel('Direction')
plt.ylabel('frequency')
plt.axvline(player_trackdata.query('PlayKey in @inj_players')['dir'].mean(),label='mean',color='red')
plt.axvline(player_trackdata.query('PlayKey in @inj_players')['dir'].std(),label='std',color='blue')
plt.legend(loc='best')
plt.show()


# <h1>More Investigations</h1>

# In[ ]:


inj_synth=injury_rec[injury_rec['Surface']=='Synthetic']['PlayKey'].tolist()
inj_natural=injury_rec[injury_rec['Surface']=='Natural']['PlayKey'].tolist()
player_trackdata.query('PlayKey in @inj_synth')['s'].plot.hist(bins=20,alpha=0.4,label='Synthetic')
player_trackdata.query('PlayKey in @inj_natural')['s'].plot.hist(bins=20,color='red',alpha=0.4,label='natural')
plt.title('injured players in synthetic VS Natural')
plt.xlabel('Speed')
plt.ylabel('frequency')
plt.legend()


# In[ ]:


print('mean of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].mean())
print('mean of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].mean())


# In[ ]:


print('std of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].std())
print('std of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].std())


# In[ ]:


print('skewness of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].skew())
print('skewness of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].skew())


# In[ ]:


print('kurtosis of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].kurtosis())
print('kurtosis of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].kurtosis())


# In[ ]:


player_trackdata.query('PlayKey in @inj_synth')['o'].plot.hist(bins=20,alpha=0.4,label='Synthetic')
player_trackdata.query('PlayKey in @inj_natural')['o'].plot.hist(bins=20,color='red',alpha=0.4,label='natural')
plt.title('injured players in synthetic VS Natural')
plt.xlabel('Orientation')
plt.ylabel('frequency')
plt.legend()
plt.show()


# In[ ]:


print('mean of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].mean())
print('mean of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].mean())


# In[ ]:


print('std of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].std())
print('std of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].std())


# In[ ]:


print('skewness of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].skew())
print('skewness of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].skew())


# In[ ]:


print('kurtosis of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].kurtosis())
print('kurtosis of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].kurtosis())


# In[ ]:


player_trackdata.query('PlayKey in @inj_synth')['dir'].plot.hist(bins=20,alpha=0.4,label='Synthetic')
player_trackdata.query('PlayKey in @inj_natural')['dir'].plot.hist(bins=20,color='red',alpha=0.4,label='natural')
plt.title('injured players in synthetic VS Natural')
plt.xlabel('Direction')
plt.ylabel('frequency')
plt.legend()
plt.show()


# In[ ]:


print('mean of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].mean())
print('mean of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].mean())


# In[ ]:


print('std of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].std())
print('std of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].std())


# In[ ]:


print('skewness of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].skew())
print('skewness of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].skew())


# In[ ]:


print('kurtosis of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].kurtosis())
print('kurtosis of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].kurtosis())

