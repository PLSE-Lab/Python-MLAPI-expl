#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis - Player Unknown Battlegrounds(PUBG)

# ![](https://lh3.googleusercontent.com/aA6vbfu63on6cGM7FEOth-yU1pYj_A9OebxFRddp1mw3ZrToKTWdyteJTPwC97vdDqFb7MSZwqCIVZSEtcIiZYuaKhS0AUWf-PXz-MaZvAjJYlhuXpKSPim0YbSoYS7zpQhoUlqs0Fh5i7BxB8ngL1vcL3huB4QOmytQiplewCYl8hu3FqpD4Xxxaj-gvmmGt8bFrZPvljWLDrlbEk-PIcCh9ZZ43Khr__iq-BiOVb3htzrGzGL6vAt9JXvvMkNw218TieawJDYeygM6IKEdsCufzXGGQvNGkCMsb742cyUrke50s4g5P55qw443dV2kEl1CNfXyjZwdZt_KcDG6VjZmMEsjYFdH-O4_Hr3pGW1tS1y9ed1CgWmw8UfskIdjIRuv1Vp9RFrBz3dDFevj9oBBwI48TZQfvoGbfA3A4i4noeSCrxXaYQlGktYyzfEBn7tveew9isUbRBxMF4nRcy-Oy45nD5e1mVbUH3WASG0s1pvrmQ_zi7L0oRC4Q5ct-uU8EA8rdevENeUJZu_SOvUa8We4FeoJPwKXnyAMtl4t3QwgAn_4pd6K_Y0FE5MiiT4hHuHK0hVbaafGJdH4dwaKvqam4zkbNskDyfPRTRSkRU2KC5X389ekfkLXL4NEZ6A0NoChTw6PHXqj0gsCWLKhpyvKly_yu9213sCetFbnTXqxof8T1ofw-4bwZH5eCjuMp0dnMHPF43KnEg=w1239-h608-no)

# In[ ]:


import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from plotly.offline import init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')


# Lets see how the data looks

# In[ ]:


train.sample(5)


# Data type,Memory usage and total number of records

# In[ ]:


train.info()


# Co-Relation Matrix 
# 

# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,center=0)
plt.show()


# couple of observations
# 1. --Walk distance is highly co-related among all other features.
# 1. --Kill place is negatively co-related with target
# 1. --Boosts and weapons acquired have the 2nd highest co-relation.
# 
# Lets break it down into combat, survival, distance covered and check the co-relations to proceeed further.

# **[Combat Features](#Combat)**

# ![](https://lh3.googleusercontent.com/d4h89nYI18GD2Xd0braJVh38eNrvRu0wXvj1gv24P5Dfgw2JWn1WR_4hp_SdKz6MpdcUmH_JmK0UI955MqSfAOuNfz8gF2XrGJaWBdtVAJIVVs9L5UeY8IhWIciFKN953lyEHLsiMM6Jltom_nrOZfXy_w0d3KeFa08N8UiIgdxmFzFVV7KCYLouxMya9QYkdYaZEL0piQx0VuW5VfwDr3Pyr2g-VFu3lNoxR9fc9lLBN3LTbblXOYKaRcdHtBcFW0HN7IoBU9aOFVVBa2J8372-gFS1gvLmT9wfGCy_I2Bo89AQKgv_v3VmCmFXHVJggxDAFfXup1fsJcTDJ3lVrQsW-SdaEPmHK8ILqGXF9TLiTjpaVUPP9TfZEvyCBRt63FblA-6rqOtzi09uUM8He4N9tp-HnaJcrKYdmW3hSXM_CyPPjzjgHcXytKQfubLKAMmAzO68lDjcbN8K_fXTQ9TgJwT-0Pbk7e1thYPG9kYuNfcEJxzSQ4JhNxIJdftzTLb3dqq9x6wb6EFxDFkMjTeSZrK0olDYsG0gGon1ITgRt8dGVS0cZOfvnKtSuaRmuoWFVhDdaQdzcK69Y0GnMhudJhSMEg4_NFdP-hENGZH8na6Q7vYLn5aHGVuHdE5A_a8_0EmOxcJsmwaFG-bRYxaB=w1081-h608-no)

# In[ ]:


corr_with_target = train.drop(['Id', 'matchId', 'groupId', 'winPlacePerc'], 1).corrwith(train['winPlacePerc'])
corr_with_target

combat_features=['assists',
'damageDealt',
'DBNOs' ,
'headshotKills',
'kills',
'killStreaks',
'vehicleDestroys',
'weaponsAcquired',
'longestKill']
combat_with_target = train[combat_features].corrwith(train['winPlacePerc'])
trace_combat = go.Scatter(
    x = combat_with_target.index,
    y = combat_with_target.values,
    mode = 'markers+lines',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))
data_combat = [trace_combat]
layout = {'title': 'Correlation between winPlacePerc and combat attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}


fig = go.Figure(data=data_combat, layout=layout)
iplot(fig, show_link=False)



# Weapons Acquired has the highest co-relation among all the other combat  features.

# **Weapons Acquired** - Number of weapons acquired in a match (this also included weapons that you have used and dropped)

# In[ ]:


print("**** Weapons Acquired Stats ****")
print("Average person acquires {:f} weapons per match".format(train['weaponsAcquired'].mean()))
print("Maximum weapons ever recorded in a match is  :{:f}".format(train['weaponsAcquired'].max()))


# In[ ]:


plt.figure(figsize=(15,10))
plt.title("weaponsAcquired",fontsize=15)
sns.distplot(train['weaponsAcquired'])


# what about people with no weapons, lets see avegage damagedealt by players with no weapons

# In[ ]:


data_fist_attack=train.copy()
plt.figure(figsize=(15,10))
data_fist_attack=data_fist_attack[data_fist_attack.weaponsAcquired==0]
data_fist_attack_grouped=data_fist_attack.groupby('kills')['damageDealt'].mean()
plt.title("Damage and kills by players without weapons")
data_fist_attack_grouped.plot()
plt.show()


# we certainly have some Muhammad Ali fans out here
# 

# **kill Streaks** - Number of enemy killed in a short time frame.

# In[ ]:


print("**** Kill Streaks Stats ****")

print("Average Kill streaks per match is {:f}.".format(train['killStreaks'].mean()))
print("highest kill streaks ever recorded in a match is  {:f}".format(train['killStreaks'].max()))
sns.distplot(train['killStreaks'])


# In[ ]:


sns.jointplot(x="winPlacePerc", y="killStreaks",  data=train,  ratio=3, color="blue")
plt.show()


# kills in a certain timeframe, not sure how long that period is.
# Couple of cases that comes to my mind:-
# - After every kill, you spend some time healing before kill the next one.
# - You run up on a 4 man squad, knock three of them, then kill the fourth, your kill streak would be 4, as the other three would insta-die, in the certain time frame as referenced by 
# - Gamers like shroud,ninja,docdisrespect wipes out an entire squad (one after another) in a span of few secs.
# 

# **Longest Kill Shots** - From what distance the,the player have killed an enemy
# 
# **Headshot Kills** - number of enemies killed via headshots

# In[ ]:


plt.figure(figsize=(15,10))
plt.title("longestKill",fontsize=15)
sns.distplot(train['longestKill'])
print("**** Longest Kill Stats ****")
print("longest kill ever recorded was :{:f} meters".format(train['longestKill'].max()))
print("Average shot is about {:f} meters".format(train['longestKill'].mean()))


# ![](https://lh3.googleusercontent.com/BBUOavltUo9SBR9TamY_j2bUG94KKTK8ej_BRhyfJPEJyEUJLepVPBOQzpZtG6-QAQGmoljyKyc-K_fuUdEZ-JHUGXzVDozVqd2EuglowasmPVnbjzM6qCcThU_Yp1AVebIdfO5GFMqGvxmhhuRnGAewnpWqHJKssrpak8XXaKLImBT1tgtlaEit81owCype1mMskdHFEroZ4gbngxwJz3Y-_LtmbITMtnQs-TJQc_mkqHsjt59f1DIZxkkpOwClDWhsOzdiuy53JUSGZO2fTd2tk7w03TPr_kN-O7n8-ql7ov95sfvZ9sOrkn56PeM7kQr3PWLMEmGItW8rxNxj9adaDCmO6HiH-ySv4fBGofXEG3VdVHzRGol9gyNyP-AKJJ8xW4ta9DKx7xJYTkJmHkI1GLmjbYE2oluZKUe8V1IKRiwYJNMSJlq0bhBl3JowgDPLkJEYUXaTcFTYArMNfOprwX3QG8ANbxCHpj9rkuDi26Ajyzvuvf9fy9lTMsTenKsBvgrwVK5MYbBv1JvDsrreGPbgTR-0q24zn7OKcYfAXTH-84SQkaWoXC6bsglrZTcYA_S6nlmLGWRyhhkAFwRQ6C2XfDgp5Rt8IxtKKelBxSeNGZPcmYFiGy3XdixzLhVkcuvBtis5iM0d6u2SZfBK3lu9WGo4IeJPpv5iYmd5dWKSbjFACGxlI8XrNR-IDmjF98FXsojqh6Ii-g=w1239-h608-no)

# In[ ]:


train_headshotkills_temp=train.copy()
train_headshotkills_temp.loc[train_headshotkills_temp['headshotKills']>4]='4+'
sns.countplot(x=train_headshotkills_temp['headshotKills'].astype('str').sort_values())


# In[ ]:


print("****HeadShot Kills Stats****")
print("Most recorded headshots ever in a match is :{:f}".format(train['headshotKills'].max()))
print("83% of people have {} headshots".format(train['headshotKills'].quantile(0.83)))
print("Avergare Headshots:{:f}".format(train['headshotKills'].mean()))


# Lets see how the winPlacePerc is distributed since 83% player have 0 headshot kills

# In[ ]:


data_headshot0 = train.copy()
data_headshot0 = data_headshot0[data_headshot0['headshotKills']==0]
sns.jointplot(x="winPlacePerc", y="kills",  data=data_headshot0,  ratio=3, color="blue")
plt.show()


# winPerc distribution from 0.4 till 1.0 is quite close, this shows that number of headshots doesnot really matter. Lets dive deep into damage and kills
# 

# **Kills** - Number of enemy players killed in a match  
# **Damage Dealt** - Amount of damage caused

# In[ ]:


train_temp=train.copy()
train_temp.loc[train_temp['kills']>7]='7+'
train_temp.groupby(['kills']).groups.keys()
plt.figure(figsize=(15,10))
sns.countplot(y=train_temp['kills'].astype('str').sort_values())


# In[ ]:


print("****Kills Stats*****")

print("Most num of kills  ever recorded in a match is :{:f}".format(train['kills'].max()))
print("90% of people have {} kills".format(train['kills'].quantile(0.99)))
print("Strangly last 10% of the data have a drastic increase in kill count, which is  {}".format(train['kills'].quantile(0.99)))
print("Avergare kills:{:f}".format(train['kills'].mean()))


# Creating Kill Categories in-order to have a better visibility

# In[ ]:


kills = train.copy()

kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])
sns.barplot(x="winPlacePerc", y="killsCategories",
              data=kills)


# Can players with 0 or just 1 kill contribute or make a difference

# In[ ]:


train_temp2=train.copy()
var=train_temp2[train_temp2.kills<2]

sns.jointplot(x=var['winPlacePerc'], y=var['kills'],  data=var,  ratio=3, color="blue")
plt.show()


# Apparently players with less then 2 kills are also winners, what are the odds
# 
# lets see how the other features contribute

# In[ ]:


train_temp3=train.copy()
var3=train_temp3[train_temp3.kills<2]
var4 = var3.corr()['winPlacePerc'].nlargest(5)
var4=var4.drop(labels=['winPlacePerc' ])

trace3 = go.Scatter(
    x = var4.index,
    y = var4.values,
    mode = 'markers+lines',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))

data2 = [trace3]

layout = {'title': '0 - 1 kills Correlation with Target(top 5)',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

fig = go.Figure(data=data2, layout=layout)
iplot(fig, show_link=False)


# walk distance,weapons acquired and boosts are still the key features here

# Kills and Damage vs Target

# In[ ]:



damage = train.copy()
damage_bins=[-1,0,200,400,600,1000,3000,6616]
damage_groups=["0","1-200","200-400","400-600","600-1000","1000-3000","3000-6616"]
damage['damageCategories'] = pd.cut(damage['damageDealt'],damage_bins,labels=damage_groups)
plt.figure(figsize=(15,10))
kill_bins=[-1, 0, 2, 5, 10, 60]
kill_groups=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills']
damage['killsCategories'] = pd.cut(damage['kills'],kill_bins, labels=kill_groups)      
sns.barplot(x="damageCategories", y="winPlacePerc",
      data=damage,hue='killsCategories')


# winners have 10+ kills and  damage caused by them is in between 200 - 400

# In[ ]:


print("Max damage dealt ever :{:f}".format(train['damageDealt'].max()))
print("50% of people have dealt: {} ".format(train['damageDealt'].quantile(0.50)))
print("min damage dealt :  {}".format(train['kills'].min()))
print("Avergare damage:{:f}".format(train['damageDealt'].mean()))


# Lets see how  other combat features are co-related to damage

# In[ ]:


train_temp4=train.copy()
kills_wins = ['kills',
              'DBNOs',              
'headshotKills',
'heals',     
'killPlace',       
'killPoints',        
'kills',        
'killStreaks',
'longestKill']
kills_wins_with_damage = train[kills_wins].corrwith(train['damageDealt'])

trace6 = go.Bar(
    x=kills_wins_with_damage.index,
    y=kills_wins_with_damage.values,
     marker=dict(
        color='rgb(49,130,189)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace6]

layout = {'title': '0 - 1 kills Correlation with Target',
          'yaxis' : {'title' : 'Combat co-relation with Damage'},
          'xaxis' : {'tickangle' : 45}}

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# kills and damage are highl co-related.
# 
# Lets comapare how Kills and Damage are co-related with Target

# In[ ]:


train_temp4=train.copy()
kills_wins = ['kills','damageDealt']
kills_wins_with_damage = train[kills_wins].corrwith(train['winPlacePerc'])

trace6 = go.Bar(
    x=kills_wins_with_damage.index,
    y=kills_wins_with_damage.values,
     marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace6]

layout = {'title': 'kills vs Damage Correlation with Target',
          'yaxis' : {'title' : 'Damage Dealth Co-relations'},
          'xaxis' : {'tickangle' : 45}}

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


#   ** [Survival Features](#Survival)**
# 

# **Boost**  - Number of Boost items used in a match.
# 
# **Heals**  - Number of Heal items used in a match.
# 

# ![](https://lh3.googleusercontent.com/XYar0zRLl0jL_YlrvwdmH0HurNHJSvMXm7SOyUph7UCxTOKylpuv9lCEegn_ReYFyQWlLl0oPUsRnqLEt4EyHR1afXji872eVvNn055zdAg-gUN9d6QheS9qQ9iA8ZGRRSBJ4S_FdFy9NcjOjzrbtcvGFlcnKr2Qr2mkWXFChOVU8vkSe-ifFvJEau0KZ-JhEOftNxvzNvctZJSRAu3G8uKyVOmgp2AiwM0TbZy2xA49mMzqSmqQlpPg4Sq82or2ELWiu7rMTqy91l4HvE6Q_LR3rGlPdm6FGybtk4wiQekrwq_cWCa16JqaONb2kjy6FN1HrnH7jCfaw9K5flucJMcdNxiysJ13s9QEg-ylg5JvKwaEL502Hv_cNigcGEcUiOjd_5rkcEH4wOf-VUyeh_EYcLCYi9-biNWaf799GPp9G2P1MJT9vEa-WrC49Nvivo_6aWGY2yymCw9Cbg_qYKbN8NtW4MljbI3GNmu7BsPAmealSRhb2MiDwvmWVlby2k0jFLjj2Z_JXytQpqa4wMDfq9PL7Fh4V9974slcGO-eDz9WflSDX2bLOhrdv7fPApPTffwOMrU5yhFDJoKchpdrgdvHs6_oLa2GlPEkz6wK9R_0W5VuAg_nqtuNcwA9fKpdaMZWOkD7dRBxW6OfExRdjMTYjqtaiQfs8gU0YmFwf7_5ofz-bXLHvr5BAEMeiM2uK0_lfgAURwDKNg=w1239-h608-no)

# In[ ]:



survival_features=[
'heals',
'boosts',]

survival_with_target = train[survival_features].corrwith(train['winPlacePerc'])
    
trace = go.Scatter(
    x = survival_with_target.index,
    y = survival_with_target.values,
    mode = 'markers+lines',
    marker = dict(
        size = 20,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))
data = [trace]
layout = {'title': 'Correlation between winPlacePerc and survival attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

# Display it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# In[ ]:


print("**** Boost and Heal Stats ****")
print("The average person uses {:.1f} heal items".format(train['heals'].mean()))
print("99% of people use {} or less,".format(train['heals'].quantile(0.99)) )
print("Maximum heal items ever used in a match {:.1f} heal items".format(train['heals'].max()))

print("The average person uses {:.1f} boost items".format(train['boosts'].mean()))
print("99% of people use {} or less,".format(train['boosts'].quantile(0.99)) )
print("Maximum heal items ever used in a match {:.1f} heal items".format(train['boosts'].max()))


# Both are positively corelated with WinPlacePerc but boost is a bit high

# **[Distance Covered Features](#Distance)**
#   

# ![](https://lh3.googleusercontent.com/wp4NJbtEcT97Js7vwZu0GBvDVMD17WiXzilU-krE0O8UYsJkE-YPX3scTEX4bOhLfTbdlQfSE29aCVzN0jlR18OWoqXFrTXWCunSXEYeki2LqUa_ktfK3GkgKZayQJuGPt7PtsdGseIa4gfq4StIF7V9uK7qwdF_3LCzRK4VjxRdTKiyX5K1D8OwVYx2myasgSJ8exbtYMOr0InIDUPzunvH67CsJQXhW_WfgL5vEnHDaITvk5M-loX8mLNxTNz_mfu2Dk685GhWuffkoe9QC_IlCYQidl4n0S78dn312xl2pyCm3kKL7TSDWnjbBqjqV7TLlQGpx22RBQxAGR8CqUog5XRhCGtDGS03MrLEwDGRSpzPWzOvAVT2vnWYOYIIBLbKmhtWSsx0xLxkyIJaTsIhfBThuecrC8JkdDr5QKGevwy-QKG3Rm0h168wAg1H02fBhkXS16PHHZwZSxTS1dgefOvnltB54DttKvmBlgj25DgoSHYrcl3Xfbdvs302_wcie4M9pkCPlYADGjkKztfMfQz22HLy-P8aguTpZtUrrzN7tz5cgKu7f7KLAffVpZSzAz6ntqtk3NHHy1bPMcHrNA9QAh_Yu6Y6FbCLIVRxx2rZDlvcKwpBH2jzUEj-SFhn1AtbRRNJoSnKKPC0iDNs=w1115-h608-no)

# ![](https://lh3.googleusercontent.com/zBCKXXknEGqFPbRJ9mnQfELBvFFaQOrSTPWN1liFno_OfgBgkLxI4ItLBYsdcpHoGRD_K3Q3qL5n7blH94WW1eztNtP-LvcEEAx5TgMBo1ExaaKEjVmhRSu_HJFE-NTUperD_ThiK1BsfwMmXb27LK7vdZJAU8qkmN5ozv83Jl3k30e8nWnCUJpOJaNuDQZFbauhCVxAc9Ea0PkD5BzaipLsT2p9gGTlFynYUW6GjJXrNbdqpMzh8XV8QsjfLteB9m9grR9NLT72K_FLY8Kvll1GAYQ8RlkDanGKCtUszS6ZnHgBbRL3I7xSXzkOFbNYp-MhNBwPtMhSyOgel0a2Klh4pQDzc3l88G4gpAfLMrPIGgwr5civgc8KFAeHJgFF617BApOJMb_ACOV26mWQ-h2qkAH7DpVryPt6e61TvntfiwgK1hX6LICcR-ut1ifFxCL11HfHhvQYVO55zj7yNHAiz5tBf8yVGmRc-fpCGqujQXj4ZSqWQT9Z-ZDXXqHk2MdJXpsmZshdcuMzTjI6ybccJ1aQqB7Hz_b27XQyjzl8-8xR5iC1nOWKlcv4gquGqMBdgnAQ3xR6ZXj_pPzBxZOjnBBGLqIUOU1I1nMfkRpX3jqu9CP4_tMne2llXYEqlUqi79aC8siY9QK9foxylvnq=w1120-h608-no)

# ![](https://lh3.googleusercontent.com/xRS__DjcEbRYOMrSIdt10vDEvrsUV2B1FPbrL5wIrGQJJ1hHU5aS5ogEm-s7zA-mMOsrNEB659FFTo1sYEyhj04ayc06UFDd7J1HD5tsbqkNjlyZZYKx13S843yRiKcZ40BQucI1uy1he7pynepxSm_RuSS_-LRFoTv2vw6QJNxAohCP7AEZf0YZzFdgsjgdkLlD28_QxQGCXv3wcZKsw5oiXfx5yJ5gg5vVo4E_N0PzKuxcpsiXdUjJrirCzecnI-IGTHB1599WK0fGH0IDuG-4vKmpqv1Nxt8DsqKK_cYCyz0vxSZOuiX78m10QLU_LmH47jLhfTXBe57mWF086NE2cpjWffV9Slpmq-ONS6wLJN3MPb7Xq4uXIVSgEIqaMBltqI7vB75rmO0PbTqcFzWxRpvKX61qSUdIUigveFNyjCX4AikMdJ3itfoYaTHEfWQ--p69Kz2InunHaSkrYoFLrgRKTY-r5iTJHm7EP0vAB1-E_G8YMWIxOYsbrj7xb8YKAmySS0Y5u_6wEZ3EiqL19AjFG3V7yrOt3A1nLA517F7HmlxnClfS4t1TSskA_tulWa_ION1htxIhBFaeYOg1cCGxWheJc9qtnZ6IK2lpr3B_wpwRFOZ9NaiXLXfBh4HWiGImsB8BGPavv9A0KDns=w1239-h608-no)

# In[ ]:


print("**** WalkDistance Stats ****")
print("The average person runs/walks {:.1f} meters".format(train['walkDistance'].mean()))
print("Maximum distance covered on foot is {:.1f} meters".format(train['walkDistance'].max()))

print("**** Swimming Stats ****")
print("The average person swims {:.1f} meters".format(train['swimDistance'].mean()))
print("Maximum swim distance covered in a match is  {:.1f} meters ".format(train['swimDistance'].max()))

print("**** Ride Distance Stats ****")
print("An average person covers : {:.2f} meters on a vehicle".format(train['rideDistance'].mean()))
print("Max distance covered on vehicle in a game is  : {} meters".format(train['rideDistance'].max()))


# In[ ]:


distance_covered_features=['rideDistance','swimDistance','walkDistance']

distance_covered_with_target = train[distance_covered_features].corrwith(train['winPlacePerc'])
    
trace = go.Scatter(
    x = distance_covered_with_target.index,
    y = distance_covered_with_target.values,
    mode = 'markers+lines',
    marker = dict(
        size = 20,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2
        )))
data = [trace]
layout = {'title': 'Correlation between winPlacePerc and distance covered attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

# Display it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# Co-Relation between walking and target is highest, keep running fellas

# In[ ]:


plt.figure(figsize=(15,10))
df1= pd.pivot_table(train, values=['walkDistance','swimDistance','rideDistance'],index='winPlacePerc')
g = sns.pairplot(df1)


# Apparently most of the players prefer riding and running over swimming

# ** Solo / Duo / Squad  games**

# In[ ]:


data_teams=train.copy()
solos=data_teams[data_teams['numGroups']>50]
duos = data_teams[(data_teams['numGroups']>25) & (data_teams['numGroups']<=50)]
squads = data_teams[data_teams['numGroups']<=25]
print("Out of total {} games,there are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(len(data_teams),len(solos), 100*len(solos)/len(data_teams), len(duos), 100*len(duos)/len(data_teams), len(squads), 100*len(squads)/len(data_teams),))


# In[ ]:


sns.lmplot(x='kills', y='winPlacePerc', data=solos,fit_reg=False)
ax = plt.gca()
ax.set_title("Solo Games ")

sns.lmplot(x='kills', y='winPlacePerc', data=duos,fit_reg=False)
ax = plt.gca()
ax.set_title("Duo Games")

sns.lmplot(x='kills', y='winPlacePerc', data=squads,fit_reg=False)
ax = plt.gca()
ax.set_title("Squad Games ")


# This clearly show kills do not contribute much in squad game play's, as it is more of a team gameplay. 

# **This was my first pubic kernal (as a beginner), feel free to advice and comment. 
# Feature Engineering and ML is in-progress**
# 
# **Cheers!!**
# 
