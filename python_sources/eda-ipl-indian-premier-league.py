#!/usr/bin/env python
# coding: utf-8

# THIS IS IN-DEPTH EXPLORATORY DATA ANALYSIS (EDA) FOR IPL MATCHES TILL DATE FOR BEGINNERS
# THIS EDA USES ** Indian Premier League (Cricket) **  DATASET FROM KAGGLE 
# THE THINGS WE CAME TO KNOW FROM THIS EDA :
#  
#  --->1)NO. OF MATCHES WON BY TEAMS IN SEASON 2017
#  --->2)HIGHEST MAN OF THE MATCHES
#  --->3)NO. OF MATCHES WON BY TEAMS IF THEY HAVE WON THE TOSS
#  --->4)TOSS DECISION BY TEAMS ACROSS THE SEASONS
#  --->5)TEAMS WITH HIGHEST NO. OF WON MATCHES  IN ALL THE SEASON
#  --->6)** PROBABILITY OF WINNING THE MATCH IF YOU HAVE  WON THE TOSS **

# In[ ]:


#IMPORT BASIC LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#load the dataset
matches=pd.read_csv("../input/matches.csv")
matches2017=matches[matches["season"]==2017]


# **THIS VISUALIZATION SHOW THE PLAYER  WHO HAVE WON THE MOST PLAYER OF MATCH (SEASON (2008-2017))**

# In[ ]:


plt.subplots(figsize=(10,10))
matches["player_of_match"].value_counts().head(7).plot.bar()


# **CHRIS GAYLE IS A CLEAR WINNER **

# **TEAM WHICH HAVE WON MOST NUMBERS OF MATCHES

# In[ ]:


plt.subplots(figsize=(10,6))
matches['toss_winner'].value_counts().plot.bar(width=0.8)
plt.show()


# **AND ALL CHEERS GOES TO MI 

# **CHOICE OF TOSS IN DIFFERENT SEASONS

# In[ ]:


plt.subplots()
sns.countplot(x="season",hue="toss_decision",data=matches)
plt.show()


# **THE CHOICE TO SELECT WHETHER TO TAKE BAT OR FIELD DIFFER IN DIFFERENT SEASONS

# **HOW MANY MATCHES  THE TEAMS HAVE WON IF THEY HAD WON THE TOSS AND IF THEY HAVE LOST THE MATCH , THEN TO WHOM THEY HAVE LOST IT(SEASON 2017 ONLY)

# In[ ]:


b=list(matches2017["toss_winner"].unique())
fig,axes1=plt.subplots(figsize=(20,13))
sns.countplot(x="toss_winner",data=matches2017,hue="winner" ,ax=axes1)

axes1.set_xticklabels(b,rotation=90)
axes1.set_xlabel("")


# **NOW THE MOST MAIN PART**
# DO THE **TEAM** WINS IF THEY **WON** THE **TOSS**

# In[ ]:


df=matches[matches['toss_winner']==matches['winner']]
slices=[len(df),(577-len(df))]
labels=['yes','no']
plt.pie(slices,labels=labels,startangle=90,autopct='%1.1f%%',colors=['r','g'])
plt.show()


# HENCE THE CHANCE OF WINNING IS AROUND 50-50 WE CAN SAY, BUT THE CHANCE ARE LITTLE BIT MORE TOWARDS  WINNING TOSS SIDE.
# 
# **HOPE YOU LIKED IT**

# 
