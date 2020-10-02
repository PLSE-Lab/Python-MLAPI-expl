#!/usr/bin/env python
# coding: utf-8

# ******OLYMPIC ANALYSIS******

# In[ ]:


from IPython.display import Image
Image(filename='../input/olypic-symbol/download.jpg')


# The modern Olympic Games or Olympics (French: Jeux olympiques[1][2]) are leading international sporting events featuring summer and winter sports competitions in which thousands of athletes from around the world participate in a variety of competitions. The Olympic Games are considered the world's foremost sports competition with more than 200 nations participating.[3] The Olympic Games are held every four years, with the Summer and Winter Games alternating by occurring every four years but two years apart.
# 
# Their creation was inspired by the ancient Olympic Games, which were held in Olympia, Greece, from the 8th century BC to the 4th century AD. Baron Pierre de Coubertin founded the International Olympic Committee (IOC) in 1894, leading to the first modern Games in Athens in 1896. The IOC is the governing body of the Olympic Movement, with the Olympic Charter defining its structure and authority.
# 
# The evolution of the Olympic Movement during the 20th and 21st centuries has resulted in several changes to the Olympic Games. Some of these adjustments include the creation of the Winter Olympic Games for snow and ice sports, the Paralympic Games for athletes with a disability, the Youth Olympic Games for athletes aged 14 to 18, the five Continental games (Pan American, African, Asian, European, and Pacific), and the World Games for sports that are not contested in the Olympic Games. The Deaflympics and Special Olympics are also endorsed by the IOC. The IOC has had to adapt to a variety of economic, political, and technological advancements. As a result, the Olympics has shifted away from pure amateurism, as envisioned by Coubertin, to allowing participation of professional athletes. The growing importance of mass media created the issue of corporate sponsorship and commercialisation of the Games. World wars led to the cancellation of the 1916, 1940, and 1944 Games. Large boycotts during the Cold War limited participation in the 1980 and 1984 Games. The latter, however, attracted 140 National Olympic Committees, which was a record at the time.[4]
# 
# The Olympic Movement consists of international sports federations (IFs), National Olympic Committees (NOCs), and organising committees for each specific Olympic Games. As the decision-making body, the IOC is responsible for choosing the host city for each Games, and organises and funds the Games according to the Olympic Charter. The IOC also determines the Olympic programme, consisting of the sports to be contested at the Games. There are several Olympic rituals and symbols, such as the Olympic flag and torch, as well as the opening and closing ceremonies. Over 13,000 athletes compete at the Summer and Winter Olympic Games in 33 different sports and nearly 400 events. The first, second, and third-place finishers in each event receive Olympic medals: gold, silver, and bronze, respectively.
# 
# The Games have grown so much that nearly every nation is now represented. This growth has created numerous challenges and controversies, including boycotts, doping, bribery, and a terrorist attack in 1972. Every two years the Olympics and its media exposure provide unknown athletes with the chance to attain national and sometimes international fame. The Games also constitute an opportunity for the host city and country to showcase themselves to the world.
# 
# 
# 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import  seaborn  as   sns


# In[ ]:


df = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')


# In[ ]:


df.head()


# ********* HIEGHT WEIGHT AND AGE OF ATHLETES******

# In[ ]:


df[['Age','Height','Weight']].describe()


# In[ ]:


sns.barplot(x='Sex',y='Age',data=df)


# In[ ]:


sns.countplot(x='Sex',data=df)
plt.title('Male VS Female participation in Olympic',size=10,color='red')


# In[ ]:


g = sns.FacetGrid(data=df,col='Sex')
g.map(plt.hist,'Age')


# ****CORELATION BETWEEN HIEGHT AND WEIGHT****

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sns.kdeplot(df['Weight'].dropna(),color='r',label='Weight')
sns.kdeplot(df['Height'].dropna(),color='b',label='Height')


# ****AGE OF ATHLETES WON MEDALS IN DIFFERENT SEASON****

# In[ ]:


g = sns.FacetGrid(df, col="Season",row ="Medal",hue='Sex')

g = g.map(plt.scatter, "Year", "Age").add_legend()


# In[ ]:


WOLRD =df.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

B =WOLRD.plot.bar(figsize=(15,4))
B.set_xticklabels(labels=df['Games'],rotation=90)

ax.set_xlabel('Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Athletes in each type of  Olympic game', size=15, color="BLUE")


# ****NO OF ATHELETES PARTICIPATED IN SUMMER OLYMPICS

# In[ ]:


Summer=df[df.Season.notnull()]
Summer_olympics=Summer[Summer.Season=='Summer']
Summer_olympics.head()
ASummer =Summer_olympics.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

C =ASummer.plot.bar(figsize=(15,4))
B.set_xticklabels(labels=Summer_olympics['Games'],rotation=90)

ax.set_xlabel('Summer Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Athletes in Summer  Olympic game', size=15, color="BLUE")


# In[ ]:


# Top 3 Countries  in Olympics All time
plt.subplot(3,1,1)
Gold_Medal  = df[df.Medal ==  "Gold"].Team.value_counts().head(3)
Gold_Medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Gold Medals")
plt.subplot(3,1,2)
silver_medal = df[df.Medal == "Silver"].Team.value_counts().head(3)
silver_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Silver Medals")
plt.subplot(3,1,3)
bronze_medal = df[df.Medal == "Bronze"].Team.value_counts().head(3)
bronze_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Bronze Medals")


# # TOP  3 PERFORMERS IN SUMMER OLYMPICS

# In[ ]:


plt.subplot(3,1,1)
Gold_Medal  = Summer_olympics[Summer_olympics.Medal ==  "Gold"].Team.value_counts().head(3)
Gold_Medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Gold Medals")
plt.subplot(3,1,2)
silver_medal = Summer_olympics[Summer_olympics.Medal == "Silver"].Team.value_counts().head(3)
silver_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Silver Medals")
plt.subplot(3,1,3)
bronze_medal = Summer_olympics[Summer_olympics.Medal == "Bronze"].Team.value_counts().head(3)
bronze_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Bronze Medals")


# ****USA PERFORMANCE IN SUMMER OLYMPICS

# In[ ]:


USA_Summer=Summer_olympics[Summer_olympics.Medal.notnull()]
USA_Summer_Olympics=USA_Summer[USA_Summer.Team=='United States']
USA_Summer_Olympics.head()
USASummer =USA_Summer_Olympics.groupby('Games').count()['Medal']
fig, ax = plt.subplots(figsize=(15,5))

D=USASummer.plot.bar(figsize=(15,4))
D.set_xticklabels(labels=USA_Summer_Olympics['Games'],rotation=90)

ax.set_xlabel('Summer Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Medals won', size=10, color="GREEN")
ax.set_title(' Performance of USA  in Summer  Olympic game', size=15, color="BLUE")


# 

# **Events in which USA Prefomance is best in Summer olympics**

# In[ ]:


gold2 = USA_Summer_Olympics[(USA_Summer_Olympics.Medal == 'Gold')]

gold2.Event.value_counts().reset_index(name='Medal').head(5)


# > ****MEdals won By USA in Summer Olympics

# In[ ]:


USA_Summer2=Summer_olympics[Summer_olympics.Team.notnull()]
USA_Summer_ololympics2=USA_Summer2[USA_Summer2.Team=='United States']
print('The youngest age athlete  of  USA in Summer Olympics is:' ,USA_Summer_ololympics2.Age.min())
print('The average age of athletes of USA in Summer Olympics is:',USA_Summer_ololympics2.Age.mean())
print('The oldest  Age of athlete of USA in Summer Olympics is:',USA_Summer_ololympics2.Age.max())


# AVG AGE OF ATHELETE WON MEDAL FOR USA IN SUMMER OLYMPICS

# In[ ]:


Bsl=USA_Summer_Olympics.pivot_table(values='Age',index='Medal',columns='Year')
f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(Bsl,annot=True, linewidths=0.05,cmap="coolwarm")
ax.set_xlabel('Summer Game Year', size=14, color="Purple")
ax.set_ylabel('Medal', size=14, color="purple")
ax.set_title(' Avg Age of USA  Athelete won Medal in Summer  Olympic games', size=18, color="Purple")


# NO OF ATHELETE OF USA  PARTICIPATED IN  SUMMER OLYMPICS

# In[ ]:


USASummer2 =USA_Summer_ololympics2.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

D=USASummer2.plot.bar(figsize=(15,4))
D.set_xticklabels(labels=USA_Summer_ololympics2['Games'],rotation=90)

ax.set_xlabel('Summer Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Participation of USA  in Summer  Olympic game', size=15, color="BLUE")


# ATHLETES PARTICIPATION IN WINTER OLYMPICS 

# In[ ]:


Winter=df[df.Season.notnull()]
Winter_olympics=Winter[Winter.Season=='Winter']
Winter_olympics.head()
AWinter =Winter_olympics.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

C =AWinter.plot.bar(figsize=(15,4))
B.set_xticklabels(labels=Winter_olympics['Games'],rotation=90)

ax.set_xlabel('Winter Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Athletes in Winter  Olympic game', size=15, color="BLUE")


# In[ ]:


print('The youngest Age of athlete  of In winter Olympic  is:' ,Winter_olympics.Age.min())
print('The average age of athletes of Winter  Olympic id:',Winter_olympics.Age.mean())
print('The Age of  oldest athlete of Winter Olypic is',Winter_olympics.Age.max())


# TOP BEST 5 Performer In Winter Olympics 

# In[ ]:


plt.subplot(3,1,1)
plt.title('Best Performer in Winter Olympics',size=18, color="Purple")
Gold_Medal  = Winter_olympics[Winter_olympics.Medal ==  "Gold"].Team.value_counts().head(5)
Gold_Medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Gold Medals")
plt.subplot(3,1,2)
silver_medal = Winter_olympics[Winter_olympics.Medal == "Silver"].Team.value_counts().head(5)
silver_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Silver Medals")
plt.subplot(3,1,3)
bronze_medal = Winter_olympics[Winter_olympics.Medal == "Bronze"].Team.value_counts().head(5)
bronze_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Bronze Medals")


# Performance of USA  in Winter Olympic gam

# In[ ]:



USA_Winter=Winter_olympics[Winter_olympics.Medal.notnull()]
USA_Winter_Olympics=USA_Winter[USA_Winter.Team=='United States']
USA_Winter_Olympics.head()
USAWinter =USA_Winter_Olympics.groupby('Games').count()['Medal']
fig, ax = plt.subplots(figsize=(15,5))

E=USAWinter.plot.bar(figsize=(15,4))
E.set_xticklabels(labels=USA_Winter_Olympics['Games'],rotation=90)

ax.set_xlabel('Winter Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Medal won', size=10, color="GREEN")
ax.set_title(' Performance of USA  in Winter Olympic game', size=15, color="BLUE")


# Medal Distrubution of USA IN WINTER OLYPMPICS

# In[ ]:


USA_Winter_Olympics.head()
sns.countplot(x='Medal',data=USA_Winter_Olympics)


# In[ ]:


USA_Winter2=Winter_olympics[Winter_olympics.Team.notnull()]
USA_Winter_Olympics2=USA_Winter2[USA_Winter2.Team=='United States']
print('The youngest age of athlete  of  USA in Winter Olympics is:' ,USA_Winter_Olympics2.Age.min())
print('The average age of athletes of USA Winter Olympics is:',USA_Winter_Olympics2.Age.mean())
print('The oldest Age of athlete of USA Winter Olympics is :',USA_Winter_Olympics2.Age.max())


#  Avg Age of USA  Athelete won Medal in Winter  Olympic games

# In[ ]:


asl=USA_Winter_Olympics.pivot_table(values='Age',index='Medal',columns='Year')


# In[ ]:


f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(asl,annot=True, linewidths=0.05,cmap="coolwarm")
ax.set_xlabel('Winter Game Year', size=14, color="Purple")
ax.set_ylabel('Medal', size=14, color="purple")
ax.set_title(' Avg Age of USA  Athelete won Medal in Winter  Olympic games', size=18, color="Purple")


# **Events in which USA Prefomance is best in winter olympics**

# In[ ]:


gold1 = USA_Winter_Olympics2[(USA_Winter_Olympics2.Medal == 'Gold')]

gold1.Event.value_counts().reset_index(name='Medal').head(5)


# In[ ]:




