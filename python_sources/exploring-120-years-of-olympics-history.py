#!/usr/bin/env python
# coding: utf-8

# # Data exploration about the Olympic Games
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/Athens_1896_report_cover.jpg">
# Cover of the official report for the 1896 Summer Olympics
# 
# **Content**
# 
# The data have been scraped from www.sports-reference.com in May 2018.
# 
# The file athlete_events.csv contains 271116 rows and 15 columns; Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events). 
# The columns are the following:
# 
# |Column|Description|
# |------|-----------|
# |ID|Unique number for each athlete|
# |Name|Athlete's name|
# |Sex|M or F|
# |Age|Integer|
# |Height|In centimeters|
# |Weight|In kilograms|
# |Team|Team name|
# |NOC|National Olympic Committee 3-letter code|
# |Games|Year and season|
# |Year|Integer|
# |Season|Summer or Winter|
# |City|Host city|
# |Sport|Sport|
# |Event|Event|
# |Medal|Gold, Silver, Bronze, or NA|

# ### I hope you enjoy this work. In the event please give me an upvote. However I will continue to update the work.

# # 0. Setup

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

print('Numpy Version     : ', np.__version__)
print('Pandas Version    : ', pd.__version__)
print('Matplotlib Version: ', mpl.__version__)
print('Seaborn Version   : ', sns.__version__)

#seaborn options
sns.set_style('white')

#pandas options
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


# ## Functions

# In[ ]:


def missingData(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    md = md[md["Percent"] > 0]
    plt.figure(figsize = (8, 4))
    plt.xticks(rotation='90')
    sns.barplot(md.index, md["Percent"],color="r",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return md

def valueCounts(dataset, features):
    """Display the features value counts """
    for feature in features:
        vc = dataset[feature].value_counts()
        print(vc)


# # 1. Importing data

# In[ ]:


data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')


# # 2. First Data Exploration

# In[ ]:


data.sample(5)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


regions.head()


# In[ ]:


regions.info()


# # 3. Merging the dataframes

# In[ ]:


allAthletes = pd.merge(data, regions, on='NOC', how='left')


# In[ ]:


allAthletes.sample(5)


# In[ ]:


allAthletes.shape


# # 4. Checking Missing Data

# In[ ]:


allAthletes.info()


# Now I use a function to visualize the missing data.

# In[ ]:


missingData(allAthletes)


# #### The NaN values in Medal Feature are the Athletes who haven't won any medal in the specific event of an Olympiad. I'll replace them with 'No Medal'.

# In[ ]:


allAthletes['Medal'].fillna('No Medal', inplace=True)
allAthletes.drop('notes', axis=1, inplace=True)


# In[ ]:


missingData(allAthletes)


# # 5. Exploring all the Olympic Athletes

# ## 5.1 What is the mean age of an Olympic athlete?

# In[ ]:


allAthletes.describe()


# ### The mean age of an Olympic Athlete is 25.55 years.
# ### The median age of an Olympic athlete is 24 years.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x= 'Age', hue='Sex', data=allAthletes )
plt.xticks(rotation = 90)
plt.legend(loc=1, fontsize='x-large')
plt.title('Distribution of olympic athletes by Age and Sex', fontsize = 20)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(allAthletes['Age'], color='b')
plt.xticks(rotation = 90)
plt.title('Distribution of olympic athletes by Age', fontsize = 20)
plt.show()


# In[ ]:


allAthletes['ID'][allAthletes['Age'] > 60].count()


# #### Who is the oldest athlete?

# In[ ]:


allAthletes['Age'].max()


# In[ ]:


allAthletes[allAthletes['Age'] == 97].head()


# #### There is certainly an error in the dataset. I say this because John Quincy Adams Ward, the sculptor, died May 1, 1910.

# ## 5.2. What is the median height/weight of an Olympic Athlete? 

# #### There are missing values in the Weight/Height features. I can't use the average or the median to fill the NaN values, because the dataset is composed of athletes with specific weights/heights. Therefore i remove directly these samples.

# In[ ]:


allAthletesHW = allAthletes[(allAthletes['Height'].notnull()) & (allAthletes['Weight'].notnull())]


# In[ ]:


allAthletesHW.describe()


# 1. The mean Height is 175.35 cm, while the median weight is 70.67 kg.
# 2. The median Height is 175 kg, while the median Weight is 70 kg.
# 
# First, I use a scatterplot to represent this data. Later, I use boxplots to represent the evolution of these data through Games.

# In[ ]:


plt.figure(figsize=(15, 15))
sns.scatterplot(x="Height", y="Weight", hue='Medal', data=allAthletesHW)
plt.title('Height VS Weight of Olympics Athletes', fontsize=20)
plt.show()


# #### Who is the heaviest athlete?

# In[ ]:


allAthletesHW['Weight'].max()


# In[ ]:


allAthletesHW[allAthletesHW['Weight'] == 214].head()


# Ricardo Blas was the Olympic heavier athlete!
# #### Who is the higher athlete?

# In[ ]:


allAthletesHW['Height'].max()


# In[ ]:


allAthletesHW[allAthletesHW['Height'] == 226].head()


# Yao Ming was the Olympic higher athlete!

# #### At this point, following the previous reasoning, I study the variation of weight/height through the Olympics.

# In[ ]:


athletesMenHW = allAthletesHW[(allAthletesHW.Sex == 'M')]
athletesWomenHW = allAthletesHW[(allAthletesHW.Sex == 'F')]

summerMenHW = athletesMenHW[(athletesMenHW.Season == 'Summer')]
winterMenHW = athletesMenHW[(athletesMenHW.Season == 'Winter')]
summerWomenHW = athletesWomenHW[(athletesWomenHW.Season == 'Summer')]
winterWomenHW = athletesWomenHW[(athletesWomenHW.Season == 'Winter')]

print('Summer Male Athletes shape   : ', summerMenHW.shape)
print('Winter Male Athletes shape   : ', winterMenHW.shape)
print('Summer Female Athletes shape : ', summerWomenHW.shape)
print('Winter Female Athletes shape : ', winterWomenHW.shape)


# ## Weight variation

# In[ ]:


plt.figure(figsize=(30,18))

#1
plt.subplot(221)
sns.boxplot('Year', 'Weight', data=summerMenHW, palette='Oranges')
plt.title('Variation of Weight for Male Athletes in summer Olympics', fontsize=20)
#2
plt.subplot(222)
sns.boxplot('Year', 'Weight', data=summerWomenHW, palette='Oranges')
plt.title('Variation of Weight for Female Athletes in summer Olympics', fontsize=20)
#3
plt.subplot(223)
sns.boxplot('Year', 'Weight', data=winterMenHW, palette='Blues')
plt.title('Variation of Weight for Male Athletes in winter Olympics', fontsize=20)
#4
plt.subplot(224)
sns.boxplot('Year', 'Weight', data=winterWomenHW, palette='Blues')
plt.title('Variation of Weight for Female Athletes in winter Olympics', fontsize=20)

plt.show()


# The line that divides the box into 2 parts represents the median of the data. The end of the box shows the upper and lower quartiles. The extreme lines shows the highest and lowest value excluding outliers.

# ## Heigth variation

# In[ ]:


plt.figure(figsize=(30,18))

#1
plt.subplot(221)
sns.boxplot('Year', 'Height', data=summerMenHW, palette='Oranges')
plt.title('Variation of Height for Male Athletes in summer Olympics', fontsize=20)
#2
plt.subplot(222)
sns.boxplot('Year', 'Height', data=summerWomenHW, palette='Oranges')
plt.title('Variation of Height for Female Athletes in summer Olympics', fontsize=20)
#3
plt.subplot(223)
sns.boxplot('Year', 'Height', data=winterMenHW, palette='Blues')
plt.title('Variation of Height for Male Athletes in winter Olympics', fontsize=20)
#4
plt.subplot(224)
sns.boxplot('Year', 'Height', data=winterWomenHW, palette='Blues')
plt.title('Variation of Height for Female Athletes in winter Olympics', fontsize=20)

plt.show()


# The line that divides the box into 2 parts represents the median of the data. The end of the box shows the upper and lower quartiles. The extreme lines shows the highest and lowest value excluding outliers.

# ## 5.3 Variation of number of athletes through the Games

# In[ ]:


athletesMen = allAthletes[(allAthletes.Sex == 'M')]
athletesWomen = allAthletes[(allAthletes.Sex == 'F')]

print('Male Athletes shape   : ', athletesMen.shape)
print('Female Athletes shape : ', athletesWomen.shape)


# In[ ]:


athletesMen.describe()


# In[ ]:


athletesWomen.describe()


# In[ ]:


summerMen = athletesMen[(athletesMen.Season == 'Summer')]
winterMen = athletesMen[(athletesMen.Season == 'Winter')]
summerWomen = athletesWomen[(athletesWomen.Season == 'Summer')]
winterWomen = athletesWomen[(athletesWomen.Season == 'Winter')]

print('Summer Male Athletes shape   : ', summerMen.shape)
print('Winter Male Athletes shape   : ', winterMen.shape)
print('Summer Female Athletes shape : ', summerWomen.shape)
print('Winter Female Athletes shape : ', winterWomen.shape)


# In[ ]:


summerTicks = list(summerMen['Year'].unique())
summerTicks.sort()
winterTicks = list(winterMen['Year'].unique())
winterTicks.sort()


# In[ ]:


plt.figure(figsize=(30,18))

plt.subplot(221)
partSummerMen = summerMen.groupby('Year')['Sex'].value_counts()
partSummerMen.loc[:,'M'].plot(linewidth=4, color='b')
plt.xticks(summerTicks, rotation=90)
plt.title('Variation of Male Athletes in summer Olympics', fontsize=20)

plt.subplot(222)
partSummerWomen = summerWomen.groupby('Year')['Sex'].value_counts()
partSummerWomen.loc[:,'F'].plot(linewidth=4, color='r')
plt.xticks(summerTicks, rotation=90)
plt.title('Variation of Female Athletes in summer Olympics', fontsize=20)

plt.subplot(223)
partWinterMen = winterMen.groupby('Year')['Sex'].value_counts()
partWinterMen.loc[:,'M'].plot(linewidth=4, color='b')
plt.xticks(winterTicks, rotation=90)
plt.title('Variation of Male Athletes in winter Olympics', fontsize=20)

plt.subplot(224)
partWinterWomen = winterWomen.groupby('Year')['Sex'].value_counts()
partWinterWomen.loc[:,'F'].plot(linewidth=4, color='r')
plt.xticks(winterTicks, rotation=90)
plt.title('Variation of Female Athletes in winter Olympics', fontsize=20)

plt.show()


# 1. World wars led to the cancellation of the 1916, 1940, and 1944 Games.
# 2. The bidding for 1936 Games was the first to be contested by IOC members casting votes for their own favorite host cities. The vote occurred in 1931, during the final years of the Weimar Republic, two years before Adolf Hitler and the Nazi Party rose to power in 1933. The 1936 Berlin Games also saw the reintroduction of the Torch Relay.
# 3. The first Winter Olympics, the 1924 Winter Olympics, were held in Chamonix, France.
# 4. Beginning with the 1994 Games, the Winter Olympics were held every four years, two years after each Summer Olympics.
# 5. In 1980 (Moscow) and 1984 (Los Angeles), the Cold War opponents boycotted each other's Games.
# 6. The 1906 Intercalated Games or 1906 Olympic Games was an international multi-sport event that was celebrated in Athens, Greece. They were at the time considered to be Olympic Games and were referred to as the "Second International Olympic Games in Athens" by the International Olympic Committee. Whilst medals were distributed to the participants during these games, the medals are not officially recognized by the IOC today and are not displayed with the collection of Olympic medals at the Olympic Museum in Lausanne, Switzerland. (Wikipedia)

# # 6. Exploring only the Gold Medals (Winter & Summer Olympics)

# In[ ]:


#crate a dataset with gold medals
goldMedals = allAthletes[(allAthletes.Medal == 'Gold')]


# In[ ]:


goldMedals.sample(5)


# In[ ]:


goldMedals.info()


# In[ ]:


missingData(goldMedals)


# #### 6.1 What is the mean age of a Gold medal athlete?

# But first, I remove the missing data in 'Age'.

# In[ ]:


allAthletesAge = allAthletes[(allAthletes['Age'].notnull())]


# In[ ]:


missingData(allAthletesAge)


# In[ ]:


goldMedals.describe()


# #### The mean age of a gold athlete is 25.9 years.
# #### The median age of a gold athlete is 25 years.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x= 'Age', hue='Sex', data=goldMedals )
plt.xticks(rotation = 90)
plt.legend(loc=1, fontsize='x-large')
plt.title('Distribution of Gold Medals by Age and Sex', fontsize = 20)
plt.show()


# #### 6.2 Exploring the gold minors athletes

# In[ ]:


goldMedals['ID'][goldMedals['Age']<18].count()


# In[ ]:


minorsAthletes = goldMedals[goldMedals['Age']<18]


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x= 'Sport', hue='Sex', data=minorsAthletes)
plt.xticks(rotation = 90)
plt.title('Minors Gold Athletes by Sport', fontsize=20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x= 'Event', hue='Sex', data=minorsAthletes)
plt.xticks(rotation=90)
plt.title('Minors Gold Athletes by Event', fontsize=20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# In[ ]:


goldMedals['ID'][goldMedals['Age'] == 13].count()


# In[ ]:


youngestAthletes = goldMedals[goldMedals['Age'] == 13]


# In[ ]:


youngestAthletes


# In[ ]:


plt.figure(figsize=(20, 10))
ticks = [0, 1, 2, 3]
sns.countplot(x='Sport', hue='Sex', data=youngestAthletes)
plt.yticks(ticks)
plt.title('Gold Athletes thirteen', fontsize=20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# 1. Gestring was 13 years and 268 days old when she competed in the Olympics in Berlin, Germany, in 1936, and helped the U.S. women's diving team win a gold medal, according to Top End Sports.
# 2. In 1994, Kim Yun-Mi of South Korea made Olympic speed-skating history when she competed at the Lillehammer Games at the age of 13. She won the gold in the 3,000-meter relay and became the youngest Olympic champion at the Winter Games, according to Sports Reference.

# #### 6.3 Exploring the oldest athletes

# In[ ]:


goldMedals['ID'][goldMedals['Age'] > 50].count()


# In[ ]:


oldAthletes = goldMedals[goldMedals['Age'] > 50]


# In[ ]:


plt.figure(figsize=(20, 10))
ticks = [0, 5, 10, 15, 20]
plt.yticks(ticks)
sns.countplot(x = 'Sport', hue='Sex', data=oldAthletes)
plt.title('Gold Athletes over 50 by sport', fontsize=20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# #### 6.4 Gold Medals through the Games

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year', hue='Season', data=goldMedals)
plt.title('Gold medals per edition of the Games', fontsize = 20)
plt.legend(loc=2, fontsize='x-large')
plt.show()


# 1. In the 1920 Games there are 156 events in 29 disciplines (comprising 22 sports), while in the 1924 Games there are 126 events in 23 disciplines (comprising 17 sports).

# #### 6.5 Countries with more gold medals.

# In[ ]:


goldMedals.region.value_counts().reset_index(name='Medal').head(10)


# In[ ]:


totalGoldMedals = goldMedals.region.value_counts().reset_index(name='Medal').head(5)

sns.catplot(x="index", y="Medal", data=totalGoldMedals,
                height=6, kind="bar", palette='afmhot')
plt.xlabel("Countries")
plt.ylabel("Number of Medals")
plt.title('Top 5 Countries', fontsize = 20)
plt.show()


# #### 6.6 What is the mean height/weight of an Olympic Medalist?

# In[ ]:


goldMedals.sample(5)


# In[ ]:


goldMedals.info()


# In[ ]:


goldMedalsHW = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]


# In[ ]:


markers = {"Summer": "d", "Winter": "h"}

plt.figure(figsize=(15, 15))
sns.scatterplot(x="Height", y="Weight", hue='Sex', style='Season', data=goldMedalsHW, markers=markers)
plt.title('Height VS Weight of Gold Medalists', fontsize=20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# The majority of the gold athletes have a linear relation between height and weight.  Now I check what sports do gold athletes with weight over 140kg.

# In[ ]:


goldOver140kg = goldMedalsHW.loc[goldMedalsHW['Weight'] > 140]


# In[ ]:


valueCounts(goldOver140kg, ['Sport'])


# In[ ]:


goldOver140kg


# There are only Weightlifter except two athletes, which one is a shot putter, Tomasz Majewski, and the other is a judoka, Hitoshi Saito.
# Now I check what sports do athletes with a height of over 200cm. But I think I already know the answer.

# In[ ]:


over200cm = goldMedalsHW.loc[goldMedalsHW['Height'] > 200]


# In[ ]:


valueCounts(over200cm, ['Sport'])


# Basketball and Volleyball athletes. Seems logic.

# # 7. Italian Athletes

# In[ ]:


itAthletes = allAthletes[(allAthletes.region == 'Italy')].sort_values('Year')


# In[ ]:


itAthletes.info()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year', hue='Season',data=itAthletes)
plt.title('Italian athletes per edition of the Games by Season', fontsize = 20)
plt.legend(loc=2, fontsize='x-large')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year',hue='Sex', data=itAthletes)
plt.title('Italian athletes per edition of the Games by Sex', fontsize = 20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# In[ ]:


itMen = itAthletes[(itAthletes.Sex == 'M')]
itWomen = itAthletes[(itAthletes.Sex == 'F')]


# In[ ]:


markers = {"Summer": "d", "Winter": "h"}

plt.figure(figsize=(15, 15))
sns.scatterplot(x="Height", y="Weight", hue='Medal', style='Season', data=itAthletes.sort_values('Year'), markers=markers)
plt.title('Height VS Weight of Olympics Italian Athletes', fontsize=20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# ## 7.1 Italian Gold medals

# In[ ]:


goldMedalsITA = itAthletes[(itAthletes.Medal == 'Gold')]


# #### By age

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Age', hue='Sex', data=goldMedalsITA)
plt.title('Distribution of Italian Gold Medals by Age', fontsize = 20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(goldMedalsITA['Age'], color='b')
plt.title('Distribution of Italian Gold Medals by Age', fontsize = 20)
plt.show()


# #### Through the games

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year',hue='Sex', data=goldMedalsITA)
plt.title('Italian Gold medals per edition of the Games', fontsize = 20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year', hue='Sex', data=goldMedalsITA)
plt.title('Distribution of Italian Gold Medals by Year', fontsize = 20)
plt.legend(loc=1, fontsize='x-large')
plt.show()


# #### Note:
# 
# |City|Year|Italian Gold Medals|
# |----|----|-------------------|
# |Los Angeles|1984|14|
# |Rome|1960|13|
# |Anversa|1920|13|
# 
# Anversa 1920 is the year with the most golds. But this is a dataset of athletes. In fact, in 1920, Italy won gold in team gymnastics. (source Wikipedia)

# In[ ]:


goldMedalsITA.Event.value_counts().reset_index(name='Medal').head(10)


# In fact it is the event where Italy has won more gold medals.At this point I divide the gold medals between men and women.

# In[ ]:


itGoldMen = goldMedalsITA[(goldMedalsITA.Sex == 'M')]
itGoldWomen = goldMedalsITA[(goldMedalsITA.Sex == 'F')]


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=itGoldMen, color='b')
plt.title('Male Italian Gold medals per edition of the Games', fontsize = 20)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=itGoldWomen, color='r')
plt.title('Female Italian Gold medals per edition of the Games', fontsize = 20)
plt.show()


# In[ ]:


plt.figure(figsize=(30,20))

#1
plt.subplot(221)
sns.countplot(x='Year', hue='Season', data=itMen)
plt.xticks(rotation=90)
plt.title('Italian Men per edition of the Games', fontsize = 20)
plt.legend(loc=2, fontsize='large')

#2
plt.subplot(222)
sns.countplot(x='Year', hue='Season', data=itWomen)
plt.xticks(rotation=90)
plt.title('Italian Women per edition of the Games', fontsize = 20)
plt.legend(loc=2, fontsize='large')
#3
plt.subplot(223)
sns.countplot(x='Year', hue='Season', data=itGoldMen)
plt.xticks(rotation=90)
plt.title('Male Italian Gold medals per edition of the Games', fontsize = 20)
plt.legend(loc=2, fontsize='large')
#4
plt.subplot(224)
sns.countplot(x='Year', hue='Season', data=itGoldWomen)
plt.xticks(rotation=90)
plt.title('Female Italian Gold medals per edition of the Games', fontsize = 20)
plt.legend(loc=2, fontsize='large')

plt.show()


# ### Variation of Weight/Height of Italian gold Medals

# In[ ]:


markers = {"Summer": "d", "Winter": "h"}

plt.figure(figsize=(15, 15))
sns.scatterplot(x="Height", y="Weight", hue='Sex',style='Season', data=goldMedalsITA)
plt.title('Height VS Weight of Italian Gold Medals', fontsize=20)
plt.show()


# ### To be continued 

# In[ ]:




