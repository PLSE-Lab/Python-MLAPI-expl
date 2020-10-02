#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to aggregate the player rankings of IPL players from 2008-2019 and profile the players.
# Player scores are aggregated based on reducing weightages for every year. The data has been scraped from iplt20.com

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax=plt.subplots()


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df2019=pd.read_excel('/kaggle/input/IPL_Player_Scores_2019.xlsx')
df2018=pd.read_excel('/kaggle/input/IPL_Player_Scores_2018.xlsx')
df2017=pd.read_excel('/kaggle/input/IPL_Player_Scores_2017.xlsx')
df2016=pd.read_excel('/kaggle/input/IPL_Player_Scores_2016.xlsx')
df2015=pd.read_excel('/kaggle/input/IPL_Player_Scores_2015.xlsx')
df2014=pd.read_excel('/kaggle/input/IPL_Player_Scores_2014.xlsx')
df2013=pd.read_excel('/kaggle/input/IPL_Player_Scores_2013.xlsx')
df2012=pd.read_excel('/kaggle/input/IPL_Player_Scores_2012.xlsx')
df2011=pd.read_excel('/kaggle/input/IPL_Player_Scores_2011.xlsx')
df2010=pd.read_excel('/kaggle/input/IPL_Player_Scores_2010.xlsx')
df2009=pd.read_excel('/kaggle/input/IPL_Player_Scores_2009.xlsx')
df2008=pd.read_excel('/kaggle/input/IPL_Player_Scores_2008.xlsx')


# In[ ]:


df2019.head(1)


# In[ ]:


df2017.head(1)


# Setting player names as index for all dataframes

# In[ ]:


df2019=df2019.set_index('PLAYER')
df2018=df2018.set_index('PLAYER')
df2017=df2017.set_index('PLAYER')
df2016=df2016.set_index('PLAYER')
df2015=df2015.set_index('PLAYER')
df2014=df2014.set_index('PLAYER')
df2013=df2013.set_index('PLAYER')
df2012=df2012.set_index('PLAYER')
df2011=df2011.set_index('PLAYER')
df2010=df2010.set_index('PLAYER')
df2009=df2009.set_index('PLAYER')
df2008=df2008.set_index('PLAYER')


# In[ ]:


#checking duplicates
df2008[df2008.index.duplicated()]
df2009[df2009.index.duplicated()]
df2010[df2010.index.duplicated()]
df2011[df2011.index.duplicated()]
df2012[df2012.index.duplicated()]
df2013[df2013.index.duplicated()]
df2014[df2014.index.duplicated()]
df2015[df2015.index.duplicated()]
df2016[df2016.index.duplicated()]
df2017[df2017.index.duplicated()]
df2018[df2018.index.duplicated()]
df2019[df2019.index.duplicated()]


# Checking the length for all years

# In[ ]:


pd.DataFrame(
   {'len2019': [df2019.index.size],
    'len2018': [df2018.index.size],
    'len2017': [df2017.index.size],
    'len2016': [df2016.index.size],
    'len2015': [df2015.index.size],
    'len2014': [df2014.index.size],
    'len2013': [df2013.index.size],
    'len2012': [df2012.index.size],
    'len2011': [df2011.index.size],
    'len2010': [df2010.index.size],
    'len2009': [df2009.index.size],
    'len2008': [df2008.index.size],
   })


# Only interested in players that played in 2019, removing other players.

# In[ ]:


df2018=df2018[df2018.index.isin(df2019.index)]
df2017=df2017[df2017.index.isin(df2019.index)]
df2016=df2016[df2016.index.isin(df2019.index)]
df2015=df2015[df2015.index.isin(df2019.index)]
df2014=df2014[df2014.index.isin(df2019.index)]
df2013=df2013[df2013.index.isin(df2019.index)]
df2012=df2012[df2012.index.isin(df2019.index)]
df2011=df2011[df2011.index.isin(df2019.index)]
df2010=df2010[df2010.index.isin(df2019.index)]
df2009=df2009[df2009.index.isin(df2019.index)]
df2008=df2008[df2008.index.isin(df2019.index)]


# checking the lengths again, very few players have been a part of IPL since 2008

# In[ ]:


pd.DataFrame(
    {'len2019': [df2019.index.size],
     'len2018': [df2018.index.size],
     'len2017': [df2017.index.size],
     'len2016': [df2016.index.size],
     'len2015': [df2015.index.size],
     'len2014': [df2014.index.size],
     'len2013': [df2013.index.size],
     'len2012': [df2012.index.size],
     'len2011': [df2011.index.size],
     'len2010': [df2010.index.size],
     'len2009': [df2009.index.size],
     'len2008': [df2008.index.size],
    })


# IPL computes scores with linear weightages. 3.5 wkts, 1 dot ball, 2.5 4s, 3.5 6s, 2.5 chatches, 2.5 stumpings. Verifying the same.

# In[ ]:


#performing linear regression on 2019 to see how scores are calculates
X=df2019.iloc[:,2:]
y=df2019.loc[:,'Pts']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# Running logistic regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
result = model.fit(X_train, y_train)
result.score(X_train,y_train)
#Score somes as 1 . full match


# In[ ]:


#mean sq error
np.mean((y_test-result.predict(X_test))**2)
#very little values. effectively zero


# In[ ]:


result.coef_
#These can be used as weights for calculating any scores.


# In[ ]:


#attribute scores
#Wkts	Dots	Fours	Sixes	Catches	Stumpings
#3.5, 1. , 2.5, 3.5, 2.5, 2.5
attribute_weights = {
    'Wkts':3.5, 
    'Dots':1, 
    'Fours':2.5,
    'Sixes' :3.5,
    'Catches' :2.5,
    'Stumpings':2.5
}


# Computing batting, bowling and fielding scores seperately.

# In[ ]:


def computeScores(df, attribute_weights):
    df['battingPts']=df['Fours']*attribute_weights.get('Fours')+df['Sixes']*attribute_weights.get('Sixes')
    df['bowlingPts']=df['Wkts']*attribute_weights.get('Wkts')+df['Dots']*attribute_weights.get('Dots')
    df['fieldingPts']=df['Catches']*attribute_weights.get('Catches')+df['Stumpings']*attribute_weights.get('Stumpings')
    return df


# Adding bowling, batting and fielding scores to the existing dataframes

# In[ ]:


df2019=computeScores(df2019,attribute_weights)
df2018=computeScores(df2018,attribute_weights)
df2017=computeScores(df2017,attribute_weights)
df2016=computeScores(df2016,attribute_weights)
df2015=computeScores(df2015,attribute_weights)
df2014=computeScores(df2014,attribute_weights)
df2013=computeScores(df2013,attribute_weights)
df2012=computeScores(df2012,attribute_weights)
df2011=computeScores(df2011,attribute_weights)
df2010=computeScores(df2010,attribute_weights)
df2009=computeScores(df2009,attribute_weights)
df2008=computeScores(df2008,attribute_weights)


# keeping the weight factor as .2, every previous year has 20% less weightage than the current year.

# In[ ]:


alpha=.2
season_weights={
    '2019':1,
    '2018':1-alpha,
    '2017':(1-alpha)**2,
    '2016':(1-alpha)**3,
    '2015':(1-alpha)**4,
    '2014':(1-alpha)**5,
    '2013':(1-alpha)**6,
    '2012':(1-alpha)**7,
    '2011':(1-alpha)**8,
    '2010':(1-alpha)**9,
    '2009':(1-alpha)**10,
    '2008':(1-alpha)**11
}


# Checking the weights

# In[ ]:


season_weights.items()


# In[ ]:


#columns to be selected
columns=['Pts','battingPts','bowlingPts','fieldingPts']


# Calculating weighted scores

# In[ ]:


#calculating scores with historic data
dfCombined=(df2019[columns]*season_weights.get('2019')).add(
    df2018[columns]*season_weights.get('2018'),fill_value=0).add(
    df2017[columns]*season_weights.get('2017'),fill_value=0).add(
    df2016[columns]*season_weights.get('2016'),fill_value=0).add(
    df2015[columns]*season_weights.get('2015'),fill_value=0).add(
    df2014[columns]*season_weights.get('2014'),fill_value=0).add(
    df2013[columns]*season_weights.get('2013'),fill_value=0).add(
    df2012[columns]*season_weights.get('2012'),fill_value=0).add(
    df2011[columns]*season_weights.get('2011'),fill_value=0).add(
    df2010[columns]*season_weights.get('2010'),fill_value=0).add(
    df2009[columns]*season_weights.get('2009'),fill_value=0).add(
    df2008[columns]*season_weights.get('2008'),fill_value=0)


# In[ ]:


print('top 5 overall')
dfCombined.sort_values(columns,ascending=[False,False,False,False]).head()   


# In[ ]:


print('top 5 batsman')
dfCombined.sort_values(['battingPts'],ascending=[False]).head()


# In[ ]:


print('top 5 bowlers')
dfCombined.sort_values(['bowlingPts'],ascending=[False]).head()


# In[ ]:


print('top 15 fielders')
print('At the top all wicketkeepers due to stumping score')
dfCombined.sort_values(['fieldingPts'],ascending=[False]).head(15)


# In[ ]:


print('top all rounders')
dfCombined[(dfCombined['battingPts']>dfCombined['battingPts'].mean()) &
           (dfCombined['bowlingPts']>dfCombined['bowlingPts'].mean()) &
           (dfCombined['fieldingPts']>dfCombined['fieldingPts'].mean())]  


# Axar Patel and Ben Stokes, dont shine in any individual department, but top the list as all rounders

# Scaling the data for plotting.

# In[ ]:


#scaling the data
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
afC=scale.fit_transform(dfCombined)
dfC=pd.DataFrame(afC,columns=['Pts','Batting','Bowling','Fielding'],index=np.arange(dfCombined.index.size))
dfC.index=dfCombined.index
dfC=dfC*100#Best Scaled values
#top 5 aggregate
dfBest=dfC.sort_values(['Pts'],ascending=[False]).head()
#top 5 batsman
dfBatsman=dfC.sort_values(['Batting'],ascending=[False]).head()

#top 5 bowlers
dfBowlers=dfC.sort_values(['Bowling'],ascending=[False]).head()

#top 5 fielders
#these are all wicketkeepers due to stimping score
dfwicketKeepers=dfC.sort_values(['Fielding'],ascending=[False]).head()

#top 5 all rounders
dfAllrounders=dfC[(dfC['Batting']>dfC['Batting'].mean()) &
           (dfC['Bowling']>dfC['Bowling'].mean()) &
           (dfC['Fielding']>dfC['Fielding'].mean())]


# Creating Spiderplots

# In[ ]:


def spiderplot(df, indx):
    #taking values out of DF
    values=df.iloc[indx].values.flatten().tolist()
    values+=values[:1]
    #defining colors
    import random
    r = lambda: random.randint(0,255)
    colorRandom = '#%02X%02X%02X' % (r(),r(),r())
    if colorRandom == '#ffffff':colorRandom = '#a5d6a7'
    basic_color = '#37474f'
    color_annotate = '#01579b'
    #spider plot
    from math import pi
    Categories=list(dfCombined)[1:]
    N=len(Categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(7,7))
    ax=plt.subplot(111,projection='polar')
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1],Categories,color='black',size=17)
    plt.yticks([25,50,75,100],["low","average","good","great"],color=color_annotate, size=10)
    plt.ylim(0,100)
    ax.plot(angles,values,color=basic_color,linewidth=1, linestyle='solid')
    ax.fill(angles,values,color=colorRandom,alpha=.3)
    plt.title(df.index[indx],size=20)
    plt.show()


# In[ ]:


print("OverAll Best after IPL 2019")
spiderplot(dfBest.iloc[:,1:],0)
print("Best Batsman")
spiderplot(dfBatsman.iloc[:,1:],0)
print("Best Bowler")
spiderplot(dfBowlers.iloc[:,1:],0)
print("Best Wicketkeeper")
spiderplot(dfwicketKeepers.iloc[:,1:],0)
print("Best Allrounder")
spiderplot(dfAllrounders.iloc[:,1:],0)


# In[ ]:




