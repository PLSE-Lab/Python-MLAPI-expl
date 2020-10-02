#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
df=pd.read_csv('../input/videogamesales/vgsales.csv')


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()
df.dropna(inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df['Name'].unique()


# In[ ]:


df['Genre'].unique()


# In[ ]:


df['Platform'].unique()


# In[ ]:


df.columns


# In[ ]:


#Analysing sales for different genres:

def analysingGenresOverTime(tempGenre,tempRegion):
    
    tempDates=[]
    tempSales=[]

    for i in range(len(df['Genre'])):
       
        if df['Genre'].iloc[i]==tempGenre:
            tempSales.append(df[tempRegion].iloc[i])
            tempDates.append(int(df['Year'].iloc[i]))
        
        
    zippedLists=zip(tempDates,tempSales)  
    sortedPairs=sorted(zippedLists)
    
    tuples = zip(*sortedPairs)
    list1, list2 = [ list(tuple) for tuple in  tuples]
    
    return list1,list2


# In[ ]:


#Analysing 'Racing' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Racing','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Racing Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Racing','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Racing Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Racing','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Racing Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


# In[ ]:


#Analysing 'Action' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Action','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Action Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Action','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Action Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Action','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Action Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


# In[ ]:


#Analysing 'Strategy' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Strategy','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Strategy Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Strategy','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Strategy Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Strategy','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Strategy Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


# In[ ]:


#Analysing 'Strategy' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Puzzle','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Puzzle Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Puzzle','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Puzzle Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Puzzle','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Puzzle Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


# In[ ]:


#Analysing 'Strategy' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Shooter','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Shooter Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Shooter','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Shotter Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Shooter','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Shooter Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


# For the North American region 'Shooting' games in 1984-85 seemed to have a surge in sales followed by a spike in 'Puzzle' games in 1990. Beyond that 'Racing' games seemed to have been the choice for the NA people very closely followed by 'Shooting' games.

# In[ ]:


#Analysing North AMerican sales over genres: 
tempNames=df['NA_Sales'].groupby(df['Genre']).mean().keys()
tempLst=list(df['NA_Sales'].groupby(df['Genre']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Genre of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the North American region')
plt.show()


#Analysing Europeran sales over genres: 
tempNames=df['EU_Sales'].groupby(df['Genre']).mean().keys()
tempLst=list(df['EU_Sales'].groupby(df['Genre']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Genre of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the European region')
plt.show()


#Analysing Japan sales over genres: 
tempNames=df['JP_Sales'].groupby(df['Genre']).mean().keys()
tempLst=list(df['JP_Sales'].groupby(df['Genre']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Genre of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the Japan')
plt.show()


# From the above Bar-charts it seem that 'Platform' based games and 'Shooting' based games have the highest average sales in the North American region over the course of years.
# The EU region shows similar trends as well with 'Shooting' based game having the highest sales closely followed by 'Platform' based games. 
# Japan seems to have a different trend with 'Shooter' based games having the least sales and 'Role Playing' games having the highest sales followed by 'Platform' based games. 

# In[ ]:


#Analysing sales over different platforms:

#Analysing North AMerican sales over genres: 
tempNames=df['NA_Sales'].groupby(df['Platform']).mean().keys()
tempLst=list(df['NA_Sales'].groupby(df['Platform']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Platform of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the North American region')
plt.show()


#Analysing Europeran sales over genres: 
tempNames=df['EU_Sales'].groupby(df['Platform']).mean().keys()
tempLst=list(df['EU_Sales'].groupby(df['Platform']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Platform of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the European region')
plt.show()


#Analysing Japan sales over genres: 
tempNames=df['JP_Sales'].groupby(df['Platform']).mean().keys()
tempLst=list(df['JP_Sales'].groupby(df['Platform']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Platform of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the Japan')
plt.show()


# For the North American region the 'NES' games seemed to have the highest average sales over the course of years. 
# For the EU region the 'GB' games seemed to have the highest sales over the course of years. 
# For Japan the 'NES' games seemed to have the highest mean sales over the course of years followed by 'GB' games. 

# In[ ]:


def topKGames(tempRegion):

    gameDict=dict()

    for i in range(len(df['Name'])):
        if df['Name'].iloc[i] not in gameDict.keys():
            gameDict.update({df['Name'].iloc[i]:tempRegion.iloc[i]})
        else:
            gameDict[df['Name'].iloc[i]]+=tempRegion.iloc[i]
            
    tempTupls=list(sorted(gameDict.items(),key=lambda x:x[1],reverse=True))
    
    tempNames=list(map(lambda temp:temp[0],tempTupls))
    tempVals=list(map(lambda temp:temp[1],tempTupls))
    
    return tempNames,tempVals


# In[ ]:


#Total Sales of top-5 games over the years for the NA region: 

tempNames,tempVals=topKGames(df['NA_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.xlabel('Top-5 games')
plt.title('Top-5 sold games in the North American region')
plt.show()


# In[ ]:


#Total Sales of top-5 games over the years for the European region: 

tempNames,tempVals=topKGames(df['EU_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.xlabel('Top-5 games')
plt.title('Top-5 sold games in the European region')
plt.show()


# In[ ]:


#Total Sales of top-5 games over the years for the Japanese region: 

tempNames,tempVals=topKGames(df['JP_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.xlabel('Top-5 games')
plt.title('Top-5 sold games in the Japanese region')
plt.show()


# In[ ]:


#top-k publishers:

def topKPublishers(tempRegion):

    gameDict=dict()

    for i in range(len(df['Publisher'])):
        if df['Publisher'].iloc[i] not in gameDict.keys():
            gameDict.update({df['Publisher'].iloc[i]:tempRegion.iloc[i]})
        else:
            gameDict[df['Publisher'].iloc[i]]+=tempRegion.iloc[i]
            
    tempTupls=list(sorted(gameDict.items(),key=lambda x:x[1],reverse=True))
    
    tempNames=list(map(lambda temp:temp[0],tempTupls))
    tempVals=list(map(lambda temp:temp[1],tempTupls))
    
    return tempNames,tempVals


# In[ ]:


#Total Sales of top-5 publishers over the years for the NA region: 

tempNames,tempVals=topKPublishers(df['NA_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.title('Top-5 publisheres in the North American region')
plt.show()


# In[ ]:


#Total Sales of top-5 publishers over the years for the European region: 

tempNames,tempVals=topKPublishers(df['EU_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.title('Top-5 publisheres in the European region')
plt.show()


# In[ ]:


#Total Sales of top-5 publishers over the years for the Japanese region: 

tempNames,tempVals=topKPublishers(df['JP_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.title('Top-5 publisheres in the japanese region')
plt.show()


# Nintendo seems to be the highest selling company/publisher in all the three regions. Ubisoft and Activision seems to one of the top-5 in the North American and European markets. Sony seems to be there in the top-5 in the Japanese market. 

# In[ ]:




