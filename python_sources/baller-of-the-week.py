#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df=pd.read_csv('../input/NBA_player_of_the_week.csv')
df.head(2)
# Any results you write to the current directory are saved as output.


# Ok , everything seems fine. But lets see the columns, if there are any columns that should be numeric, but are not, will be changed as such

# In[ ]:


df.info()
df['Height'].unique()


# Ah there you go! some of the heights are expressed as ft-in and some are displayed as x cm. Since most of them are expressed as ft-in, we'll convert all the values to feet, and while we are at it, lets also convert the date acoordingly.
# 
# We'll also add a new column that acts as a bin for the height , although we could do that through a histogram, I added this like a categorical column.

# In[ ]:


df['Date']=pd.to_datetime(df['Date'])
def height_categ(x):
    """
    """
    a=x.split('-')
    if len(a) > 1:
        a=float(a[0])+float(a[1])/12
        if (a<6.0):
            return 'below 6'
        elif (a>=6.0) & (a<=6.5):
            return '6 to 6.5 feet'
        elif (a>6.5) & (a<7.0):
            return '6.5 to 7 feet'
        elif (a>=7.0):
            return 'over 7'
    else:
        a=x
        a=a.replace('cm',' ')
        a=float(a)*0.0328084
        if (a<6.0):
            return 'below 6'
        elif (a>=6.0) & (a<=6.5):
            return '6 to 6.5 feet'
        elif (a>6.5) & (a<7.0):
            return '6.5 to 7 feet'
        elif (a>=7.0):
            return 'over 7'

df['Height_category']=df['Height'].apply(height_categ)


# Now that we are done with that, lets plot and check.

# In[ ]:


#df['Height_category'].unique()
#df['Position'].unique()
temp=df.groupby('Height_category')['Height_category'].count()
plt.bar(temp.index,temp)


# Ok, most of the players lie in the range of 6-7 feet and thats pretty obvious! However this is something surprising that we have players below 6 as well! thats cool. Now lets look at the weight.

# In[ ]:


df['Weight'].unique()


# Yup, by the looks of it, the weight is expressed both in kilograms and pounds (after all, what athelete would weigh like 310 Kgs????) Lets convert that as well.

# In[ ]:


def conv_weight(x):
    if 'kg' in x:
        a=x
        a=x.replace('kg','')
        a=float(a)*2.20462
        a=int(a)
        return a 
    else:
        return x
df['Weight']=df['Weight'].apply(conv_weight)
def weight_categ(x):
    if x>100 and x<151:
        return '100 to 150 pounds'
    elif x>150 and x<201:
        return '150 to 200 pounds'
    elif x>200 and x<251:
        return '200 to 250 pounds'
    elif x>250 and x<301:
        return '250 to 300 pounds'
    elif x>300:
        return 'over 300 pounds!!!'
    else:
        return 'None'
df['Weight']=pd.to_numeric(df['Weight']);
df['weight_categ']=df['Weight'].apply(weight_categ);
df['weight_categ'].unique()


# So we have the weight categories as well. Really hope the 3 exclamation marks after 300 pounds doesnt overdo it. Lets look at the positions now.

# In[ ]:


df['Position'].unique()


# So its kinda like abbreviated positions. Ok well, I being a foreigner to basketball (and U.S) may not know much about positions in basketball but some of the folks over at a focussed reddit group seem to know a great deal. Check 'em out! Here's where I got the info from:
# [reddit link](https://www.reddit.com/r/nba/comments/g6i6k/can_someone_explain_the_various_positions_in/)
# 

# In[ ]:


#https://www.reddit.com/r/nba/comments/g6i6k/can_someone_explain_the_various_positions_in/
positions={
'PG' : 'point guard' ,
'SG' : 'shooting guard'  ,
'F' : 'forward' ,
'C' : 'center' ,
'SF' : 'small forward' ,
'PF' : 'power forward' ,
'G' :  'guard' ,
'FC' : 'forward center'  ,
'GF' : 'guard forward' ,
'F-C': 'forward center'  ,
'G-F': 'guard forward' 
}
df['positions_descr']=df['Position'].map(positions)
df[['Position','positions_descr']].head(2)


# Now that that's all set nice and good, lets get to the fun part of the kernel.
# 
# My first question to the prospective data scientist and an NBA baller in you is , According to height , which position should be played by which height ? Lets find out!!!

# In[ ]:


plt.scatter(df['Height_category'],df['positions_descr'])


# In[ ]:


#plt.subplot(4,1,1)
temp=df[df['Height_category']=='below 6']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.bar(temp.index,temp)
plt.xticks(rotation=60)
plt.title('below 6 feet player positions' )
plt.show()

#plt.subplot(4,1,2)
temp=df[df['Height_category']=='6 to 6.5 feet']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.xticks(rotation=60)
plt.bar(temp.index,temp)
plt.title('6 to 6.5 feet player positions')
plt.show()

#plt.subplot(4,1,3)
temp=df[df['Height_category']=='6.5 to 7 feet']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.xticks(rotation=45)
plt.bar(temp.index,temp)
plt.title('6.5 to 7 feet player positions')
plt.show()

#plt.subplot(4,1,3)
temp=df[df['Height_category']=='over 7']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.xticks(rotation=45)
plt.bar(temp.index,temp)
plt.title('over 7 feet player positions')
plt.show()


# Wow, so couple of things.
# 6.5 to 7 feet players can play the most positions.
# The extremes, over 7 feet and below 6 feet dont enjoy a lot of diversity in their positions.
# Also the 6.5 to 7 feet players kinda go for the power forward and forward positions a lot more than other positions.
# 
# 
# Ok, now that that's all dealt with , 
# 
# My second question to the prospective data scientist in you is , What's the fortune formula for a successful basketball player in terms of height and weight? Read on below to find out!!!

# In[ ]:


temp=df
temp['count']=0
temp=temp.groupby(['Height_category','weight_categ'])['count'].count()
temp=temp.reset_index()
temp.sort_values(ascending=False,by='count')


# Ok, so we see that more than half of the successful players lie in the category of 6.5 - 7 feet with 200 to 250 pound weight. 
# 
# Let's see it a lot differently, The top five only and in one place in a chart.

# In[ ]:


temp['ht_wt_category']=temp['Height_category']+ ' and ' +temp['weight_categ']
temp=temp.sort_values(ascending=False,by='count')
plt.barh(temp['ht_wt_category'].head(5),temp['count'].head(5))
plt.xticks(rotation=45)


# Pretty impressive huh? Lets see a simpler plot and see which of the players positions are most likely to get a player of the week award by seeing the total entries in the dataset grouped by positions.

# In[ ]:


temp=df.groupby('positions_descr')['count'].count()
temp=temp.sort_values(ascending=False)
plt.bar(temp.index,temp)
plt.xticks(rotation=60)


# And the age group most likely to get a player of the week award?

# In[ ]:


temp=df.groupby(['Age'])['count'].count()
temp=temp.sort_values(ascending=False)
plt.bar(temp.index,temp)


# While we are at it , lets see the team with the most player of the week awards.

# In[ ]:


temp=df.groupby(['Team'])['count'].count()
temp=temp.sort_values(ascending=False)
plt.bar(temp.index[:5],temp[:5])
plt.xticks(rotation=60)


# Great!!! 
# 
# Really hope you enjoyed reading through this as much as i enjoyed creating it. However, we dont need to stop here. There's still a lot of un answered questions.
# 
# Which team has the highest number of player of the week awards per season?
# Does that team's winning player's age comply with the Age vs awards graph we checked ?
# 
# I know there can be a lot more questions that can be covered and I'd love to hear them from you in the comments.
# 
# Any and all feedback's most wellcome , Thanks!
