#!/usr/bin/env python
# coding: utf-8

# ## Ramen: Data Insight
# 
# It's a weekend and I thought of working on some intresting data set and after couple of random search, this Ramen data seem pretty intresting. I will try to extract various insight as part of EDA practice. I am a grad student and Ramen would probably be 60% constituent of my blood. Its tasty, so quick to make and cheap!!! I can't imagine any substitute for that.
# 
# This is a simple Data Analysis/Tutorial : Feel free to add more parts and ask intresting question.
# 

# In[1]:


#Load Basic Libraries
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# In[4]:


ramen = pd.read_csv("../input/ramen-ratings.csv")
ramen.head()


# So, the data have seven columns. Let try to answer some the question. Let get some basic stats from the data.

# In[18]:


#How many entries brands are there?
ramen.groupby(['Brand']).count()['Review #']


# In[20]:


print('Total Unique Brand Entries: ',pd.unique(ramen['Brand']).shape[0])


# 1. WoW! There are 355 Brands data. Let us explore to get top 10 Brands which has largest number of reviews rows

# In[27]:


ramen.groupby(['Brand']).count()['Review #'].nlargest(10)


# Great!  For these largest 10 review count lets try to find which countries voted for them.
# 

# In[62]:


top_brand = list(ramen.groupby(['Brand']).count()['Review #'].nlargest(10).index)
df_agg = ramen.groupby(['Brand','Country']).count()


# In[107]:


df_agg.iloc[df_agg.index.isin(top_brand,level = 'Brand')].sort_values(by='Review #',ascending=False).sort_index(level='Brand', sort_remaining=False)


# Wow! You can see Nissin and Japan are having triple digit connection! US also like Nissin or choose to review Nissin. (Its my favourite also). 
# 
# We know the top ten Brand from the ranking. Does Top rank also mean largest reviews? Lets find out

# In[115]:


ramen[pd.notna(ramen['Top Ten'])]


# Oopz Some cleaning is required to get our result.

# In[132]:


def get_year(val):
    if len(val)<4:
        return np.NaN #to handle '\n'
    return int(val[:4])

def get_rank(val):
    if len(val)<4:
        return np.NaN #to handle '\n'
    return int(val[6:])

#Remove NAN terms
df_rank = ramen[pd.notna(ramen['Top Ten'])].reset_index()
df_rank['Year of Rank'] = df_rank['Top Ten'].apply(lambda x: get_year(x))
df_rank['Rank'] = df_rank['Top Ten'].apply(lambda x: get_rank(x))
df_rank.dropna(inplace=True)


# In[135]:


df_rank.sort_values(by=['Year of Rank','Rank'],ascending=[False,True])


# So, Top Rankers are not having largest reviews. However the ranking is not complete so can't comment. Finally Lets try to find out which Brand has the highest ranking Star.
# 
# To tackle this I will try to count reviews where star == 5 

# In[150]:


ramen.loc[ramen['Stars'] == 'Unrated','Stars'] = 0 # To remove the unrated columns
ramen['Stars'] = ramen['Stars'].astype(float)


# In[169]:


print("Reviews with 5 Star Ratings : ", ramen[ramen['Stars'] == 5].shape[0])
Brand_count = ramen[ramen['Stars'] == 5].groupby('Brand').count()['Stars'].nlargest(1).values[0]
Brand_name =  ramen[ramen['Stars'] == 5].groupby('Brand').count()['Stars'].nlargest(1).index.values[0]
print("Brand which has highest 5 star Rating: ", Brand_name," with Ratings: ", Brand_count)


# Yipppee!! Winner is Nissin. Sorry for being "Baised". Let do one last analysis (I am enjoying it so much). Let Try to see how Nissin Ranking is overall doing across the globe.

# In[170]:


nissin_data = ramen[ramen['Brand'] =='Nissin']
nissin_data['Stars'].describe()


# You can see the ranking is as low as 1.5! Lets try to find out why so using our data. We can use boxplot to identify the top two countries which are not happy

# In[174]:


nissin_data.boxplot(column=['Stars'], by='Country',fontsize=15,rot=90)


# So, India and USA are two are where you can see some low ranking and the range is also telling us to dive deeper into the result. I think I will stop my analysis for now and will try to come back to do some more of the analysis. Let me know what you think of my simple analysis. I 
