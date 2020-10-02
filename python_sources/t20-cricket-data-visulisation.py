#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


matches = pd.read_csv('/kaggle/input/t20matches/t20_matches.csv')
series = pd.read_csv('/kaggle/input/t20matches/t20_series.csv')
matches.shape,series.shape


# In[ ]:


matches.head(5)


# In[ ]:


matches.home.nunique()


# In[ ]:


matches.away.nunique()


# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns

names = ' '
for name in matches.home:
    name = str(name)
    names = names + name + ' '
from wordcloud import WordCloud, STOPWORDS 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',  
                min_font_size = 10).generate(names) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[ ]:


names = ' '
for name in matches.away:
    name = str(name)
    names = names + name + ' '
from wordcloud import WordCloud, STOPWORDS 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',  
                min_font_size = 10).generate(names) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[ ]:


values = matches.away.value_counts().sort_values(ascending=False).head(10)
labels = values.index
plt.figure(figsize=(15,8))
sns.barplot(x=values, y=labels)


# In[ ]:


values = matches.home.value_counts().sort_values(ascending=False).head(10)
labels = values.index
plt.figure(figsize=(15,8))
sns.barplot(x=values, y=labels)


# In[ ]:


matches.columns


# In[ ]:


values = matches.winner.value_counts().sort_values(ascending=False).head(10)
labels = values.index
plt.figure(figsize=(15,8))
sns.barplot(x=values, y=labels)


# In[ ]:


matches['innings1_runs'].max(),matches['innings1_runs'].min()


# In[ ]:


#Match with maximum score of 263 runs
matches.loc[matches['innings1_runs'].idxmax()]


# In[ ]:


#Match with minimum score of 1 run
matches.loc[matches['innings1_runs'].idxmin()]


# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.set()
sns.set(style="darkgrid")
ax = sns.countplot(x=matches['innings1_wickets'], data=matches)


# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.set()
sns.set(style="darkgrid")
ax = sns.countplot(x=matches['innings2_wickets'], data=matches)


# In[ ]:


plt.plot(matches['innings1_overs_batted'],matches['innings1_runs'])


# In[ ]:


plt.plot(matches['innings2_overs_batted'],matches['innings2_runs'])


# In[ ]:


names = ' '
for name in matches.venue:
    name = str(name)
    names = names + name + ' '
from wordcloud import WordCloud, STOPWORDS 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',  
                min_font_size = 10).generate(names) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[ ]:


#Top 10 Grounds
values = matches.venue.value_counts().sort_values(ascending=False).head(10)
labels = values.index
plt.figure(figsize=(15,8))
sns.barplot(x=values, y=labels)


# In[ ]:


#Match with max no of runs margin for victory
matches.loc[matches['win_by_runs'].idxmax()]


# In[ ]:


#Match with min no of runs margin for victory
matches.loc[matches['win_by_runs'].idxmin()]


# In[ ]:


series.head()


# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.set()
sns.set(style="darkgrid")
ax = sns.countplot(x=series['season'], data=series)


# In[ ]:


#Top 10 Series Winners
values = series.winner.value_counts().sort_values(ascending=False).head(10)
labels = values.index
plt.figure(figsize=(15,8))
sns.barplot(x=values, y=labels)


# In[ ]:


series['series'] = series.series.str.split('(',expand=True)


# In[ ]:


#Top 10 Series 
values = series.series.value_counts().sort_values(ascending=False).head(10)
labels = values.index
plt.figure(figsize=(15,8))
sns.barplot(x=values, y=labels)


# In[ ]:


x = series.margin.str.split('(',expand=True)[1]
x = x.str.split(')',expand=True)[0]


# In[ ]:


series['Margin'] = x


# In[ ]:


series.drop(['margin'],axis=1,inplace=True)


# In[ ]:


series.head()


# In[ ]:


#Series Win by max margin
series.info()


# In[ ]:


series = series.astype({"Margin": float})


# In[ ]:


series.loc[series['Margin'].idxmax()]


# In[ ]:


#All the series where winning Margin was more than or equal to 4 
series.loc[series['Margin']>=4]


# **Model will be made for winner prediction in next update**

# In[ ]:





# In[ ]:





# In[ ]:




