#!/usr/bin/env python
# coding: utf-8

# This notebook is an analysis of religious attacks using the dataset provided by Kaggle. This is inspired by the notebooks created by Keyshin: https://www.kaggle.com/keyshin/d/argolof/predicting-terrorism/religious-attacks-a-start and ArjonnSharma: https://www.kaggle.com/arjoonn/d/argolof/predicting-terrorism/scoutscript . I used some parts of the code from ScoutScript and added my own.

# In[ ]:


# import libraries
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import datetime
from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud, STOPWORDS
import random


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams


# Change the country here to your liking. I'm using my country for this notebook as an example

# In[ ]:


country = "Philippines"


# In[ ]:


# load dataset
df = pd.read_csv('../input/attacks_data_UTF8.csv',
                 encoding='latin1', parse_dates=['Date'],
                 infer_datetime_format=True,
                 index_col=1,
                )


# I created a new 'Victims' column that just add the counts of the affected people (Killed and Injured)

# In[ ]:


df['Victims'] = df['Killed'] + df['Injured']


# Here we create a new dataframe for the specific country specified. If not specified, use all the data instead.

# In[ ]:


if country is not None:
    dfc=df.loc[df['Country']==country]
else:
    dfc = df


# In[ ]:


# Just a quick count of the data

country_rank = df.Country.value_counts().rank(numeric_only=True,ascending=False).loc[country]
country_attacks = df.Country.value_counts()[country]
country_killed = dfc.Killed.sum()
country_injured = dfc.Injured.sum()
print("%s is ranked %.0f with %d attacks resulting to %d deaths and %d injuries" % (country, country_rank, country_attacks, country_killed, country_injured))


# From the code above, we can see the ranking of the country based on the number of attacks and a quick summary of the number of deaths and injured people. Our country is ranked 13 in number of attacks.

# Now we get the number of attacks by city for the country.

# In[ ]:


dfc.City.value_counts().plot(kind='bar', figsize=(17, 7))
plt.title('Number of attacks by city')


# The plot above shows that most of the attacks happened on the areas on the southern part of our country.

# Next we get the graphs of the attack on cities with the most victims. It also shows the graph of those Killed and Injured for comparison

# In[ ]:


dfc.groupby('City').sum()[['Victims','Killed', 'Injured']].sort_values(by='Victims',ascending=0).plot(kind='bar', figsize=(17, 7), subplots=True)


# The graphs above shows that the city with the most victims in total is not necessarily those with the most deaths. It can also be seen that the city with most death has a very low Injured people count. 

# The code below gets the attacks with the most victims, killed and injuries and the corresponding description.

# In[ ]:


# Attack with most victims
most_victim = dfc.sort_values(by='Victims',ascending=False).head(1)
# most_victim.index.strftime("%Y-%m-%d")
print("Attack with most victims happened on %s on %s with %d killed, %d injuries with a total of %d victims with the following article: \n'%s' \n" % (most_victim.City.values[0], most_victim.index.strftime("%B %d,%Y")[0], most_victim.Killed, most_victim.Injured, most_victim.Victims, "%s" % most_victim.Description.values[0]))
# Attack with most killed
most_killed = dfc.sort_values(by='Killed',ascending=False).head(1)
print("Attack with the most deaths happened on %s on %s with %d killed, %d injuries with a total of %d victims with the following article: \n'%s' \n" % (most_killed.City.values[0], most_killed.index.strftime("%B %d,%Y")[0], most_killed.Killed, most_killed.Injured, most_killed.Victims, "%s" % most_killed.Description.values[0]))
#Attack with most injuries
most_injuries = dfc.sort_values(by='Injured',ascending=False).head(1)
print("Attack with the most injuries happened on %s on %s with %d killed, %d injuries with a total of %d victims with the following article: \n'%s' \n" % (most_injuries.City.values[0], most_injuries.index.strftime("%B %d,%Y")[0], most_injuries.Killed, most_injuries.Injured, most_injuries.Victims, "%s" % most_injuries.Description.values[0]))


# The attack with most deaths came from a ferry explosion and explains why there are very few injured people compared to those killed.

# Next we get the plot of deaths and injuries by year

# In[ ]:


# Over the years
dfc.groupby(dfc.index.year).sum()[['Victims','Killed', 'Injured']].sort_values(by='Victims',ascending=0).plot(kind='bar', figsize=(17, 7), subplots=False)


# From 2003 to 2015, the number of victims generally goes down.

# Next we get if there are differences between the number of killed and injured by day or by month. I grouped them by weekday and month to see if there are patterns.

# In[ ]:


killedbyday = dfc.groupby([dfc.index.map(lambda x: x.weekday),dfc.index.year], sort=True).agg({'Killed': 'sum'})
rcParams['figure.figsize'] = 20, 10
killedbyday.unstack(level=0).plot(kind='bar', subplots=False)
killedbyday.unstack(level=1).plot(kind='bar', subplots=False)


# In[ ]:


# Check if there is a difference in attack victims by month
killedbymonth = dfc.groupby([dfc.index.map(lambda x: x.month),dfc.index.year], sort=True).agg({'Killed': 'sum'})
rcParams['figure.figsize'] = 20, 10
killedbymonth.unstack(level=0).plot(kind='bar', subplots=False)
killedbymonth.unstack(level=1).plot(kind='bar', subplots=False)


# Next we create a word cloud based on the attack descriptions. First we created a text variable that concatenates all the description words. Then we create a word cloud based on those words.

# In[ ]:


# Word cloud
text = dfc.Description.str.cat(sep=' ')
stopwords = set(STOPWORDS)


# In[ ]:


wc = WordCloud(background_color="white",max_words=100, stopwords=stopwords, margin=10,
               random_state=1).generate(text)


# The function below just makes the word cloud use random grayscale colors

# In[ ]:


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(20, 80)


# In[ ]:


default_colors = wc.to_array()
rcParams['figure.figsize'] = 10, 10
plt.title("Attack description word cloud")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
plt.axis("off")
plt.figure()


# The word cloud shows that the most frequent word is Abu, which came from one of the terrorist groups in the Philippines called the "Abu Sayyaf". An improvement for this word cloud is to create a corpus of the terrorist groups so that it can be grouped into one term / token as in "Abu Sayyaf" or "Moro Islamic Liberation Front". 

# I'll be modifying this notebook to show better visualization on the day and month (probably via heatmaps) like in KeyShin's notebook and get more analysis on the terrorist groups taken from the descriptions.
