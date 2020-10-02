#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from gensim import corpora, models

from collections import Counter


# In[ ]:


data = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv')


# # Data overview

# In[ ]:


data.head()


# # Crashes plots

# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].map(lambda x: x.day)
data['Year'] = data['Date'].map(lambda x: x.year)
data['Month'] = data['Date'].map(lambda x: x.month)


# In[ ]:


crashes_per_year = Counter(data['Year'])
years = list(crashes_per_year.keys())
crashes_year = list(crashes_per_year.values())


# In[ ]:


crashes_per_day = Counter(data['Day'])
days = list(crashes_per_day.keys())
crashes_day = list(crashes_per_day.values())


# In[ ]:


def get_season(month):
    if month >= 3 and month <= 5:
        return 'spring'
    elif month >= 6 and month <= 8:
        return 'summer'
    elif month >= 9 and month <= 11:
        return 'autumn'
    else:
        return 'winter'

data['Season'] = data['Month'].apply(get_season)


# In[ ]:


crashes_per_season = Counter(data['Season'])
seasons = list(crashes_per_season.keys())
crashes_season = list(crashes_per_season.values())


# In[ ]:


sns.set(style="whitegrid")
sns.set_color_codes("pastel")

fig = plt.figure(figsize=(14, 10))

sub1 = fig.add_subplot(211)
sns.barplot(x=years, y=crashes_year, color='g', ax=sub1)
sub1.set(ylabel="Crashes", xlabel="Year", title="Plane crashes per year")
plt.setp(sub1.patches, linewidth=0)
plt.setp(sub1.get_xticklabels(), rotation=70, fontsize=9)

sub2 = fig.add_subplot(223)
sns.barplot(x=days, y=crashes_day, color='r', ax=sub2)
sub2.set(ylabel="Crashes", xlabel="Day", title="Plane crashes per day")

sub3 = fig.add_subplot(224)
sns.barplot(x=seasons, y=crashes_season, color='b', ax=sub3)
texts = sub3.set(ylabel="Crashes", xlabel="Season", title="Plane crashes per season")

plt.tight_layout(w_pad=4, h_pad=3)


# The first graph shows that in the heydey of the civil aviation (approximately 1940-1970 years) number of crashes is increasing. Since 1970 by now, the number of accidents slowly begins to decrease. Is related to the improvement of the quality of civil planes, technology and skill of the pilots. Flying is getting safer.

# The two lower graphics look pretty uniform (contrary to my expectations). That means there isn't some dependence between a number of crashes and days or seasons.

# # Survived and dead plots

# In[ ]:


survived = []
dead = []
for year in years:
    curr_data = data[data['Year'] == year]
    survived.append(curr_data['Aboard'].sum() - curr_data['Fatalities'].sum())
    dead.append(curr_data['Fatalities'].sum())


# In[ ]:


f, axes = plt.subplots(2, 1, figsize=(14, 10))

sns.barplot(x=years, y=survived, color='b', ax=axes[0])
axes[0].set(ylabel="Survived", xlabel="Year", title="Survived per year")
plt.setp(axes[0].patches, linewidth=0)
plt.setp(axes[0].get_xticklabels(), rotation=70, fontsize=9)

sns.barplot(x=years, y=dead, color='r', ax=axes[1])
axes[1].set(ylabel="Fatalities", xlabel="Year", title="Dead per year")
plt.setp(axes[1].patches, linewidth=0)
plt.setp(axes[1].get_xticklabels(), rotation=70, fontsize=9)

plt.tight_layout(w_pad=4, h_pad=3)


# # The worst operators

# In[ ]:


oper_list = Counter(data['Operator']).most_common(12)
operators = []
crashes = []
for tpl in oper_list:
    if 'Military' not in tpl[0]:
        operators.append(tpl[0])
        crashes.append(tpl[1])
print('Top 10 the worst operators')
pd.DataFrame({'Count of crashes' : crashes}, index=operators)


# # The most dangerous locations

# In[ ]:


loc_list = Counter(data['Location'].dropna()).most_common(15)
locs = []
crashes = []
for loc in loc_list:
    locs.append(loc[0])
    crashes.append(loc[1])
print('Top 15 the most dangerous locations')
pd.DataFrame({'Crashes in this location' : crashes}, index=locs)


# # Exploring the causes of crashes

# Let's analyse a "Summary" column and highlight the main themes that occur in the texts in this column. I will use gensim topic modelling library for this.

# In[ ]:


summary = data['Summary'].tolist()
punctuation = ['.', ',', ':']
texts = []

for text in summary:
    cleaned_text = str(text).lower()   
    for mark in punctuation:
        cleaned_text = cleaned_text.replace(mark, '')       
    texts.append(cleaned_text.split())


# In[ ]:


dictionary = corpora.Dictionary(texts)


# In[ ]:


word_list = []
for key, value in dictionary.dfs.items():
    if value > 100:
        word_list.append(key)


# In[ ]:


dictionary.filter_tokens(word_list)
corpus = [dictionary.doc2bow(text) for text in texts]


# In[ ]:


np.random.seed(76543)
lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=5)


# In[ ]:


topics = lda.show_topics(num_topics=10, num_words=15, formatted=False)
for topic in topics:
    num = int(topic[0]) + 1
    print('Cause %d:' % num, end=' ')
    print(', '.join([pair[0] for pair in topic[1]]))


# While I was analysing the words in each topic, I have identified the following causes (they may not reflect the actual reason for the crash):

# Cause 1: Spatial disorientation due to bad weather conditions.  
# Cause 2: Stalled the engine. The explosion or destruction of the aircraft from falling to the ground or collision with the building.  
# Cause 3: Failure of the rotor or problems with the fuselage (specifically problems with the tail). It is also a possible mistake of the Air Traffic Control centre.  
# Cause 4: Bad weather conditions: strong wind, snow, ice. The plane disappeared from radar.  
# Cause 5: Taking off without clearance from ATC. ATC or pilots error.  
# Cause 6: Crash due to manoeuvring. Most likely refers to the testing and training missions.  
# Cause 7: The plane was hijacked or captured by the rebels. Fell to the ground due to issues with piloting or bad weather conditions.  
# Cause 8: The plane was destroyed by explosion and destruction of the fuselage. The cause of the explosion could be a bomb or fuel tank.  
# Cause 9: The malfunction of the autopilot and remote control systems. Most likely associated with transport aircraft.  
# Cause 10: Navigation problems, technical malfunction.
