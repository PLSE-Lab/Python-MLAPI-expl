#!/usr/bin/env python
# coding: utf-8

# # Things to add to the Project:
# 
# 1. Word Clouds
# 2. Top reviewers scores for albums
# 3. Top reviewers scores by genre & differences

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw_data = pd.read_csv("../input/pitchfork-reviews/reviews.csv")
raw_data.head()


# In[ ]:


# Drop columns we won't be needing
raw_data.drop(['url', 'pub_weekday', 'pub_day', 'pub_month',  'reviewid.1', 'Unnamed: 0'], axis = 1, inplace = True)
raw_data.head(3)


# In[ ]:


# Check the datatypes of the dataset
# raw_data['author'] = raw_data['author'].astype(str)
raw_data.dtypes


# In[ ]:


raw_data.shape


# In[ ]:


# Check for null values in the dataset
print(pd.isnull(raw_data).sum())

# fill in the null values for the genre column
raw_data['genre'].fillna("No Genre", inplace = True)


# In[ ]:


# Drop the rows with null values for artist column
print(np.where(pd.isna(raw_data['artist'])))
raw_data.drop([3388, 3440])
raw_data.shape


# In[ ]:


# Change the score number to a whole number ranging from 1 - 100
raw_data['score'] = raw_data['score'] * 10
raw_data['score'] = raw_data['score'].astype(int)
raw_data['score']


# In[ ]:


raw_data.head()


# In[ ]:


strings = raw_data['author'].value_counts()[:52].index # 52 authors over 100 reviews or more

# create a list containing the authors with >= 100 reviews
author_names = []
for x in strings:
    author_names.append(x)


# In[ ]:


raw_data['author'].value_counts(normalize = True)[:52].sum() # percentage of all reviews


# In[ ]:


(raw_data['author'].value_counts() >= 100).sum() # proves how many authors have over 100 reviews or more


# In[ ]:


author_less = []
strings = raw_data['author'].value_counts()[52:].index
for x in strings:
    author_less.append(x)
(author_less)


# In[ ]:


# get the indices of the authors with >= 100 reviews, needed for data cleaning
keep_indices = []
increment = -1
for i in raw_data['author']: # iterate through the author column of the dataframe
    increment += 1 # increase when the iteration for the row has been complete to check if the author is in the list
    for j in range(len(author_names)): # iterate through the names with >= 100 reviews
        if i == author_names[j]: # check if they are the same
           keep_indices.append(increment) # append the indice
print(keep_indices, end = ' ')


# In[ ]:


# keep_indices.remove(3388)
# keep_indices.remove(3440)


# In[ ]:


# compare if the amonut of reviews is the same as the authors total reviews with >= 100 
print(len(keep_indices))
print(raw_data['author'].value_counts()[:52].sum())


# In[ ]:


# get the indices we want to drop
remove_indices = []
increment = -1
for i in raw_data['author']:
    increment += 1
    for j in range(len(author_less)):
        if i == author_less[j]:
            remove_indices.append(increment)
print(remove_indices)


# In[ ]:


# compare if the amonut of reviews is the same as the authors total reviews with >= 100 
print(len(remove_indices))
print(raw_data['author'].value_counts()[52:].sum())


# In[ ]:


subdata = raw_data.drop(remove_indices)
subdata['author'].value_counts() # check if the <= 100 reviews have been removed


# In[ ]:


less100data = raw_data.drop(keep_indices)
less100data['author'].value_counts()


# In[ ]:


plt.figure(figsize = (10, 10))
subdata['author'].value_counts().plot(kind='barh')


# In[ ]:


subdata['genre'].value_counts().plot.pie(figsize = (10, 10), autopct = '%1.1f', fontsize = 15)


# In[ ]:


less100data['genre'].value_counts().plot.pie(figsize = (10, 10), autopct = '%1.1f', fontsize = 15)


# In[ ]:


subdata['score'].value_counts().plot.pie(figsize = (10,10), autopct = "%1.1f")


# In[ ]:


# Data clean to separate the genres for >= 100 reviews
subdata.columns


# In[ ]:


strings = set(subdata['genre'])

genres = []
for i in strings:
    genres.append(i)
genres


# In[ ]:


# Solution to separating the genres for authors with >= 100 reviews for Pitchfork
rap_data = subdata[subdata['genre'] == 'rap']
electronic_data = subdata[subdata['genre'] == 'electronic']
pop_and_rb_data = subdata[subdata['genre'] == 'pop/r&b']
experimental_data = subdata[subdata['genre'] == 'experimental']
metal_data = subdata[subdata['genre'] == 'metal']
global_data = subdata[subdata['genre'] == 'global']
rock_data = subdata[subdata['genre'] == 'rock']
folk_and_country_data = subdata[subdata['genre'] == 'folk/country']
jazz_data = subdata[subdata['genre'] == 'jazz']


# In[ ]:


# # Another Solution to separating the genres, but takes longer to do compared to the first solution
electronic = (subdata['genre'] == 'electronic')
electronic_data =  subdata[electronic]


# In[ ]:


electronic_data['score'].value_counts()


# ## Create new dataframe: Find the total score, total albums, average score, genre of all artists*

# In[ ]:


artist_data = pd.DataFrame(columns = ['artist', 'total score', 'albums count', 'genre', 'Average Score'])
artist_data


# In[ ]:


for i in raw_data.index:
    if raw_data.loc[i, 'artist'] in artist_data.values:
        artist_data.append(raw_data.loc[i, 'artist'])


# In[ ]:


# Grab the artist names first 
artist_names = []
for i in raw_data['artist'].index:
    name = raw_data.loc[i, 'artist']
    
    if name in artist_names:
        continue;
    else:
        artist_names.append(name)


# In[ ]:


# total_artists = len(artist_names) + counter 
# print(len(artist_names))
# print(total_artists)


# In[ ]:


raw_data['artist'].value_counts().sum()


# In[ ]:


artist_data['artist'] = artist_names


# In[ ]:


artist_data.shape


# In[ ]:


artist_data.head()


# In[ ]:


raw_data['genre'].value_counts()


# In[ ]:


artist_data.drop(2731, inplace = True)


# In[ ]:


scores = []
total_albums = []
genres = []

for i in artist_data['artist']:
    temp_data = raw_data[raw_data['artist'] == i]
    temp_score = temp_data['score'].sum()
    temp_album = len(temp_data)
    temp_genre = (temp_data['genre']).value_counts().index[0]
    
    scores.append(temp_score)
    total_albums.append(temp_album)
    genres.append(temp_genre)


# In[ ]:


artist_data['total score'] = scores


# In[ ]:


artist_data['albums count'] = total_albums


# In[ ]:


# Get the average score for each artist
artist_data['Average Score'] = artist_data['total score'] / artist_data['albums count']


# In[ ]:


artist_data['genre'] = genres


# In[ ]:


artist_data.head()


# ### Top 50 Best & Worst reviewed artists based on average score

# In[ ]:


best_artists = artist_data.sort_values(by = 'Average Score', ascending = False)[:50]


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(best_artists['artist'], best_artists['Average Score'])


# In[ ]:


worst_artists = artist_data.sort_values(by = 'Average Score', ascending = True)[:50]


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(worst_artists['artist'], worst_artists['Average Score'])


# ### Score differences between genres

# In[ ]:


artist_data.hist('Average Score', 'genre', figsize = (20, 10), bins = 10)


# ## Best New Music Scores

# In[ ]:


best_new_music = raw_data[raw_data['best_new_music'] == 1] # grab all the rows with a 1 that represents it got best new music by Pitchfork
best_new_music.head()


# In[ ]:


best_new_music['pub_year'].unique() # find out the years range for best new music


# In[ ]:


x = best_new_music['pub_year']
y = best_new_music['score']

plt.figure(figsize = (10, 10))
plt.plot(x, y, 'b.')
plt.xticks(np.arange(2003, 2017))


# ## Music Labels

# In[ ]:


raw_data['label'].value_counts()


# In[ ]:


label_data = pd.DataFrame(columns = ['label name', 'total score', 'total albums', 'average score', 'genres'])
label_data.head()


# In[ ]:


# Grab the label names first 
label_names = []
for i in raw_data['label'].index:
    name = raw_data.loc[i, 'label']
    
    if name in label_names:
        continue;
    else:
        label_names.append(name)


# In[ ]:


label_data['label name'] = label_names
label_data.head()


# In[ ]:


scores = []
total_albums = []
for i in label_data['label name']:
    temp_data = raw_data[raw_data['label'] == i]
    temp_score = temp_data['score'].sum()
    temp_album = len(temp_data)
    
    scores.append(temp_score)
    total_albums.append(temp_album)


# In[ ]:


label_data['total score'] = scores
label_data['total albums'] = total_albums
label_data['average score'] = label_data['total score'] / label_data['total albums']


# ##### Top 20 labels based on having at least 10 albums reviewed and average score
# 

# In[ ]:


top20_labels = label_data[label_data['total albums'] >= 10]    .sort_values(by = 'average score', ascending = False)[:20]


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(top20_labels['label name'], top20_labels['average score'])

