#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Understanding the data and finding varaibles that make a song popular

# Loading the csv file as a DataFrame named songs and displaying the column names.

# In[ ]:


songs = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1', index_col=0)
songs.columns.rename('column_name').to_frame().reset_index(drop=True)


# Renaming columns, resetting index and displaying first 5 rows.

# In[ ]:


songs = songs.rename(columns={'top genre':'genre', 'nrgy':'energy', 'dnce':'dance', 'dB':'db', 'dur':'length', 'acous':'acoustic', 'spch':'speech'}).reset_index(drop=True)
songs.index += 1
songs.head()


# ```
# songs.head() outputs the first 5 rows in the songs dataframe
# The default number is 5 but the parentehsis can take an integer value and display those many rows
# For example, songs.head(10) outputs the first 5 rows in the songs dataframe
# ```

# Displaying no of rows and columns

# In[ ]:


songs.shape


# ```
# songs.shape outputs the numbers of rows and columns in the songs dataframe, in this case it is 603 rows and 14 columns
# songs.shape[0] will output the number of rows only and songs.shape[1] will output the number of columns only
# ```

# Top 10 most popular songs in the last decade

# In[ ]:


songs[['pop','title','artist','year']].sort_values(by='pop', ascending=False).iloc[:10].set_index('pop')


# ```
# sort_values(by='pop', ascending=False), sorts the output in descending order of the pop column
# The ascending=False means descending and if left blank, the default is True which means ascending order
# ```

# ```
# iloc[:10] displays all the rows until the 9th
# The brackets takes a starting(inclusive) and ending(exclusive) value which when left blank will take the value of 0 or the last row
# iloc[10:] displays all the rows from row 10 till the last row in the dataframe
# iloc[:] displays all rows in the dataframe
# ```

# Top 10 artists with the most number of popular songs in the last decade

# In[ ]:


songs.groupby('artist').size().sort_values(ascending=False).rename('no_of_songs').head(10).to_frame()


# ```
# songs.groupby('artist').size() groups the dataframe by the artist column and outputs the number of rows of that artist in the dataframe
# In this case, the output is the number of hit songs in the past decade by the artist
# ```

# List of songs by the artist with the most number of popular songs in the last decade

# In[ ]:


temp = songs.groupby('artist').size().idxmax()
songs[songs.artist == temp].set_index('pop').sort_values(by='pop',ascending=False)


# ```
# idxmax() gives the index of the highest size
# ```

# Average details of songs grouped by popularity

# In[ ]:


songs[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech', 'pop']].groupby('pop').mean().iloc[-5:].sort_values(by='pop', ascending=False)


# ```
# songs[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech', 'pop']].groupby('pop').mean()
# groups the data by the popularity column and gives the mean of the details of all songs with that popularity
# ```

# Top 5 genre's by popularity

# In[ ]:


songs.groupby('genre').pop.count().rename('number').sort_values(ascending=False).head(5).to_frame().plot.pie(subplots=True)


# ```
# songs.groupby('genre').pop.count() groups the data by genre  and gives the number of songs under that genre
# ```

# ```
# plot.pie() plots a pie chart of the output data
# ```

# Details of the top 10 most popular songs in the most popular genre

# In[ ]:


songs[songs.genre == 'dance pop'].sort_values(by='pop', ascending = False).set_index('pop').head(10)


# Average key variable values of top 100 popular songs over the past decade by year

# In[ ]:


songs.set_index('pop').sort_values(by='pop',ascending=False).iloc[:100].groupby('year')[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech']].mean().round(2).sort_values(by='year',ascending=False)


# # Conclusion

# ### The analysis tells us that a song in the dance pop genre with the following details is most likely to get popular

# In[ ]:


songs[songs.genre == 'dance pop'].sort_values(by=['pop'],ascending=False)[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech']].head(100).mean().rename('mean').to_frame().plot.barh()


# ```
# plot.barh() plots a horizontal bar graph of the output data
# ```
