#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
plt.style.use("ggplot")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Read in the data

# In[ ]:


songs = pd.read_csv("../input/top50.csv", encoding="latin")


# # 2. Visualize the data

# In[ ]:


print(songs.info())
print(songs.head())


# # 3. What is the mean of minutes that a top music has?

# In[ ]:


mean_seconds = songs["Length."].mean()
mean_minutes = round(mean_seconds / 60, 2)
print(mean_minutes)


# # 4. Which singer has the most songs?

# In[ ]:


singers = songs["Artist.Name"].value_counts()
singers.plot.bar()
plt.xlabel("Singers")
plt.ylabel("# of Songs")
plt.title("The # of songs each singer has")
plt.show()


# # 5. Which is the most common genre?

# In[ ]:


genres = songs["Genre"].value_counts()
genres.plot.bar()
plt.xlabel("Genres")
plt.ylabel("# of Songs")
plt.title("The # of songs each genre has")
plt.show()


# # 6. Which singer has the most popularity?

# In[ ]:


singer_popularity = (
    songs.groupby("Artist.Name")["Popularity"].sum().sort_values(ascending=False)
)
singer_popularity.plot.bar()
plt.xlabel("Singers")
plt.ylabel("Total popularity")
plt.title("Total popularity each singer has")
plt.show()


# # 7. Which genre has the most popularity?

# In[ ]:


genre_popularity = (
    songs.groupby("Genre")["Popularity"].sum().sort_values(ascending=False)
)
genre_popularity.plot.bar()
plt.xlabel("Genres")
plt.ylabel("Total popularity")
plt.title("Total popularity each genre has")
plt.show()


# # 8. Are there any relationship between danceability and popularity?

# In[ ]:


plt.scatter("Danceability", "Popularity", data=songs.sort_values(by=["Danceability"]))
plt.title("The relationship between danceability and popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.show()


# # 9. Are there any relationship between dB and popularity?

# In[ ]:


plt.scatter(
    "Loudness..dB..", "Popularity", data=songs.sort_values(by=["Loudness..dB.."])
)
plt.title("The relationship between dB and popularity")
plt.xlabel("dB")
plt.ylabel("Popularity")
plt.show()


# # 10. Are there any relationship between liveness and popularity?

# In[ ]:


plt.scatter("Liveness", "Popularity", data=songs.sort_values(by=["Liveness"]))
plt.title("The relationship between liveness and popularity")
plt.xlabel("Liveness")
plt.ylabel("Popularity")
plt.show()


# # 11. Conclusion
# 1. The average length of those 50 songs is 3.35min. 
# 2. Ed Sheeran has most songs on this list. He has 4.
# 3. The most popular genre is Dance Pop -- 8 songs belong to it. And the following is pop -- 7 songs belong to it.
# 4. Ed Sheeran also has the most popularity and he has a huge advantage.
# 5. Dance Pop has the most popularity.
# 6. The majority of those songs have danceability larger than 50. However, when the danceability is larger than 50, there's no evidence could show that the higher the danceability, the more the popularity.
# 7. The majority of those songs have dB between -8 and -4.
# 8. The majority of those songs have liveness between 0 and 20.
