#!/usr/bin/env python
# coding: utf-8

# Hi! I'm just messing around with different metrics. New to Python and Kaggle. I welcome any and all tips!

# # Average IMDB Score for Different Genres
# ## Setting Up The Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for graphing
import sys

# Grab the data
df = pd.read_csv('../input/movie_metadata.csv')
print(df.head(1))


# In[ ]:


# For now, let's just do a bar graph of category ratings
cat_df = df[['genres', 'imdb_score']]
print(cat_df.head(5))


# ## Calculating IMDB Score By Genre
# 
# Here, I am going through each genre and counting the IMDB score for that movie towards that genre. If a movie has genres "Action" and "Thriller" with a score of 8.5, both sums for "Action" and "Thriller" will receive 8.5.
# 
# Once I have the total sums for each genre and total counts, I can calculate the mean.

# In[ ]:


# Calculate category counts, sums, and means
cat_count = {}
cat_sum = {}
mean_score = 0
# calculate category counts and sums
for row in cat_df.itertuples():
    mean_score += getattr(row, "imdb_score")
    for cat in getattr(row, "genres").split("|"):
        if not cat in cat_count:
            cat_count[cat] = 1
            cat_sum[cat] = getattr(row, "imdb_score")
        else:
            cat_count[cat] += 1
            cat_sum[cat] += getattr(row, "imdb_score")

# print(cat_count)
# print(cat_sum)
mean_score /= len(cat_df)

cat_mean = {k: v/cat_count[k] for k, v in cat_sum.items()}
print(cat_mean)
# print("Mean score is: {}".format(mean_score))


# ## Plotting
# 
# Green / red = above / below the mean score.

# In[ ]:


# plot category mean imdb rating
colors = {k: 'green' if (v > mean_score) else 'red' for k, v in cat_mean.items()}
plt.figure(figsize=(10,7.5))
plt.title('Average IMDB Rating By Category')
plt.barh(range(len(cat_mean)), cat_mean.values(), align='center', color=colors.values())
plt.yticks(range(len(cat_mean)), list(cat_mean.keys()))
plt.plot([mean_score, mean_score], [-0.5, len(cat_mean) - 0.5], "k--")
plt.show()


# # Top Directors
# ## Calculating IMDB Score By Director

# In[ ]:


# Now we need the director and imdb score info
dir_df = df[['director_name', 'imdb_score']]
print(dir_df.head(5))


# In[ ]:


# Similar process to above
dir_count = {}
dir_score = {}
for row in dir_df.itertuples():
    score = getattr(row, "imdb_score")
    dir = getattr(row, "director_name")
    if dir not in dir_score:
        dir_count[dir] = 1
        dir_score[dir] = score
    else:
        dir_count[dir] += 1
        dir_score[dir] += score

dir_mean = {k: (dir_score[k] / dir_count[k]) for k in dir_score.keys()}
# This time: mean should be average score of a director's movies
mean = np.mean(np.array(list(dir_mean.values())))
print("The average director rating is: {}".format(mean))
# print(dir_mean)

from collections import Counter

# Let's say we only want to view the top 25
top_dirs_list = Counter(dir_mean).most_common(25)
top_dirs = dict(reversed(top_dirs_list))
print(top_dirs)


# In[ ]:


# plot top 25 directors with average imdb rating
colors = {k: 'green' if (v > mean) else 'red' for k, v in top_dirs.items()}
plt.figure(figsize=(10,7.5))
plt.title('Average IMDB Score For Top 25 Directors')
plt.barh(range(len(top_dirs)), top_dirs.values(), align='center', color=colors.values())
plt.yticks(range(len(top_dirs)), list(top_dirs.keys()))
for i, v in enumerate(top_dirs_list):
    plt.text(v[1], len(top_dirs) - i - 1.25, str(np.round(v[1], 2)), color='black', fontweight='bold')
plt.show()


# In[ ]:




