#!/usr/bin/env python
# coding: utf-8

# # What Influences the Rating of Animes & More

# <img src="https://i.imgur.com/9zk8gfO.png" alt="animes girl" align = "left">

# <strong>Anime</strong> is an art form, specifically animation, that includes all genres found in cinema, but it can be mistakenly classified as a genre. In Japanese, the term anime is used as a blanket term to refer to all forms of animation from around the world. In English, anime  is more restrictively used to denote a "Japanese-style animated film or television entertainment" or as "a style of animation created in Japan". (<a href="https://en.wikipedia.org/wiki/Anime">Source</a>)

# # Table of contents
# 
# [<h3>1. Content of the dataset & cleaning</h3>](#1)
# 
# [<h3>2. How many genres do animes have?</h3>](#2)
# 
# [<h3>3. Which genres have the best ratings?</h3>](#3)
# 
# [<h3>4. Animes with more votes also have better ratings?</h3>](#4)
# 
# [<h3>5. Number of episodes depending on the genre</h3>](#5)
# 
# [<h3>6. Most popular animes by number of votes</h3>](#6)

# ## 1. Content of the dataset & cleaning<a class="anchor" id="1"></a>
# 

# This dataset represents all animes crawled from Crunchyroll in this repository, containing its rating, number of votes and genre.

# - <strong>anime</strong> - English name of the anime<br/><br/>
# - <strong>anime_url</strong> - anime URL on Crunchyroll<br/><br/>
# - <strong>anime_img</strong> - anime image URL hosted by Crunchyroll<br/><br/>
# - <strong>episodes</strong> - number of episodes hoster by Crunchyroll of the anime<br/><br/>
# - <strong>votes</strong> - number of votes of the anime<br/><br/>
# - <strong>weight</strong>- the sum of rated stars received of the anime<br/><br/>
# - <strong>rate</strong> - an average rating out of 5 stars of the anime<br/><br/>
# - <strong>rate_1</strong> - the quantity of 1 stars votes of the anime<br/><br/>
# - <strong>rate_2</strong> - the quantity of 2 stars votes of the anime<br/><br/>
# - <strong>rate_3</strong> - the quantity of 3 stars votes of the anime<br/><br/>
# - <strong>rate_4</strong> - the quantity of 4 stars votes of the anime<br/><br/>
# - <strong>rate_5</strong> - the quantity of 5 stars votes of the anime<br/><br/>
# - <strong>genres</strong> - 29 different genres of the anime, one column for each genre with 1 and 0<br/><br/>
# 

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv('../input/crunchyroll-anime-ratings/animes.csv')


# In[ ]:


df.describe()


# Some of the animes have 0 episode, this is probably an error.
# 
# The genres are binary, whether an anime is part of the genre (1) or isn't (0). In the dataframe, the genres are of the type float (0.0 or 1.0), therefore we will change their type to integer.

# In[ ]:


# Change the type of the columns with the genre to integer
for c in df.columns[12:]:
    df[c] = df[c].astype("int")
    
# Delete "genre_" from the columns to have clean genres
idx = []
for i in df.columns:
    idx.append(i.replace("genre_",""))
df.columns = idx


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull())
plt.title("Missing values?", fontsize = 18)
plt.show()


# There is no missing value in the dataset.

# # 2. How many genres do animes have?<a class="anchor" id="2"></a>

# It would be difficult to categorize each anime in only one genre. Therefore, in the dataset each anime can have from 0 to 9 different categories. It also means that some animes don't have any genre.

# In[ ]:


# Get the number of animes with how many genres
number_genre = df[df.columns[12:]].sum(axis = 1).value_counts().sort_index()

# Create a color list depending on the number the genres
colors = []
maxg = max(number_genre)
for n in number_genre:
    x = 0.8 - n/(2*maxg)
    colors.append((0.7, x, x))
average = number_genre.mean()

# Plot the result
plt.figure(figsize=(10,6))
number_genre.plot.bar(color = colors)
plt.title("Repartition of the number of genres", fontsize = 18)
plt.axhline(average, 0 ,1, color = "black", lw = 3)
plt.text(9.6, 120, "average", fontsize = 14)
plt.ylabel("Animes count", fontsize = 14)
plt.xlabel("\nNumber of genres", fontsize = 14)
plt.show()


# Most of the animes have between 2 and 4 genres.

# # 3. Which genres have the best ratings? <a class="anchor" id="3"></a>

# As we have seen before, one anime can have more than one genre, therefore it wouldn't be accurate to calculate the mean rating for each genre directly, otherwise animes having more than one genre would count more. 
# 
# In the following, the weighted categories will be calculated. The more genres an anime has, the less each one will count. See the example of transformation below.

# In[ ]:


def transf(x):
# Some animes don't have any category, 
# to avoid diving by 0, categories of animes with
# no categories will be divided by 1 instead
    if x == 0:
        return 1
    else:
        return x

def weight_df(df, col_start = 12):
# Transform the genres into weighted genres
    
    fact = df[df.columns[col_start:]].sum(axis = 1).apply(lambda x:transf(x))
    df_va = df.values
    for m in range(len(df_va)):
        df_va[m]
        for i in range(col_start, len(df_va[m])):
            df_va[m][i] = df_va[m][i] / fact[m]
    return pd.DataFrame(df_va, columns=df.columns)
    

lst = [["anime 1", 1,1,0,1,1,0],["anime 2", 0,0,0,0,0,1],["anime 3", 1,0,1,1,0,0]]
cols = ["Anime", "category_1", "category_2", "category_3", "category_4", "category_5", "category_6"]

# Without transformation
example = pd.DataFrame(lst, columns = cols)
example


# In[ ]:


# After transformation
weight_df(example, col_start = 1)


# In[ ]:


# Get the weighted categories
df_weighted = weight_df(df)

# Get the number of animes with no genre
nb_0_genre = (df[df.columns[12:]].sum(axis = 1) == 0).sum()

# Calculate the cantity of each category without "no genre"
weighted_betw = df_weighted[df_weighted.columns[12:]].sum()

# Add "no genre"
weighted_betw["NO genre"] = nb_0_genre

# Compute the percentage of each genre
distrib_genre = 100 * weighted_betw/weighted_betw.sum()

# Sort the values
distrib_genre = distrib_genre.sort_values(ascending = False)


# In[ ]:


# Display the results
plt.figure(figsize =(15,10))
bar = sns.barplot(distrib_genre.index, distrib_genre)
plt.title("Distribution of genres", fontsize = 18)
plt.ylabel("%", fontsize = 18)
bar.tick_params(labelsize=16)

# Rotate the x-labels
for item in bar.get_xticklabels():
    item.set_rotation(90)


# In[ ]:


mean_ratings = []
for g in df_weighted.columns[12:]:
    rating = ((df_weighted["rate"] * df_weighted[g]).sum()) / df_weighted[g].sum()
    mean_ratings.append([g, rating])

mean_ratings = pd.DataFrame(mean_ratings, columns = ["Genre", "Rating"]).sort_values(by = "Rating", ascending = False)

# Display the results
plt.figure(figsize =(15,10))
bar = sns.barplot("Genre", "Rating", data = mean_ratings, palette = "coolwarm")
plt.title("Mean Rating for each Genre", fontsize = 18)
plt.ylabel("Mean Rating", fontsize = 18)
plt.xlabel("")
bar.tick_params(labelsize=16)

# Rotate the x-labels
for item in bar.get_xticklabels():
    item.set_rotation(90)


# # 4. Animes with more votes also have better ratings?<a class="anchor" id="4"></a>

# In[ ]:


# Categorize the number of votes for each anime in 6 bins
def create_bins(v):
    if v > 10000:
        return ">10000"
    elif v > 2000:
        return "2000-10000"
    elif v > 500:
        return "500-2000"
    elif v > 100:
        return "100-500"
    elif v >= 10:
        return "10-100"
    else:
        return "<10"

df["votes_cat"] = df["votes"].apply(create_bins)


# In[ ]:


plt.figure(figsize=(10,7))
bar = sns.countplot(df["votes_cat"])
plt.ylabel("Count of animes", fontsize = 14)
plt.xlabel("\nNumber of votes", fontsize = 14)
plt.title("Number of animes by number of votes", fontsize = 18)
bar.tick_params(labelsize=14)
plt.show()


# In[ ]:


rate_votes_cat = pd.pivot_table(df, values = "rate", index = "votes_cat").sort_values(by = "rate", ascending = False)

bar = rate_votes_cat.plot.bar(figsize=(10,7), color = "grey")
plt.title("Mean Rating by number of votes", fontsize = 18)
plt.ylabel("Rating", fontsize = 14)
plt.xlabel("\nNumber of votes", fontsize = 14)
bar.tick_params(labelsize=12)
plt.show()


# The more votes, the better the mean rating. This is logical, because better animes are watched more.

# # 5. Number of episodes depending on the genre<a class="anchor" id="5"></a>

# In[ ]:


# Calculate the number of episodes depending on the genre
mean_episodes = []
for g in df_weighted.columns[12:]:
    episodes = ((df_weighted["episodes"] * df_weighted[g]).sum()) / df_weighted[g].sum()
    mean_episodes.append([g, episodes])
mean_episodes = pd.DataFrame(mean_episodes, columns = ["Genre", "episodes"]).sort_values(by = "episodes", ascending = False)

# Display the results
plt.figure(figsize =(15,10))
bar = sns.barplot("Genre", "episodes", data = mean_episodes, palette = "coolwarm")
plt.title("Mean number of episodes for each genre", fontsize = 18)
plt.ylabel("Mean number of episodes", fontsize = 12)
plt.xlabel("")
bar.tick_params(labelsize=16)

# Rotate the x-labels
for item in bar.get_xticklabels():
    item.set_rotation(90)


# Some genres don't have a mean because more than 1/3 of the animes in the dataset have 0 episode as we can see in the following:

# In[ ]:


nb_0_epi = df[df["episodes"] <1].shape[0]
print(f"Number of animes with 0 episode: {nb_0_epi}")
print(f"Total number of animes in the dataset: {df.shape[0]}")


# # 6. Most popular animes by number of votes<a class="anchor" id="6"></a>

# In[ ]:


bar = df[df["votes_cat"] == ">10000"].sort_values("rate",ascending = False)[:10].plot.barh("anime", "rate", figsize = (10,10), color = sorted(colors, reverse = True))
bar.tick_params(labelsize=16)
plt.title("Best rated animes with > 10.000 votes", fontsize = 18)
plt.xlabel("Mean rating", fontsize = 14)
plt.ylabel("")
plt.show()  


# In[ ]:


bar = df[df["votes_cat"] == "2000-10000"].sort_values("rate",ascending = False)[:10].plot.barh("anime", "rate", figsize = (10,10), color = sorted(colors, reverse = True))
bar.tick_params(labelsize=16)
plt.title("Best rated animes with 2.000 - 10.000 votes", fontsize = 18)
plt.xlabel("Mean rating", fontsize = 14)
plt.ylabel("")
plt.show()  


# In[ ]:


bar = df[df["votes_cat"] == "500-2000"].sort_values("rate",ascending = False)[:10].plot.barh("anime", "rate", figsize = (10,10), color = sorted(colors, reverse = True))
bar.tick_params(labelsize=16)
plt.title("Best rated animes with 500 - 2.000 votes", fontsize = 18)
plt.xlabel("Mean rating", fontsize = 14)
plt.ylabel("")
plt.show()  


# # Thank you

# <img src="https://i.imgur.com/9TCWLiA.png" align = "left">
