#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS: INTRODUCTION
# 
# In this notebook we will take a look at the datasets available in this repository. We will focus on the dataset 1950.csv, which contains:
# 
# * all the songs from the Spotify's playlist "All out 50s" that you can find [here](https://open.spotify.com/playlist/37i9dQZF1DWSV3Tk4GO2fq). This playlist collects the most popular and iconic songs from a given decade, in this case it collects the most popular songs from the 50s (1950-1959).
# * for each listed song, a series of attributes are collected in every column of the dataset, as explained in the description of the dataset.
# 
# The objective is to better understand the data, and study some relationships between the various attributes.

# # 1: LOAD PACKAGES AND DATA
# 
# Let's begin by importing the required packages and data. We will need `pandas`, `matplotlib` and `seaborn`.
# 
# Once the data is loaded in the dataframe `df`, we can take a quick look with `.info()` and `.describe()` to get a sense of the data we will be working with. We can also print the first five rows of `df` with `.head(5)`.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../input/spotify-past-decades-songs-50s10s/1950.csv')

# Show info and description of dataset
print(df.head(5))
df.info()
df.describe()


# We can see that `df` has a column `Number` which shows an interger ID for each listed song. This is quite redundant with the index of the dataframe created by `pandas`. What we can do is set the column `Number` as index of the dataframe. This is easily done with `.set_index()`.
# 
# We can also see that the column `year` contains some incoherent values: the release year should be no later than 1959, but we can see values exceeding this threshold. This is due to the fact that some songs have been re-issued/re-released (as explained in the description of the dataset). Since we won't be conducting any analysis based on the release year, we can drop this column with `.drop()`.

# In[ ]:


# Set column Number as index
df.set_index('Number', inplace=True)

# Drop year column
df.drop(['year'], axis=1, inplace=True)

# Show updated dataframe
df.head(5)


# # 2: HANDLE MISSING VALUES
# 
# From the `.info()` method run above, we saw that the column `top genre` has some missing values. This is confirmed when we run `isna().sum()`: there are a total of 12 missing values.

# In[ ]:


# Find missing values
df.isna().sum()


# Here we have two options:
# 1. we can fill the missing values manually, by searching the respective song online and finding the genre.
# 2. we can drop the row.
# 
# Since I'm too lazy to search for each song's genre, I will drop the rows where we don't have a genre value.

# In[ ]:


df.dropna(how='any', inplace=True)

# Take a look at the cleaned dataset
print(df.head(5))
df.info()
df.describe()


# # 3: GENRE ANALYSIS
# 
# First, let's focus on the analysis of the categorical variable `top genre`. We can show a pie chart of the distribution of the genre. The pie chart shows that more than 50% of the most popular songs in the 50s are *adult standards* songs (whatever that means, I guess it's pop music). Clearly, we don't have any electronic music, or rap/hip-hop for that matter.

# In[ ]:


# Find percent of each genre
df_genre = df['top genre'].value_counts() / len(df)
sizes = df_genre.values.tolist()
labels = df_genre.index.values.tolist()

# Pie chart for genre
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, textprops={'fontsize': 14})
ax1.axis('equal')
plt.show()


# # 4: BOXPLOT
# 
# Let's now take a look at the `boxplot` chart. This is an effective graphical method to study statistical description of the dataset in terms of:
# * min value
# * 25% quantile
# * 50% quantile (median value)
# * 75% quantile
# * max value
# 
# The boxplot does highlight the presence of some outliers. We could decide to filter them out, for instance:
# * we could remove the top 1% values in the case of `bpm`, `nrgy`, `live`, `dur`, `pop`
# * we could remove the bottom 1% values in the case of `acous`
# 
# However, it must also be considered that the outliers are not completely useless information, so we must be careful when modifying them.

# In[ ]:


# Plot boxplot (variables only)
sns.boxplot(data=df.drop(['title', 'artist', 'top genre'], axis=1))
plt.xlabel('Features')
plt.ylabel('Value')
plt.show()


# # 5: PAIRPLOT
# 
# We can now begin to study the relationships between the variables available in the dataset. For this purpose, we clearly don't need the column `title` and `artist`, so we can drop them. To study the relationships between variables I like to use `sns.pairplot()`, which gives us a quick overview of all relationships.

# In[ ]:


sns.pairplot(data=df.drop(['title', 'artist'], axis=1), hue='top genre')
plt.show()


# Looking at the resulting plot, we can see that a good number of variables are somewhat linearly releated, even if the datapoints are quite sparse. With that being said, it is true that we are considering all the available genres: to gain a more fine understaning of the relationships we should probably limit the scope of the analysis to 2-3 genres. We can limit the number of samples to the songs belonging to the top-3 genres, namely *adult standards, brill building pop, deep adult standards*.

# In[ ]:


# Top 3 most popular genres
genre_list = ['adult standards', 'brill building pop', 'deep adult standards']

# Extract sample from df
df_ = df.loc[df['top genre'].isin(genre_list)]

# Plot pairplot with limited data
sns.pairplot(data=df_.drop(['title', 'artist'], axis=1), hue='top genre')
plt.show()


# # 6: CORRELATION MATRIX
# 
# Finally, we can study the correlation matrix. This is one of the most commonly used tools to quickly understand how and how much the variables' behavior are related to each other. Furthermore, it allows us to better understand the observations/hypotheses we did when looking at the `pairplot`. 
# 
# The correlation matrix shown below confirms what we already noticed previoulsy: most of the variables are positively related, meaning that they show a similar trend/behavior. The only variables showing a different behavior are `dur` and `acous`. Looking at the correlation matrix, one could infer that:
# * the longer the song, the less is has energy, danceability, loudness, liveness. This is understandable in terms of enery and danceability, less so in terms of loudness.
# * the longer the song, the more it is popular. This is somewhat counter-intuitive, since most viral songs are short and catchy, at least nowadays. Almost certainly, music taste has changed from the 50s to today, and will change in the future as well.

# In[ ]:


# Plot linear correlation matrix
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
plt.title('LINEAR CORRELATION MATRIX')
plt.show()


# This wraps up this notebook. We took a good look at the attributes that define the most popular songs from the 50s. 
# 
# The steps showed can be applied to the other available datasets, with the intent of tracking how the human music taste changed from the 50s to today.
