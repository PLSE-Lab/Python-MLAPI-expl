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


# In[ ]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud

# display settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# # Reading and understanding data

# In[ ]:


# initializing dataframe
df = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")


# In[ ]:


# dataframe shape
df.shape


# In[ ]:


# column info
list(df.columns)


# ### Columns info:
# **URL** The URL
# <br>**ID** The assigned ID
# <br>**Name** The name
# <br>**Subtitle** The secondary text under the name
# <br>**Icon URL** 512px x 512px jpg
# <br>**Average User Rating** Rounded to nearest .5, requires at least 5 ratings
# <br>**User Rating Count** Number of ratings internationally, null means it is below 5
# <br>**Price** Price in USD
# <br>**In-app Purchases** Prices of available in-app purchases
# <br>**Description** App description
# <br>**Developer** App developer
# <br>**Age Rating** Either 4+, 9+, 12+ or 17+
# <br>**Languages** ISO2A language codes
# <br>**Size** Size of the app in bytes
# <br>**Primary Genre** Main genre
# <br>**Genres** Genres of the app
# <br>**Original Release Date** When it was released
# <br>**Current Version Release Date** When it was last updated

# In[ ]:


# columns info
df.info()


# In[ ]:


# printing the first 5 rows
df.head()


# # Data pre-processing and cleaning

# ### Dropping the unnecessary columns

# In[ ]:


# dropping the columns URL, ID, Icon URL
df.drop(columns=["URL","ID","Icon URL"], inplace=True)


# ### Column-wise missing values 

# In[ ]:


# finding the no. of missing values in each column
df.isna().sum()


# ### Subtitle

# In[ ]:


# taking an initial look at the values and scanning for junk values
df["Subtitle"].dropna().head()


# Looks like there are no junk values.

# ### Average User Rating

# In[ ]:


# no. of rows in each rating bracket
df["Average User Rating"].value_counts()


# In[ ]:


# summary statistics
df["Average User Rating"].describe()


# In[ ]:


# plotting the boxplot to understand the outliers
sns.boxplot(df["Average User Rating"])
plt.show()


# Looks like there are no (negligeble) outliers or junk values.

# ### User Rating Count

# In[ ]:


# taking an initial look at the values and scanning for junk values
df["User Rating Count"].dropna().sort_values().head()


# In[ ]:


# summary statistics
df["User Rating Count"].describe()


# Looks like there are few outlier values at the higher end. Let's plot boxplot to find it out.

# In[ ]:


# boxplot and distribution plot
fig, ax = plt.subplots(1,2, figsize=(15,6))
sns.distplot(df["User Rating Count"].dropna(), ax=ax[0])
sns.boxplot(df["User Rating Count"].dropna(), ax=ax[1])
plt.show()


# Since these are user rating count values, we do not clean the outliers as they have no or little impact. 

# ### Price

# In[ ]:


# checking the different price brackets
df["Price"].value_counts().sort_index()


# In[ ]:


# summary statistics
df["Price"].describe()


# Looks like we can categorize the price into price range brackets.

# ### In-app Purchaces

# In[ ]:


# splitting the string into float values and storing them as a list
df["In-app Purchases"] = df["In-app Purchases"].dropna().map(lambda x: list(float(i) for i in x.split(", ")))


# ### Languages

# In[ ]:


# splitting the string and storing the values as a list
df["Languages"] = df["Languages"].dropna().map(lambda x: x.split(", "))


# ### Size

# In[ ]:


# filling the missing value
df["Size"].fillna(method="ffill", inplace=True)


# In[ ]:


# converting size in bytes to mega-bytes
df["Size"] = df["Size"].map(lambda x: round(x/(1024 * 1024), 2))
df["Size"].head()


# ### Genres

# In[ ]:


# splitting the string and storing the values as a list
df["Genres"] = df["Genres"].map(lambda x: x.split(", "))
df["Genres"].head()


# ### Original Release Date

# In[ ]:


# converting string to date
df["Original Release Date"] = df["Original Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))


# ### Current Version Release Date

# In[ ]:


# converting string to date
df["Current Version Release Date"] = df["Current Version Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))


# # Feature engineering

# ## Pricing bracket
# Creating four pricing brackets based on the Price variable.
# 
# **Free** 0.00
# <br>**Low Price** 0.99 - 4.99
# <br>**Medium Price** 5.99 - 19.99
# <br>**High Price** > 19.99

# In[ ]:


# categorizing price
df["Price Range"] = df["Price"].dropna().map(lambda x: "Free" if x == 0.00 else("Low Price" if 0.99 <= x <= 4.99 else("Medium Price" if 5.99 <= x <= 19.99 else "High Price")))
df["Price Range"].value_counts()


# ## Total In-app Purchases
# 
# Creating a new variable to show the sum of the In-app Purchases values.

# In[ ]:


df["Total In-app Purchases"] = df["In-app Purchases"].dropna().map(lambda x: sum(x))
df["Total In-app Purchases"].dropna().value_counts().head()


# ## Game Genre
# Creating a new variable to capture the genre of each game.

# In[ ]:


df["Game Genre"] = df[df["Primary Genre"] == "Games"]["Genres"].map(lambda x: x[1])
df["Game Genre"].head()


# ### Release Year and Month

# In[ ]:


df["Release Year"] = df["Original Release Date"].map(lambda x: x.strftime("%Y"))
df["Release Month"] = df["Original Release Date"].map(lambda x: x.strftime("%m"))


# # Exploratory data analysis

# In[ ]:


df.info()


# ## Which Primary Genre has most no. of apps?
# * Clearly, Games genre is having unparalleled edge over the other genres.

# In[ ]:


top_genres = list(df["Primary Genre"].value_counts().head(10).index)


# In[ ]:


df[df["Primary Genre"].isin(top_genres)]["Primary Genre"].value_counts().plot.bar(figsize=(8,5))
plt.title("Bar plot of primary genre wise apps")
plt.show()


# ## What are the most popular genres?
# * Games, Strategy and Entertainment are the three most popular genres.
# * Strategy, Battle and Puzzle are the most frequently used words in subtitles.

# In[ ]:


def word_cloud(list_variable):
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    for i, variable in enumerate(list_variable):
        corpus = df[variable].dropna()
        if variable not in ("Genres"):
            corpus = corpus.map(lambda x: x.replace(",", "").split(" "))
            corpus = corpus.map(lambda x: [word for word in x if len(word) > 3])
        corpus = ",".join(word for word_list in corpus for word in word_list)
        wordcloud = WordCloud(max_font_size=None, background_color="white", collocations=False, width=1500, height=1500).generate(corpus)
        ax[i].imshow(wordcloud)
        ax[i].set_title(variable)
        ax[i].axis("off")
    plt.show()

word_cloud(["Genres", "Subtitle", "Description"])


# ## What is the highest and lowest rated genre?
# * Sports genre is having the highest average user rating.
# * Business genre is having the lowest average user rating.

# In[ ]:


df[df["Primary Genre"].isin(top_genres)].groupby("Primary Genre")["Average User Rating"].agg("mean").sort_values().plot.bar(figsize=(8,6))
plt.title("Primary Genre wise Average User Rating")
plt.show()


# ## Genre wise age rating proportions.
# * In all the genres, 4+ age rating is having the higest proportion of apps.
# * Productivity genre is having all of its apps with 4+ age rating.
# * Sports genre is having almost 50% of the apps under 12+ and 17+ rating. 

# In[ ]:


ct_genre_agerating = pd.crosstab(df[df["Primary Genre"].isin(top_genres)]["Primary Genre"], df["Age Rating"], normalize=0)
ct_genre_agerating.plot.bar(stacked=True, figsize=(8,5))
plt.title("Primary Genre repartition by Age Rating")
plt.show()


# ## Which age rating has the highest and least proportion of apps?
# * Age rating 4+ has the highest proportion of apps.
# * Age rating 17+ has the least proportion of apps.
# * As age rating increases, proportion of apps decreases.

# In[ ]:


df["Age Rating"].value_counts().plot.pie(autopct="%1.1f", explode=[0,0,0.1,0], figsize=(6,6))
plt.title("Age Rating wise app proportions")
plt.show()


# ## Average size of apps in each Primary Genre
# * Games genre is having average highest average size of apps, around 110 MB.

# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(data=df[df["Primary Genre"].isin(top_genres)], x="Primary Genre", y="Size")
plt.xticks(rotation=90)
plt.title("Primary Genre wise average size of apps")
plt.show()


# ## Which price range bracket has the highest and least proportion of apps?
# * Most of the apps available are for free (83.7%).
# * There are few apps which are medium (1.6%) and high (0.2%) priced.
# * As the price of the apps increases, we see number of apps descreases drastically.

# In[ ]:


df["Price Range"].dropna().value_counts().plot.pie(autopct="%1.1f", explode=[0,0.1,0,0], figsize=(6,6))
plt.title("Price Range wise proportion of apps")
plt.show()


# ## Age Rating repartitioned by Price Range
# * There seems to be no significant change in the proportion of price range in all the four age rating categories.

# In[ ]:


ct_agerating_pricerange = pd.crosstab(df["Age Rating"], df["Price Range"], normalize=0)
ct_agerating_pricerange.plot.bar(stacked=True, figsize=(8,5))
plt.xticks(rotation=0)
plt.title("Age Rating repartioned by Price Range")
plt.show()


# ## Average user rating for each price range
# * There seems to be no difference in average user rating across price range.

# In[ ]:


plt.figure(figsize=(8,5))
sns.barplot(data=df, x="Price Range", y="Average User Rating")
plt.title("Average user rating in each price range")
plt.show()


# ## Which genre and age rating has the most number of in-app purchases on average?
# * Finance genre is having the highest no. of in-app purhases on average.
# * Finance and Games genres are having more no. of in-app purhases than education on an average.
# * Age rating group 12+ is having the highest no. of in-app purhases on average.
# 
# ## Which genre and age rating has the highest value of in-app purchases on average?
# * Finance genre is having the highest value of in-app purchases on average.
# * Age rating 12+ is having the highest value of in-app purchases on average.

# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(15,10))
sns.barplot(data=df[df["Primary Genre"].isin(top_genres)], x="Primary Genre", y=df["In-app Purchases"].dropna().map(lambda x: len(x)), ax=ax[0,0]).set_xticklabels(ax[0,0].get_xticklabels(), rotation=45)
sns.barplot(data=df, x="Age Rating", y=df["In-app Purchases"].dropna().map(lambda x: len(x)), ax=ax[0,1])
sns.barplot(data=df[df["Primary Genre"].isin(top_genres)], x="Primary Genre", y="Total In-app Purchases", ax=ax[1,0]).set_xticklabels(ax[1,0].get_xticklabels(), rotation=90)
sns.barplot(data=df, x="Age Rating", y="Total In-app Purchases", ax=ax[1,1])
ax[0,0].set_title("Average no. of in-app purchase in each genre")
ax[0,1].set_title("Average no. of in-app purchase in each age rating")
ax[1,0].set_title("Average value of in-app purchase in each genre")
ax[1,1].set_title("Average value of in-app purchase in each age rating")
plt.show()


# ## Which genres is games are having most no. of games?
# * Strategy genre is having the most no. of games.
# 
# ## What is the highest rated genre?
# * There seems to be no difference in the average user rating across genres.

# In[ ]:


# creating list of top game genres
top_game_genre = list(df["Game Genre"].value_counts().head(11).index)


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.countplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", ax=ax[0]).set_xticklabels(ax[0].get_xticklabels(), rotation=90)
sns.barplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", y="Average User Rating", ax=ax[1]).set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[0].set_title("Count of games in each genre")
ax[1].set_title("Average rating of games in each genre")
plt.show()


# ## In-app purchases trend in different genres of games
# * Role playing games are having highest no. of in-app purchases and total value of in-app purchases on an average.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18,5))
sns.barplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", y=df["In-app Purchases"].dropna().map(lambda x: len(x)), ax=ax[0]).set_xticklabels(ax[0].get_xticklabels(), rotation=90)
sns.barplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", y="Total In-app Purchases", ax=ax[1]).set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_title("Average no. of in-app purchases game genre wise")
ax[1].set_title("Average total value of in-app purchases game genre wise")
plt.show()


# ## Correlation heat map of continuous variables
# * There seems to be no strong positive or negative correlation between any two continuous variables.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df[["Price","Average User Rating","Total In-app Purchases","Age Rating","Size"]].corr(), annot=True, cmap="coolwarm")
plt.show()


# ## Timeseries analysis of apps
# * No. of apps released showed increasing trend till 2016, peaked in 2016, and then decreased in following years.
# * No. of apps released in highest during the month of September.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,6))
df.groupby("Release Year")["Name"].agg("count").plot(ax=ax[0])
df.groupby("Release Month")["Name"].agg("count").plot(ax=ax[1])
ax[0].set_ylabel("No. of apps")
ax[1].set_ylabel("No. of apps")
ax[0].set_title("No. of apps released in each year")
ax[1].set_title('No. of apps released in each month')
plt.show()

