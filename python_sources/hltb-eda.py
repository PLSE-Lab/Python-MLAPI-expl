#!/usr/bin/env python
# coding: utf-8

# # NOTE
# This kernel is in progress.  I am editing it now and again.  It is just for fun!

# # Introduction
# In this notebook, I will be exploring my [HowLongToBeat](https://howlongtobeat.com) games [dataset](https://www.kaggle.com/kasumil5x/howlongtobeat-games-completion-times).  The data comes from my open-source web scraper available on [GitHub](https://github.com/KasumiL5x/hltb-scraper).  Let's begin!

# # Imports and data loading
# Let's get the imports and loading of base CSV files out of the way now.

# In[ ]:


import re
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-ticks')
import seaborn as sns


# In[ ]:


games = pd.read_csv('../input/all-games-processed.csv', index_col='id')
completions = pd.read_csv('../input/all-completions.csv', index_col='id')


# # Missing data
# Let's check what's missing.  If there are any problems, we can terminate the rows if necessary or impute them where required.

# In[ ]:


games.isnull().sum()


# Some observations about missing data in the `games` set:
# 
# * There are two missing `title` entries, so we can investigate and likely remove them.
# * The average times (scraped, not calculated), `main_story`, `main_plus_extras`, `completionist`, `all_styles`, `coop`, and `versus` have a large number of missing values, but that is acceptable as it simply means that an entry doesn't have an average time specified for that particular type of play.  `coop` and `versus` especially make sense, as there are far fewer games of this type than others.
# * `type` has quite a lot of missing values, but given that it only exists to differentiate between a specific subtype of game, this is expected.
# * `developers` and `publishers` have a few thousand missing values, which means that if we were to aggregate some statistic based on a per-developer or per-publisher basis, we would be missing them.  These cannot be safely imputed other than looking up the developers and publishers manually.  Games in the same series can be developed and even published by different companies.
# * `platforms` likewise has quite a number of missing values.  As with the above, we could look up all of the data, but unfortunately not all data on the source website is complete.
# * `genres` has a few thousand missing values which again is down to the source having a lack of data.  This could likewise be resolved by looking up the specific data for each entry.
# * `release_na`, `release_eu`, and `release_jp` have expected missing values, although it's almost guaranteed that there are genuine missing values here, too.  It is expected that Japan has the lease released games, given the scale of the video games industry in the west.  Europe has a surprisingly high number of missing values given its prominence in the gaming world.  Given the difference in `release_na` and `release_eu`, I'm concluding that there are a large number of actual missing entries here for release data since there's definitely not *that* big of a margin between the NA and EU games.

# ## Missing titles
# Let's see what has missing titles and decide what to do.

# In[ ]:


games[games.title.isna()]


# I have found the first game's [Wikipedia page](https://en.wikipedia.org/wiki/N.U.D.E.@_Natural_Ultimate_Digital_Experiment), verified by the developer, publisher, and precise Japanese release date, as well as checking the developer's portfolio to check nothing else was released at the same time. The game appears to be `Xbox` only and has an *interesting* title.
# 
# I've also browsed the web and found that *Stormcloud Games* have only released a single game that was out on August 09 2016.  The [list of PlayStation4 games](https://en.wikipedia.org/wiki/List_of_PlayStation_4_games) on WikiPedia shows the name we are looking for as well as the Japanese release date!  The [official page](https://stormcloudgames.com/portfolio/brutal) doesn't much extra info, but does tell us that the game is available on `PlayStation 4` and `PC`.
# 
# Let's replace those missing titles and any extra info that was found.  This could be done for all games with missing information, but I don't have an army of people to do it for me!

# In[ ]:


# first one
games.loc[19962, ['title', 'platforms']] = ['N.U.D.E.@ Natural Ultimate Digital Experiment', 'Xbox']

# second one
games.loc[38799, ['title', 'platforms', 'release_jp']] = ['Brut@l', 'PlayStation 4, PC', 'October 20, 2016']


# ## Missing platforms
# With `completions`, we only have missing values in the `platform` column.  In this dataframe, `platform` refers to the singular platform on which the completion was done, whereas in the `games` dataset, it refers to a comma-separated list of *all* platforms the game is available on.  It is possible that some `platform` values from the `completions` table can be used to infer a platform for the missing values in the `games` table.  Note the print telling us how many null values we had before and how many we gathered from this replacement process!

# In[ ]:


completions.isnull().sum()


# In[ ]:


# Replace missing `platform` entries with an empty string for concat purposes.
completions['platform'] = completions.platform.fillna('')

# `games` identifiers where platform is null AND there is at least one valid completion entry (not all games will have entries).
missing_platform_ids = games[games.platforms.isnull()].index.intersection(completions.index)

# Create a comma-space separated list of all unique platforms (there may be multiple entries and for different platforms).
replacement_platforms = pd.DataFrame(completions.loc[missing_platform_ids].groupby('id').platform.unique().transform(lambda x: ', '.join(x)))
# Drop any that resulted in an empty string.
replacement_platforms = replacement_platforms[replacement_platforms.platform != '']
# Rename column to match that in `games` for a proper merge.
replacement_platforms.columns = ['platforms']

# Use `update()` to replace NaN entries with any matches.
games.update(replacement_platforms)

# 15021 is from above output. If this changes, the data changed!
print('Reduced `game.platform` NaN values from {} to {}!'.format(15021, games.platforms.isnull().sum()))


# Given the information that we have, without looking things up manually, there's not much else to do in the way of fixing missing data.  As discussed above, lots of missing data is not present for a good reason rather than being an error, and that which *is* and error is not practical to fix.

# # Minor cleanups
# There are a few things that need cleaning up before continuing.

# ## NaN to empty string
# For simplicity, I'm going to convert all NaN values in `developers`, `publishers`, `platforms`, and `genres` into empty strings, which makes dealing with it a lot easier as we don't have to filter out null values each time.

# In[ ]:


games['developers'] = games.developers.fillna('')
games['publishers'] = games.publishers.fillna('')
games['platforms'] = games.platforms.fillna('')
games['genres'] = games.genres.fillna('')


# ## Bad endings!
# I noticed that some of the platform entries are wrong.  We will remove `, ` and `,` from the *end* of any platforms which will fix `3do,`.  I know of another present in the genres, in that `arcade,` and `arcade` do not match due to the trailing comma.  Let's just remove from all four of the same columns to be safe.

# In[ ]:


games['developers'] = games.developers.apply(lambda x: re.sub(r',\s?$', '', x))
games['publishers'] = games.publishers.apply(lambda x: re.sub(r',\s?$', '', x))
games['platforms'] = games.platforms.apply(lambda x: re.sub(r',\s?$', '', x))
games['genres'] = games.genres.apply(lambda x: re.sub(r',\s?$', '', x))


# I've also noticed that a number of `developers` entries, even when split, contain rogue `(` characters.  It appears that some entries have `(platform)` after the developer's name to indicate that they only worked on one platform in the series, such as when porting.  We could pull out the values in brackets and make a new column of the contribution each developer has, but there is no standard as this is user-entered data, so I'm just going to remove this.  I mostly care about developer working on the *game*, not the particular platform or that they ported it.

# In[ ]:


games['developers'] = games['developers'].str.replace(r'\s?\(.*?\)', '')


# There's also a rogue `;), ` entry in `loc=47947`.  Since I know the ID and full string, I'll just overwrite it by hand.

# In[ ]:


games.loc[47947, 'developers'] = 'quickdraw studios'


# ## Bad genres!
# While exploring, I noticed three genres that are incorrectly reported.  Two are URLs, probably included for self-promotion.  The third is a [Sanic joke game](https://scratch.mit.edu/projects/22036060/).  I'm choosing to remove this one given its genre of `ANAZIGN` and incorrect title.  It feels like a joke entry.

# In[ ]:


# `http://www.cobramobile.com/` and `http://prinnies.com/`
games = games[~games.genres.str.contains('http', na=False)]

# Sanic joke.
games = games[games.genres != 'ANAZIGN']


# ## Case mismatch
# The problem with user-submitted data is that nobody adheres to proper standards.  While we could manually sift through slight misspellings, that would take too long.  Another common error is that captialization is not followed.  This means that `Microsoft` and `microsoft` are not the same.  Converting all appropriate entries to lower case should help to solve this and remove at least some duplicates.

# In[ ]:


games['developers'] = games.developers.str.lower()
games['publishers'] = games.publishers.str.lower()
games['platforms'] = games.platforms.str.lower()
games['genres'] = games.genres.str.lower()


# # Exploring the number of games per...
# Let's begin by browsing through the `games` dataset with respect to a few different columns.
# 
# Below I've defined a function to take a given column, split every entry by `', '`, and then return all *unique* entries from all rows.

# In[ ]:


# Helper function to split a column by ', ' and return all unique values.
def unique_from_split(col):
    tmp = pd.Series([item for sublist in games[col].str.split(', ') for item in sublist]).unique().tolist()
    tmp.remove('')
    return tmp
#end


# Here I'm making a lookup table for optimization.  It does take memory, but at the cost of computation being worse.  To compute the upcoming sections, we'd need to split the rows again and again, once for each platform, developer, and so on.  Adding those together, we'd be splitting pretty much every string more than 40k times.  This can be avoided by splitting them all once now, and then looking up the already split values in this lookup table .  This way, we only split once!  The checks are of course still done, but minus the string splitting.

# In[ ]:


games_lookup = {}

# i = 0
for idx, row in games.iterrows():
    # split devs and remove empty strings
    devs = row.developers.split(', ')
    if '' in devs:
        devs.remove('')    
    
    # publishers
    pubs = row.publishers.split(', ')
    if '' in pubs:
        pubs.remove('')
        
    # platforms
    plats = row.platforms.split(', ')
    if '' in plats:
        plats.remove('')
        
    # genres
    genres = row.genres.split(', ')
    if '' in genres:
        genres.remove('')
    
    games_lookup[idx] = {
        'developers': devs,
        'publishers': pubs,
        'platforms': plats,
        'genres': genres
    }
#end


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# developers\ndevelopers_lookup = {\n}\nfor dev in unique_from_split('developers'):\n    developers_lookup[dev] = [i for i in games_lookup.keys() if dev in games_lookup[i]['developers']]\n    \n# publishers\npublishers_lookup = {\n}\nfor pub in unique_from_split('publishers'):\n    publishers_lookup[pub] = [i for i in games_lookup.keys() if pub in games_lookup[i]['publishers']]\n\n# platforms\nplatforms_lookup = {\n}\nfor plat in unique_from_split('platforms'):\n    platforms_lookup[plat] = [i for i in games_lookup.keys() if plat in games_lookup[i]['platforms']]\n\n# genres\ngenres_lookup = {\n}\nfor gen in unique_from_split('genres'):\n    genres_lookup[gen] = [i for i in games_lookup.keys() if gen in games_lookup[i]['genres']]")


# ## ...platform
# `pc` is the obvious winner by quite a margin.  Given that the website appears to aim toward PC gamers the most (such as allowing you to check your Steam library), this makes sense.  All of the PlayStation and Xbox consoles, except the original Xbox, are present in the top 10.  Given that this is user-submitted information, we cannot guarantee that `playstation` refers purely to the original playstation, as users submitting a PS4 game could have used just the `playstation` name.  We could check this by hand, or make assumptions based on the release date (if present), but that would take a while.  The same applies for Xbox.  Regardless, both the PlayStation and Xbox platforms hold the next few places, as expected.  The nintendo family follow playstation.  That is all of the 'major players,' so to speak, which is expected.

# In[ ]:


count_per_platform = pd.DataFrame(
    [(key, len(val)) for key, val in platforms_lookup.items()],
    columns=['Platform', 'Count']
)

g = sns.barplot(data=count_per_platform.sort_values('Count', ascending=False).head(10), x='Platform', y='Count')
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('Total games per platform')
plt.show()


# ## ...developer
# `konami` surprisingly came out on top.  Worth noting is that Konami have multiple divisions, so even though the graph below lists `konami` at close to 400, there are some extra subsidiaries that act in different countries.  They are all essentially Konami and could perhaps have been merged.  Regardless, looking through the list of games that `'konami'` have listed, there is a lot of Castevania games listed, many Dance Dance Revolution variations, Pro Evolution Soccer, Silent Hill, Yu-Gi-Oh, and so on.  Konami develop (and publish) a surprising number of games popular all around the world.  Similarly with 'capcom', they have a handful of popular series that contribute to their close to 350 count, such as the Resident Evil series, Monster Hunter, Phoenix Wright, Street Fighter, and so on.  Following this, we have `sega`, before the count starts dropping steadily.
# 
# In the top 10 displayed below, only **three** are not Japanese developers.  That is quite interesting.  Whether this is due to sheer numbers or that those using this website have a certain taste in games (as not every game will be submitted) is hard to say.

# In[ ]:


count_per_developer = pd.DataFrame(
    [(key, len(val)) for key, val in developers_lookup.items()],
    columns=['Developer', 'Count']
)

g = sns.barplot(data=count_per_developer.sort_values('Count', ascending=False).head(10), x='Developer', y='Count')
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('Total games per developer')
plt.show()


# ## ...publisher
# Given that a number of the Japanese developers are also publishers, it comes at no surprise to see set few of the same right at the top.  It's also worth keeping in mind that a lot of the older game systems were from Japan (Sega's whole hardware series alone is quite substantial), and Nintendo are still in the hardware game today.  It does therefore make sense to see a large number of Japanese publishers listed.
# 
# However, we do see `electronic arts`, `ubisoft`, and `activision` sitting at around the same values.  In the western part of the world, and to an extent even in the east, these three behemoths publish a wide range of in-house titles and third-party titles.  It is not unexpected to see them here.

# In[ ]:


count_per_publisher = pd.DataFrame(
    [(key, len(val)) for key, val in publishers_lookup.items()],
    columns=['Publisher', 'Count']
)

g = sns.barplot(data=count_per_publisher.sort_values('Count', ascending=False).head(10), x='Publisher', y='Count')
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('Total games per publisher')
plt.show()


# ## ...genre
# To no surprise, `action` leads the way by quite a margin.  However, it must be considered here that this is data entered by users without any formal procedure.  The term `action` is quite a squishy word which casts a wide net.  Further terms, which are also present on the graph like `shooter` and even `first-person` are often, if not always bound together with `action`.  Therefore, we may see artifical inflation of `'action'` titles.  Regardless, it is a popular genre and it is expected to be high.
# 
# `adventure` games likewise cast a wide net in terms of description.  However, their presence here is expected, as these games are often long and epic experiences, and knowing how long they take to beat can be a critical choice for those wanting to play them.  The `role-playing` category that follows can be explained in the same way.
# 
# This column feels especially noisy and must be taken with quite a substantial pinch of salt.  More work could be done to check ngrams (perhaps bi or even tri) of the genres for a given entry to see what common patterns appear (such as `action` + `first-person` + `shooter`, which I'm sure we've all heard of before).

# In[ ]:


count_per_genre = pd.DataFrame(
    [(key, len(val)) for key, val in genres_lookup.items()],
    columns=['Genre', 'Count']
)

g = sns.barplot(data=count_per_genre.sort_values('Count', ascending=False).head(10), x='Genre', y='Count')
sns.despine()
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('Total games per genre')
plt.show()


# # Preferred platforms and genres...
# We have the top 10 in all categories now.  It would also be interesting to see what the most popular platform and genre is per developer and/or publisher.  This could provide some insight, but more importantly can help us better confirm accuracy of the data.  If Microsoft's favorite platform is Dreamcast, then there's something wrong!

# ## ...by developer
# blah

# ## ...by publisher
# blah

# * Which genre to the top 10 developers prefer?
# * Which platform to the top 10 developers prefer?

# # DEBUG AREA

# In[ ]:


# debug for checking values when writing
games[games.genres.str.contains('action')][['title', 'platforms']].values

