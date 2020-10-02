#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import ast
from collections import defaultdict

# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # Use seaborn default style

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Import Datasets

# In[ ]:


ranking_df = pd.read_csv('../input/boardgamegeek-reviews/2019-05-02.csv')
ranking_df.head()


# In[ ]:


review_df = pd.read_csv('../input/boardgamegeek-reviews/bgg-13m-reviews.csv', index_col=0)
review_df.head()


# In[ ]:


detail_df = pd.read_csv('/kaggle/input/boardgamegeek-reviews/games_detailed_info.csv', index_col=0)
detail_df.iloc[:, :20].head()


# In[ ]:


detail_df.iloc[:, 20:40].head()


# In[ ]:


detail_df.iloc[:, 40:].head()


# # EDA Ranking Data
# 
# - No null values in all variables except thumbnail
# - No duplicate ids
# 

# In[ ]:


ranking_df.shape


# In[ ]:


ranking_df.dtypes


# In[ ]:


ranking_df.describe()


# In[ ]:


ranking_df.isna().sum()


# In[ ]:


ranking_df['ID'].value_counts().head()


# In[ ]:


ranking_df['Year'].plot(kind='hist')
plt.show()


# In[ ]:


ranking_df['Average'].plot(kind='hist')
plt.show()


# In[ ]:


ranking_df['Bayes average'].plot(kind='hist')
plt.show()


# In[ ]:


ranking_df['Users rated'].plot(kind='hist')
plt.show()


# # EDA Reviews Data
# 
# - All variables do not have null values except comments
# - Left skewed distribution for ratings
# - About 80% of ratings have no comments

# In[ ]:


review_df.shape


# In[ ]:


review_df.dtypes


# In[ ]:


review_df.describe()


# In[ ]:


review_df.isna().sum()


# In[ ]:


review_df['rating'].plot(kind='hist')
plt.show()


# In[ ]:


review_df['ID'].value_counts().head()


# In[ ]:


review_df['comment'].isna().sum() / review_df.shape[0] * 100


# # EDA Game Details
# 
# - All categories have unique ranks for each game, except the main board game ranking
# - Majority of the 15 categories only have 1 board game, except abstract, children, customisable, family, party, strategy, thematic and war (categories exclude main board game rating)
# - All numerical variables excluding categories have no null values

# In[ ]:


detail_df.shape


# In[ ]:


detail_df.dtypes


# In[ ]:


# Describe category features
detail_df.iloc[:, :16].describe()


# In[ ]:


# Describe all other features
detail_df.iloc[:, 16:].describe()


# In[ ]:


detail_df.isna().sum()


# In[ ]:


detail_df['Board Game Rank'].value_counts().head()


# In[ ]:


# not_ranked_ids = detail_df[detail_df['Board Game Rank'] == 'Not Ranked']['id']
# ranking_df.loc[ranking_df['ID'].isin(not_ranked_ids)]


# In[ ]:


detail_df['Board Game Rank'] = detail_df['Board Game Rank'].replace('Not Ranked', np.nan)                                                             .astype('float')


# In[ ]:


categories = detail_df.columns[:16]

for category in categories:
    num_unique = detail_df[category].nunique()
    count = detail_df[category].count()
    print('Category:', category, '\n',
          'Number of games:', count, '\n',
          'Has all unique values?', num_unique == count, '\n',
          'First game id:', detail_df.loc[detail_df[category] > 0]['id'].min())


# In[ ]:


detail_df['average'].plot(kind='hist')
plt.show()


# In[ ]:


detail_df['averageweight'].plot(kind='hist')
plt.show()


# In[ ]:


detail_df['bayesaverage'].plot(kind='hist')
plt.show()


# In[ ]:


detail_df['id'].plot(kind='hist')
plt.show()


# In[ ]:


for col in detail_df.columns[33:43]:
    detail_df[col].plot(kind='hist')
    plt.title(col)
    plt.show()


# In[ ]:


detail_df['stddev'].plot(kind='hist')
plt.show()


# In[ ]:


detail_df['trading'].plot(kind='hist')
plt.show()


# In[ ]:


for col in detail_df.columns[-4:]:
    detail_df[col].plot(kind='hist')
    plt.title(col)
    plt.show()


# # Repeated Columns Analysis
# 
# - IDs are present for every table (join condition)
# - Ranking and game detail tables have published year, rank, average, bayes average, users rated and thumbnail columns
# - Looking at Kaggle, the dataset was last updated on 2nd June, and looking at the number of users rated, the game detailed info table looks more updated
# - Should drop ranking data's repeated columns and join the rest (BGG URL and Name) with game detail dataset

# In[ ]:


ranking_sub_df = ranking_df[['ID', 'Year', 'Rank', 'Average', 'Bayes average', 'Users rated', 'Thumbnail']]
ranking_sub_df.head()


# In[ ]:


detail_sub_df = detail_df[['id', 'yearpublished', 'Board Game Rank', 'average', 'bayesaverage', 'usersrated', 'thumbnail']]
detail_sub_df.head()


# In[ ]:


joined_df = ranking_sub_df.merge(detail_sub_df, left_on='ID', right_on='id', how='left')
joined_df.head(20)


# # Ordered Dictionaries in Game Detail
# 
# - Might be useful as additional features

# In[ ]:


detail_df['suggested_language_dependence'].iloc[1]


# In[ ]:


detail_df['suggested_num_players'].iloc[1]


# In[ ]:


detail_df['suggested_playerage'].sample(1).iloc[0]


# In[ ]:


# keys = set()

# for string in detail_df['suggested_language_dependence']:
#     if not type(string) is float:
#         lst = ast.literal_eval(string.replace('OrderedDict', ''))
#         for nested_lst in lst:
            
#             for key, value in nested_lst:
#                 keys.add(key)
#     elif not np.isnan(string):
#         # Not null, print value
#         print(string)
        
# keys


# In[ ]:


# import ast
# import collections
# import re

# str_list_SLD = detail_df['suggested_language_dependence'].iloc[1]
# #values = re.search(r"OrderedDict\((.*)\)",str_list_SLD).group(1)
# #mydict = collections.OrderedDict(ast.literal_eval(values))
# def convert_ordered_dict(str_list):
#     str_list = str_list.replace("OrderedDict","")
#     return ast.literal_eval(str_list)

# list_SLD = convert_ordered_dict(str_list_SLD)

# for list_of_tup in list_SLD:
#     for tup in list_of_tup:
#         print(tup)


# In[ ]:


# 1 : No necessary in-game text
# 2 : Some necessary text - easily memorized or small crib sheet
# 3 : Moderate in-game text - needs crib sheet or paste ups
# 4 : Extensive use of text - massive conversion needed to be playable
# 5 : Unplayable in another language

# sld_df = pd.DataFrame()

# sld_df["id"] = detail_df["id"]

# str_list_SLD = detail_df['suggested_language_dependence']

# def convert_ordered_dict(str_list):
#     str_list = str_list.replace("OrderedDict","")
#     return ast.literal_eval(str_list)

# def get_suggested_language_dependence_df(str_list_SLD):
#     ratings_df = pd.DataFrame(columns=["1","2","3","4","5"])
#     for items in str_list_SLD:
#         if isinstance(items,str):
#             list_items = convert_ordered_dict(items)
#             new_dict = {}
#             for i,lists in enumerate(list_items):
#                 new_dict[str(i+1)] = lists[2][1] 
#             ratings_df = ratings_df.append(new_dict, ignore_index=True)
#         else:
#             none_dict = {"1":None,"2":None,"3":None,"4":None,"5":None}
#             ratings_df = ratings_df.append(none_dict, ignore_index=True)
    
#     return ratings_df

# ratings_df = get_suggested_language_dependence_df(detail_df['suggested_language_dependence'])


# In[ ]:


# sld_df = sld_df.merge(ratings_df,left_index=True,right_index=True)
# sld_df


# In[ ]:


# str_list_SNP = detail_df['suggested_num_players']

# def get_suggested_num_players(str_list_SNP):
#     iter_count = 0
#     snp_df = pd.DataFrame()
    
#     for items in str_list_SNP:
#         list_SNP = convert_ordered_dict(items)
#         if isinstance(list_SNP[0],list):
#             for tup in list_SNP:
#                 new_dict = {}
#                 num_players = tup[0][1]
#                 result_list = tup[1][1]
#                 best_votes = result_list[0][1][1]
#                 rec_votes = result_list[1][1][1]
#                 not_rec_votes = result_list[2][1][1]
#                 new_dict["id"] = detail_df.iloc[iter_count]["id"]
#                 new_dict["num_players"] = num_players
#                 new_dict["best_votes"] = best_votes
#                 new_dict["recommended_votes"] = rec_votes
#                 new_dict["not_recommended_votes"]= not_rec_votes
#                 snp_df = snp_df.append(new_dict, ignore_index=True)
#         else:
#             if len(list_SNP)==2:
#                 new_dict = {}
#                 new_dict["id"] = detail_df.iloc[iter_count]["id"]
#                 new_dict["num_players"] = list_SNP[0][1]
#                 new_dict["best_votes"] = list_SNP[1][1][0][1][1]
#                 new_dict["recommended_votes"] = list_SNP[1][1][1][1][1]
#                 new_dict["not_recommended_votes"]= list_SNP[1][1][2][1][1]
#                 snp_df = snp_df.append(new_dict, ignore_index=True)
#             else:
#                 new_dict = {}
#                 new_dict["id"] = detail_df.iloc[iter_count]["id"]
#                 new_dict["num_players"] = list_SNP[0][1]
#                 new_dict["best_votes"] = None
#                 new_dict["recommended_votes"] = None
#                 new_dict["not_recommended_votes"]= None
#                 snp_df = snp_df.append(new_dict, ignore_index=True)
#         break
#         iter_count += 1
#     return snp_df
            
# snp_df = get_suggested_num_players(str_list_SNP)


# In[ ]:


# str_list = detail_df[detail_df["id"]==13]["suggested_num_players"]

# print(type(str_list))

# snp_df[snp_df["id"]==13], l_list


# In[ ]:


# detail_df.iloc[0]["suggested_num_players"],detail_df.iloc[0]["id"],


# In[ ]:


# test2 = detail_df.iloc[5038]["suggested_num_players"]
# list_test = convert_ordered_dict(test2)
# list_test


# In[ ]:


# detail_df.iloc[5038]


# In[ ]:


# detail_df['suggested_playerage'].iloc[1]


# In[ ]:


# Flatten ordered dictionaries into dataframe


# # Correlation Plots

# In[ ]:


corr = detail_df.iloc[:, 16:].corr()
corr = corr.dropna(how='all', axis=1).dropna(how='all', axis=0).round(2)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
plt.subplots(figsize=(20,20))
sns.heatmap(corr, cmap='RdYlGn', annot=True, linewidths=.5)
plt.title('Correlation plot without genre variables')
plt.show()


# In[ ]:


corr = detail_df[detail_df.columns.difference(['Accessory Rank', "Amiga Rank", "Arcade Rank", "Atari ST Rank","Commodore 64 Rank",
                                               "RPG Item Rank", "Video Game Rank", "median", "thumbnail", "id", 
                                               "image"])] \
       .corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr, annot=True, cmap='RdYlGn')
plt.title('Correlation plot with genre variables that has >1 board game (excluding median, thumbnail, id and image)')
plt.show()


# In[ ]:


sns.pairplot(data=detail_df, vars=['averageweight', 'bayesaverage', 'maxplayers', 'maxplaytime', 'minage', 
                                   'minplayers', 'minplaytime','playingtime'])
plt.title('Correlation plot of selected features for modelling')
plt.show()


# # Descriptive Analytics

# ## What is the trend of board game publishes?
# 
# - Two spikes, one in year 0 and another starting from 20th century (1900~)

# In[ ]:


plt.figure(figsize=(20, 5))
detail_df['yearpublished'].value_counts().sort_index().plot()
plt.xlabel('Year')
plt.ylabel('Board Games Published')
plt.title('Number of Board Games Published over Time')
plt.show()


# In[ ]:


detail_df['yearpublished'].plot(kind='box')
plt.show()


# ## What are the trends during periods of high volume board game publishes?
# 
# - Board games with no published year for year 0 publishes
# - Exponential increase in board games since the 20th century, with a dip in board game publishes in the last few years

# In[ ]:


detail_df['yearpublished'].loc[(detail_df['yearpublished'] >= -500) & (detail_df['yearpublished'] <= 1000)].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Board Games Published')
plt.title('Number of Board Games Published over Time (500BC to 1000)')
plt.show()


# In[ ]:


detail_df.merge(ranking_df[['ID', 'Name']], how='left', left_on='id', right_on='ID').loc[detail_df['yearpublished'] == 0]['Name'].to_list()


# In[ ]:


detail_df['yearpublished'].loc[(detail_df['yearpublished'] >= 1900) & (detail_df['yearpublished'] < 2019)].value_counts().sort_index().plot()
plt.xlabel('Year')
plt.ylabel('Board Games Published')
plt.title('Number of Board Games Published over Time (1900 to 2018)')
plt.show()


# ## What are the Bayesian average distributions for board games published in the recent decades?

# In[ ]:


for year in range(1950, 2011, 10):
    detail_df[(detail_df['yearpublished'] >= year) & (detail_df['yearpublished'] < (year + 10))]['bayesaverage']         .hist(bins=np.arange(0, 10, 0.5))
    plt.xlabel('Bayes Average')
    plt.ylabel('Board Games Count')
    plt.title('Distribution of Bayes Average for board games published between ' + str(year) + ' and ' + str(year + 10))
    plt.show()


# ## Is there any relation between the difficulty of board games and their Bayesian averages?
# 
# - Marginally positive correlation

# In[ ]:


detail_df[["averageweight", 'bayesaverage']].plot(kind='scatter', x='averageweight', y='bayesaverage')
plt.title('Correlation between Average Weight and Bayesian Average')
plt.show()


# In[ ]:


detail_df[["averageweight", 'bayesaverage']].corr()


# ## Who are the board game artists that has the most board games published?

# In[ ]:


plt.figure(figsize=(20, 7))
board_game_artists = detail_df['boardgameartist'].map(lambda artist: ast.literal_eval(artist) if type(artist) != float else artist)                                                  .dropna()                                                  .sum()
pd.Series(board_game_artists).value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Artist')
plt.ylabel('Board Game Count')
plt.title('Top 50 Board Game Artists')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 7))
board_game_artists = detail_df.loc[detail_df['bayesaverage'] >= detail_df['bayesaverage'].median()]['boardgameartist']                               .map(lambda artist: ast.literal_eval(artist) if type(artist) != float else artist)                               .dropna()                               .sum()
pd.Series(board_game_artists).value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Artist')
plt.ylabel('Board Game Count')
plt.title('Top 50 Board Game Artists for Top 50% Board Game Titles based on Bayesian Average')
plt.show()


# ## Who are the board game designers with the most board games published?

# In[ ]:


plt.figure(figsize=(20, 7))
board_game_designer = detail_df['boardgamedesigner'].map(lambda artist: ast.literal_eval(artist) if type(artist) != float else artist)                                                  .dropna()                                                  .sum()
pd.Series(board_game_designer).value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Designer')
plt.ylabel('Board Game Count')
plt.title('Top 50 Board Game Designers')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 7))
board_game_designer = detail_df.loc[detail_df['bayesaverage'] >= detail_df['bayesaverage'].median()]['boardgamedesigner']                                .map(lambda artist: ast.literal_eval(artist) if type(artist) != float else artist)                                .dropna()                                .sum()
pd.Series(board_game_designer).value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Designer')
plt.ylabel('Board Game Count')
plt.title('Top 50 Board Game Designers for Top 50% Board Game Titles based on Bayesian Average')
plt.show()


# ## How many board games are there in each category?

# In[ ]:


plt.figure(figsize=(20, 7))
board_game_category = detail_df['boardgamecategory'].map(lambda artist: ast.literal_eval(artist) if type(artist) != float else artist)                                                  .dropna()                                                  .sum()
pd.Series(board_game_category).value_counts()[:50].plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Board Game Count')
plt.title('Top 50 Board Game Categories')
plt.show()


# ## What are the board games with the most expansions?

# In[ ]:


sub_df = detail_df.merge(ranking_df[['Name', 'ID']], how='left', left_on='id', right_on='ID')[['Name', 'boardgameexpansion', 'bayesaverage']]
sub_df['numexpansions'] = sub_df['boardgameexpansion'].map(lambda x: len(ast.literal_eval(x)) if not type(x) == float else x)
sub_df.head()


# In[ ]:


plt.figure(figsize=(20, 7))
sub_df.set_index('Name')['numexpansions'].sort_values(ascending=False)[:100].plot(kind='bar')
plt.xlabel('Board Game Name')
plt.ylabel('Count')
plt.title('Top 100 Board Games with the Most Expansions')
plt.show()


# ## Is there a correlation between the number of board game expansions and the Bayesian average?

# In[ ]:


sub_df = detail_df[['boardgameexpansion', 'bayesaverage']].copy()
sub_df['numexpansions'] = sub_df['boardgameexpansion'].map(lambda x: len(ast.literal_eval(x)) if not type(x) == float else x)
sub_df.head()


# In[ ]:


sub_df.plot(kind='scatter', x='numexpansions', y='bayesaverage')
plt.show()


# In[ ]:


sub_df.loc[sub_df['numexpansions'] < 100].plot(kind='scatter', x='numexpansions', y='bayesaverage')
plt.show()


# In[ ]:


sub_df[['numexpansions', 'bayesaverage']].corr()


# ## Is there a correlation between the number of board game expansions and the year published?

# In[ ]:


sub_df = detail_df[['boardgameexpansion', 'yearpublished']].copy()
sub_df['numexpansions'] = sub_df['boardgameexpansion'].map(lambda x: len(ast.literal_eval(x)) if not type(x) == float else x)
sub_df.head()


# In[ ]:


sub_df.plot(kind='scatter', x='yearpublished', y='numexpansions')
plt.show()


# In[ ]:


sub_df.loc[sub_df['yearpublished'] > 1900].plot(kind='scatter', x='yearpublished', y='numexpansions')
plt.show()


# ## What are the popular board game mechanics among board games?

# In[ ]:


mechanics = detail_df['boardgamemechanic'].map(lambda x: ast.literal_eval(x) if not type(x) == float else x)                                           .dropna()                                           .sum()
plt.figure(figsize=(20, 7))
pd.Series(mechanics).value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Mechanic')
plt.ylabel('Count')
plt.title('Top 50 Board Game Mechanics')
plt.show()


# ## Which are the major board game publishers?

# In[ ]:


publishers = detail_df['boardgamepublisher'].map(lambda x: ast.literal_eval(x) if not type(x) == float else x)                                             .dropna()                                             .sum()
plt.figure(figsize=(20, 7))
pd.Series(publishers).value_counts()[:50].plot(kind='bar')
plt.xlabel('Publisher Name')
plt.ylabel('Board Game Count')
plt.title('Top 50 Board Game Publishers')
plt.show()


# ## Which are the most rated board games?

# In[ ]:


plt.figure(figsize=(20, 7))
review_df['name'].value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Name')
plt.ylabel('Rating Count')
plt.title('Top 50 Most Rated Board Games')
plt.show()


# ## Which are the most reviewed board games?

# In[ ]:


plt.figure(figsize=(20, 7))
review_df.loc[review_df['comment'].notna()]['name'].value_counts()[:50].plot(kind='bar')
plt.xlabel('Board Game Name')
plt.ylabel('Review Count')
plt.title('Top 50 Most Reviewed Board Games')
plt.show()

