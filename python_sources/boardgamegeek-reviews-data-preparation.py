#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import ast
from collections import Counter
from collections import defaultdict

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import math

plt.style.use('ggplot')

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


# # Repeated Columns Analysis
# 
# - IDs are present for every table (join condition)
# - Ranking and game detail tables have published year, rank, average, bayes average, users rated and thumbnail columns
# - Looking at Kaggle, the dataset was last updated on 2nd June, and looking at the number of users rated, the game detailed info table looks more updated

# In[ ]:


ranking_sub_df = ranking_df[['ID', 'Year', 'Rank', 'Average', 'Bayes average', 'Users rated', 'Thumbnail']]
ranking_sub_df.head()


# In[ ]:


detail_sub_df = detail_df[['id', 'yearpublished', 'Board Game Rank', 'average', 'bayesaverage', 'usersrated', 'thumbnail']]
detail_sub_df.head()


# In[ ]:


joined_df = ranking_sub_df.merge(detail_sub_df, left_on='ID', right_on='id', how='left')
joined_df.head(20)


# # View Columns

# In[ ]:


ranking_df.columns


# In[ ]:


detail_df.columns


# # Extract Features

# In[ ]:


output_df = detail_df.loc[: , ['averageweight', 'bayesaverage', 'description', 'id', 'image', 
                               'minage', 'minplayers', 'minplaytime', 'primary', 'thumbnail', 
                               'yearpublished']]
output_df.columns = ['complexity_score', 'bayes_average', 'description', 'id', 'image', 
                     'official_min_age', 'official_min_players', 'official_min_playtime', 'name', 'thumbnail', 
                     'year_published']
output_df.columns


# # Create Features for Board Game Success
# - Success is determined by us as above the 75th percentile

# In[ ]:


third_quartile = np.percentile(detail_df['bayesaverage'], [75])

print('3rd Quartile:', third_quartile)


# In[ ]:


output_df['is_success'] = detail_df['bayesaverage'].apply(lambda x, threshold: 1 if x > third_quartile else 0, args=[third_quartile])
output_df['is_success'].value_counts(normalize=True)


# In[ ]:


output_df.columns[-1]


# # Feature Engineering for Average Weight (Complexity of Game)

# # Success Rates Feature Creation for Artist, Designer and Publisher
# 
# Board Game Artist, Designer and Publisher success rates for each board game
# 
# Computed as $\frac{\text{Number of successful board games}}{\text{Number of board games}}$

# In[ ]:


def clean_names(name_lst):
    # Catch NaN
    if type(name_lst) is float:
        return []
    # Convert to list
    if type(name_lst) is str:
        name_lst = ast.literal_eval(name_lst)
    # Trim white space
    for idx, name in enumerate(name_lst):
        name_lst[idx] = name.strip()
    # Remove duplicates
    return list(set(name_lst))


# In[ ]:


def calculate_rate(name_lst, success_count_dict, overall_count_dict):
    success_count = 0
    total_count = 0
    for name in name_lst:
        success_count += success_count_dict[name]
        total_count += overall_count_dict[name]
    if total_count == 0:
        return 0
    return success_count / total_count


# In[ ]:


# Text cleaning
detail_df['boardgameartist'] = detail_df['boardgameartist'].apply(clean_names)

# Filter success rows
sub_df = detail_df.loc[output_df['is_success'] == 1]


# In[ ]:


success_count_dict = Counter(sub_df['boardgameartist'].sum())
overall_count_dict = Counter(detail_df['boardgameartist'].sum())

print(success_count_dict.most_common(10))
print()
print(overall_count_dict.most_common(10))


# In[ ]:


output_df['artist_success_rate'] = detail_df['boardgameartist'].apply(calculate_rate, 
                                                                      args=[success_count_dict, overall_count_dict])
output_df['artist_success_rate'].describe()


# In[ ]:


sns.violinplot(output_df['artist_success_rate'])
plt.show()


# In[ ]:


# Text cleaning
detail_df['boardgamedesigner'] = detail_df['boardgamedesigner'].apply(clean_names)

# Filter success rows
sub_df = detail_df.loc[output_df['is_success'] == 1]


# In[ ]:


success_count_dict = Counter(sub_df['boardgamedesigner'].sum())
overall_count_dict = Counter(detail_df['boardgamedesigner'].sum())

print(success_count_dict.most_common(10))
print()
print(overall_count_dict.most_common(10))


# In[ ]:


output_df['designer_success_rate'] = detail_df['boardgamedesigner'].apply(calculate_rate, 
                                                                          args=[success_count_dict, overall_count_dict])
output_df['designer_success_rate'].describe()


# In[ ]:


sns.violinplot(output_df['designer_success_rate'])
plt.show()


# In[ ]:


# Text cleaning
detail_df['boardgamepublisher'] = detail_df['boardgamepublisher'].apply(clean_names)

# Filter success rows
sub_df = detail_df.loc[output_df['is_success'] == 1]


# In[ ]:


success_count_dict = Counter(sub_df['boardgamepublisher'].sum())
overall_count_dict = Counter(detail_df['boardgamepublisher'].sum())

print(success_count_dict.most_common(10))
print()
print(overall_count_dict.most_common(10))


# In[ ]:


output_df['publisher_success_rate'] = detail_df['boardgamepublisher'].apply(calculate_rate, 
                                                                               args=[success_count_dict, overall_count_dict])
output_df['publisher_success_rate'].describe()


# In[ ]:


sns.violinplot(output_df['publisher_success_rate'])
plt.show()


# In[ ]:


output_df.columns[-3:]


# # One-Hot Encode Board Game Category

# In[ ]:


detail_df['boardgamecategory'] = detail_df['boardgamecategory'].replace(np.nan, '[]')
vectorizer = CountVectorizer(tokenizer=ast.literal_eval, lowercase=True, binary = True)
dummies_df = pd.DataFrame(vectorizer.fit_transform(detail_df['boardgamecategory']).toarray(), 
                          columns=vectorizer.get_feature_names())
dummies_df.columns = ['category_' + col_name.replace(' ', '_') for col_name in dummies_df.columns]

dummies_df.head()


# In[ ]:


output_df.shape, dummies_df.shape


# In[ ]:


output_df = pd.concat([output_df, dummies_df], axis=1)
output_df.head()


# In[ ]:


output_df.columns[-83:]


# # One-Hot Encode Board Game Mechanic

# In[ ]:


detail_df['boardgamemechanic'] = detail_df['boardgamemechanic'].replace(np.nan, '[]')
vectorizer = CountVectorizer(tokenizer=ast.literal_eval, lowercase=True, binary = True)
dummies_df = pd.DataFrame(vectorizer.fit_transform(detail_df['boardgamemechanic']).toarray(), 
                          columns=vectorizer.get_feature_names())
dummies_df.columns = ['mechanic_' + col_name.replace(' ', '_') for col_name in dummies_df.columns]

dummies_df.head()


# In[ ]:


output_df.shape, dummies_df.shape


# In[ ]:


output_df = pd.concat([output_df, dummies_df], axis=1)
output_df.head()


# In[ ]:


output_df.columns[-53:]


# # Categorise Max Players
# 
# - Very right skewed distribution
# - 90th percentile for max players is 8 - likely "safe" to categorise 8+ max players as 8

# In[ ]:


detail_df['maxplayers'].plot(kind='hist')
plt.show()


# In[ ]:


np.percentile(detail_df['maxplayers'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


# In[ ]:


# Trying out pd.cut()
pd.cut([1, 2, 3, 4, 5, 6, 7, 8, 9], bins=[0, 1, 2, 3, 4, 5, 6, 7, 100], labels=['1', '2', '3', '4', '5', '6', '7', '8'])


# In[ ]:


# # Max Players
# maxplayers_arr = detail_df['maxplayers'].values.tolist()
# # maxplayers_arr
# for i in range(len(maxplayers_arr)):
#     numplayers = maxplayers_arr[i]
#     if (numplayers >= 8):
#         maxplayers_arr[i] = "8+"
#     else:
#         maxplayers_arr[i] = str(numplayers)
# newfeature_df['new_maxplayers'] = maxplayers_arr
# dummydf_mpy = newfeature_df['new_maxplayers'].str.join(sep='').str.get_dummies()
# newfeature_df = pd.concat([newfeature_df, dummydf_mpy], axis=1, join='inner')
# print(newfeature_df)

# label = 8 is for 8+
output_df['official_max_players_categorised'] = pd.cut(detail_df['maxplayers'], 
                                                       bins=[detail_df['maxplayers'].min()-1, 1, 2, 3, 4, 5, 6, 7, detail_df['maxplayers'].max()+1], 
                                                       labels=['1', '2', '3', '4', '5', '6', '7', '8'])


# In[ ]:


output_df.columns[-1]


# # Categorise Max Playtime

# In[ ]:


# maxplaytime_arr = detail_df['maxplaytime'].values.tolist()
# df = pd.DataFrame()
# df['new_maxplaytime'] = maxplaytime_arr
# df

# df['num'] = 1
# df1 = df.groupby('new_maxplaytime').count()
# filter1 = df1['num'] > (0.01*17000)
# temp = df1.where(filter1).dropna()
# # temp2 = df1.where(filter2)
# print(temp)

# for i in range(len(maxplaytime_arr)):
#     playtime = int(maxplaytime_arr[i])
    
#     if (playtime <= 30):
#         maxplaytime_arr[i] = 1
#     elif (playtime > 30 and playtime <= 60):
#         maxplaytime_arr[i] = 2
#     elif (playtime > 60 and playtime <= 120):
#         maxplaytime_arr[i] = 3
#     elif (playtime > 120 and playtime <= 180):
#         maxplaytime_arr[i] = 4
#     elif (playtime > 180):
#         maxplaytime_arr[i] = 5

# newfeature_df['new_maxplaytime'] = maxplaytime_arr
# newfeature_df

output_df['official_max_playtime_categorised'] = pd.cut(detail_df['maxplaytime'], 
                                                        bins=[detail_df['maxplaytime'].min()-1, 
                                                              30, 60, 120, 180, 
                                                              detail_df['maxplaytime'].max()+1], 
                                                        labels=['1', '2', '3', '4', '5'])


# In[ ]:


output_df.columns[-1]


# # Average Suggested Language Dependence
# 
# - Scores are given to the language dependence
#     - 1 : No necessary in-game text
#     - 2 : Some necessary text - easily memorized or small crib sheet
#     - 3 : Moderate in-game text - needs crib sheet or paste ups
#     - 4 : Extensive use of text - massive conversion needed to be playable
#     - 5 : Unplayable in another language

# In[ ]:


# View one of the values
detail_df['suggested_language_dependence'][0]


# In[ ]:


def eval_ordered_dict(str_list):
    if isinstance(str_list, str):
        str_list = str_list.replace("OrderedDict","")
        return ast.literal_eval(str_list)
    return list()

def get_suggested_language_dependence(lst):
    if len(lst) == 0:
        return np.nan
    dictionary = defaultdict(None)
    for idx, l in enumerate(lst):
        dictionary[idx+1] = int(l[2][1])
    dependences, votes = list(zip(*dictionary.items()))
    return np.sum(np.array(dependences) * np.array(votes)) / np.sum(votes)


# In[ ]:


output_df['suggested_language_dependence_mean'] = detail_df['suggested_language_dependence']                                                             .apply(eval_ordered_dict)                                                             .apply(get_suggested_language_dependence)


# In[ ]:


output_df['suggested_language_dependence_mean'].head()


# In[ ]:


# 4k NaN values: Continuous variable - fill NA with mean
suggested_language_dependence_mean = output_df['suggested_language_dependence_mean'].mean()
print('Suggested Language Dependence Mean:', suggested_language_dependence_mean)

output_df['suggested_language_dependence_mean'] = output_df['suggested_language_dependence_mean'].fillna(suggested_language_dependence_mean)


# In[ ]:


detail_df['suggested_language_dependence'].isna().sum(), output_df['suggested_language_dependence_mean'].isna().sum()


# In[ ]:


output_df.columns[-1]


# # Weighted Average Suggested Num Players
# 
# Sum product of the number of players and the number of votes
# 
# Average sum product with the total number of votes for each number of players
# 
# Get the mean of the weighted averages

# In[ ]:


# View one of the values
detail_df['suggested_num_players'][0]


# In[ ]:


def get_suggested_num_players(str_list_SNP):
    iter_count = 0
    snp_df = pd.DataFrame()
    snp_dict = {"suggested_num_players_feat":[]}
    
    for items in str_list_SNP:
        list_SNP = eval_ordered_dict(items)
        final_dict = {}
        if isinstance(list_SNP[0],list):
            for tup in list_SNP:
                try:
                    num_players = float(tup[0][1])
                except:
                    num_players = float(tup[0][1][0]) + 1
                result_list = tup[1][1]
                best_votes = result_list[0][1][1]
                rec_votes = result_list[1][1][1]
                not_rec_votes = result_list[2][1][1]
                if final_dict and final_dict["id"] == detail_df.iloc[iter_count]["id"]:
                    n_votes = int(best_votes) + int(rec_votes) + int(not_rec_votes)
                    if n_votes == 0:
                        weighted_average = None
                    else:
                        weighted_average = float((int(best_votes)*2 + int(rec_votes) + int(not_rec_votes)*(-1))/n_votes)
                    if weighted_average and final_dict["votes_weighted_average"][0]:
                        if weighted_average == final_dict["votes_weighted_average"][0]:
                            final_dict["votes_weighted_average"].append(weighted_average)
                            final_dict["num_players"].append(num_players)
                        elif final_dict["votes_weighted_average"][0] < weighted_average:
                            final_dict["votes_weighted_average"] = [weighted_average]
                            final_dict["num_players"] = [num_players]
                    elif weighted_average:
                        final_dict["votes_weighted_average"] = [weighted_average]
                        final_dict["num_players"] = [num_players]
                else:
                    final_dict["id"] = detail_df.iloc[iter_count]["id"]
                    final_dict["num_players"] = [num_players]
                    n_votes = int(best_votes) + int(rec_votes) + int(not_rec_votes)
                    if n_votes != 0:
                        final_dict["votes_weighted_average"] = [(int(best_votes)*2 + int(rec_votes) + int(not_rec_votes)*(-1))/n_votes]
                    else:
                        final_dict["votes_weighted_average"] = [None]
        else:
            if len(list_SNP)==2:
                try:
                    num_players = float(list_SNP[0][1])
                except:
                    num_players = float(list_SNP[0][1][0]) + 1
                best_votes = int(list_SNP[1][1][0][1][1])
                rec_votes = int(list_SNP[1][1][1][1][1])
                not_rec_votes = int(list_SNP[1][1][2][1][1])
                if final_dict and final_dict["id"] == detail_df.iloc[iter_count]["id"]:
                    n_votes = best_votes + rec_votes + not_rec_votes
                    if n_votes == 0:
                        weighted_average = None
                    else:
                        weighted_average = float((int(best_votes)*2 + int(rec_votes) + int(not_rec_votes)*(-1))/n_votes)
                    if weighted_average == final_dict["votes_weighted_average"][0]:
                        if weighted_average in final_dict["votes_weighted_average"]:
                            final_dict["num_players"].append(num_players)
                            final_dict["votes_weighted_average"].append(weighted_average)
                        elif final_dict["votes_weighted_average"][0] < weighted_average:
                            final_dict["votes_weighted_average"] = [weighted_average]
                            final_dict["num_players"] = [num_players]
                    elif weighted_average:
                        final_dict["votes_weighted_average"] = [weighted_average]
                        final_dict["num_players"] = [num_players]
                    else:
                        final_dict["votes_weighted_average"].append(weighted_average)
                        final_dict["num_players"].append(num_players)
                else:
                    final_dict["id"] = detail_df.iloc[iter_count]["id"]
                    final_dict["num_players"] = [num_players]
                    n_votes = int(best_votes) + int(rec_votes) + int(not_rec_votes)
                    if n_votes != 0:
                        final_dict["votes_weighted_average"] = [(int(best_votes)*2 + int(rec_votes) + int(not_rec_votes)*(-1))/n_votes]
                    else:
                        final_dict["num_players"] = [None]
                        final_dict["votes_weighted_average"] = [None]
            else:
                final_dict["id"] = detail_df.iloc[iter_count]["id"]
                try:
                    num_players = float(list_SNP[0][1])
                except:
                    num_players = float(list_SNP[0][1][0]) + 1
                final_dict["num_players"] = [None]
                final_dict["votes_weighted_average"] = [None]
        iter_count += 1
        num_players = final_dict["num_players"]
        votes_weighted_average = final_dict["votes_weighted_average"]
        if len(num_players) > 1 and len(votes_weighted_average) > 1:
            counter = 0
            for val in votes_weighted_average:
                if val == None or val == 0.0:
                    del num_players[counter]
                    votes_weighted_average.remove(val)
                else:
                    counter+=1
            final_dict["num_players"] = num_players
            final_dict["votes_weighted_average"] = votes_weighted_average
        elif not votes_weighted_average[0]:
            final_dict["num_players"] = [None]
        if final_dict["num_players"][0]:
            final_dict["mean_num_players"] = np.mean(final_dict["num_players"])
            final_dict["mean_votes_weighted_average"] = np.mean(final_dict["votes_weighted_average"])
        else:
            final_dict["mean_num_players"] = None
            final_dict["mean_votes_weighted_average"] = None
        snp_dict["suggested_num_players_feat"].append(final_dict)
    return snp_dict

snp_dict = get_suggested_num_players(detail_df['suggested_num_players'])
sugg_num_players_df = pd.DataFrame(snp_dict["suggested_num_players_feat"])
sugg_num_players_df.head()


# In[ ]:


output_df['suggested_num_players_weighted_average'] = sugg_num_players_df['mean_num_players']

# Free memory
del sugg_num_players_df


# In[ ]:


detail_df['suggested_num_players'].isna().sum(), output_df['suggested_num_players_weighted_average'].isna().sum()


# In[ ]:


# Normalise suggested players wrt min/max players: >0.5 suggested closer to max. <0.5 suggested closer to min
numerator = (output_df['suggested_num_players_weighted_average'] - output_df['official_min_players']).mean()
denominator = (output_df['official_max_players_categorised'].astype(float) - output_df['official_min_players']).mean()
percentile_from_suggested_min_players = numerator/denominator
print('Percentile from suggested min players', percentile_from_suggested_min_players)


# In[ ]:


# if min max player info available, take midpoint round up
minmax_avg = output_df['official_min_players'] +                 (output_df['official_max_players_categorised'].astype(float) - output_df['official_min_players']) *                 percentile_from_suggested_min_players
minmax_avg = minmax_avg.apply(round)

# if NaN then fill with above values
output_df['suggested_num_players_weighted_average'] = output_df['suggested_num_players_weighted_average'].fillna(minmax_avg)
output_df['suggested_num_players_weighted_average'].describe()


# In[ ]:


detail_df['suggested_num_players'].isna().sum(), output_df['suggested_num_players_weighted_average'].isna().sum()


# In[ ]:


output_df.columns[-1]


# # Average Suggested Player Age

# In[ ]:


# View one of the values
detail_df['suggested_playerage'][0]


# In[ ]:


def get_suggested_playerage(str_list_SPA):
    spa_df = pd.DataFrame()
    counter = 0
    for items in str_list_SPA:
        new_df = {}
        new_df["id"] = detail_df["id"].iloc[counter]
        if isinstance(items,str):
            list_SPA = eval_ordered_dict(items)
            new_df["2"] = int(list_SPA[0][1][1])
            new_df["3"] = int(list_SPA[1][1][1])
            new_df["4"] = int(list_SPA[2][1][1])
            new_df["5"] = int(list_SPA[3][1][1])
            new_df["6"] = int(list_SPA[4][1][1])
            new_df["8"] = int(list_SPA[5][1][1])
            new_df["10"] = int(list_SPA[6][1][1])
            new_df["12"] = int(list_SPA[7][1][1])
            new_df["14"] = int(list_SPA[8][1][1])
            new_df["16"] = int(list_SPA[9][1][1])
            new_df["18"] = int(list_SPA[10][1][1])
            new_df["21+"] = int(list_SPA[11][1][1])
        else:
            new_df["2"] = None
            new_df["3"] = None
            new_df["4"] = None
            new_df["5"] = None
            new_df["6"] = None
            new_df["8"] = None
            new_df["10"] = None
            new_df["12"] = None
            new_df["14"] = None
            new_df["16"] = None
            new_df["18"] = None
            new_df["21+"] = None
        counter += 1       

        spa_df = spa_df.append(new_df,ignore_index=True)
    return spa_df

spa_df = get_suggested_playerage(detail_df['suggested_playerage'])
spa_df.head()


# In[ ]:


spa_df["n"] = spa_df[['2', '3', '4', '5', '6', '8', '10', '12', '14', '16', '18', '21+']].sum(axis=1)
spa_df["total_n"] = (spa_df["2"]*2 + 
                     spa_df["3"]*3 + 
                     spa_df["4"]*4 + 
                     spa_df["5"]*5 + 
                     spa_df["6"]*6 + 
                     spa_df["8"]*8 + 
                     spa_df["10"]*10 + 
                     spa_df["12"]*12 + 
                     spa_df["14"]*14 + 
                     spa_df["16"]*16 + 
                     spa_df["18"]*18 + 
                     spa_df["21+"]*21)
spa_df.head()


# In[ ]:


output_df["suggested_player_age_mean"] = spa_df.apply(lambda r : r["total_n"]/r["n"] if r["n"] > 0 else None, axis=1)

# Free memory
del spa_df


# In[ ]:


# if NaN then just use the official value
output_df['suggested_player_age_mean'] = output_df['suggested_player_age_mean'].fillna(detail_df['minage'])


# In[ ]:


detail_df['suggested_playerage'].isna().sum(), output_df["suggested_player_age_mean"].isna().sum()


# In[ ]:


output_df.columns[-1]


# # Check and Reorder Columns

# In[ ]:


# Check rows for NA
output_df.isna().sum().loc[output_df.isna().sum() > 0]


# In[ ]:


columns = list(filter(lambda col_name: not col_name.startswith('category_') and 
                      not col_name.startswith('mechanic_'), output_df.columns)) + \
            list(filter(lambda col_name: col_name.startswith('category_'), output_df.columns)) + \
            list(filter(lambda col_name: col_name.startswith('mechanic_'), output_df.columns))
columns[:5], columns[-5:]


# In[ ]:


output_df.columns = columns
output_df.head()


# # Write CSV

# In[ ]:


output_df.to_csv('cleaned_games_detailed_info.csv', index=False)

