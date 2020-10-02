#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install flatten_json')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import re
import gc

#to make json in flatten object
from flatten_json import flatten
#!pip install flatten_json


#For plotting
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')

#Prediction related model
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Until fuction: line seperator
def print_dashes_and_ln():
    print('-'*100, '\n')
    
# Formatter to display all float format in 2 decimal format
#pd.options.display.float_format = '{:.2f}'.format

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Json file opening, parsing in strin and flattened after that data frame is creating.
f = open('/kaggle/input/pro-kabaddi-2014-2019/stats_match.json')
data = json.load(f)
f.close()
data_flattened = [flatten(d) for d in data]
#print(data_flattened)
match_df = pd.DataFrame(data_flattened)
match_df.head()


# # 1. Data Cleaning and manipulation

# In[ ]:


# getting total number of rows and column in the dataframe
def shape_of_dataframe(match_df):
    print(f" Shape of the dataframe = {match_df.shape}"); print_dashes_and_ln();
    totalrows=match_df.shape[0]
    print(f" Total number of rows in the dataset =  {totalrows}"); print_dashes_and_ln();
shape_of_dataframe(match_df)


# In[ ]:


# Drop unwanted columns
columns_to_drop = list(match_df.filter(regex="^(events|zones|match_detail_clock|match_detail_clock|match_detail_date|match_detail_start_time"
                                           +"|match_detail_gmtoffset|match_detail_status|match_detail_status|match_detail_result_outcome|match_detail_result_value|match_detail_player_of_the_match"
                                          +"|match_detail_series_parent_series|match_detail_venue|match_detail_stage|match_detail_group"
                                          +"|teams_home_team|teams_team_._state_of_play"
                                           +"|teams_team_._stats_points_all_out|teams_team_._stats_points_declare"
                                           +"|teams_team_._stats_points_raid_points|teams_team_._stats_points_tackle_points"
                                           +"|teams_team_._squad).*", axis = 1).columns)

match_df.drop(columns_to_drop, inplace=True, axis = 1)


# In[ ]:


# after dropping the required columns we have n columns in the data
shape_of_dataframe(match_df)


# In[ ]:


#Remove Columns with only One Value
match_df = match_df.loc[:,match_df.apply(pd.Series.nunique) > 1]


# In[ ]:


#Renaming of column name which help to shortane the column name with proper name.
match_df.rename(columns=lambda x: x.replace('match_detail_result', 'result'), inplace=True)
match_df.rename(columns=lambda x: x.replace('teams_team', 'team'), inplace=True)
match_df.rename(columns=lambda x: x.replace('match_detail', 'match'), inplace=True)
match_df.rename(columns=lambda x: x.replace('stats_raids', 'raid'), inplace=True)
match_df.rename(columns=lambda x: x.replace('stats_tackles', 'tackle'), inplace=True)


# In[ ]:


# After cleaning and manupulation dataframe will look like
match_df.head()


# In[ ]:


#Remove or substitute respective relevant value for nan and none with in a data frame
#1. Data frame columns containg null value
def columns_cointaing_null_value(match_df):
    match_df_contianing_null = match_df.columns[match_df.isna().any()].tolist()
    print(f'Column containing null values are: = {match_df_contianing_null}'); print_dashes_and_ln();
columns_cointaing_null_value(match_df)
#2. Fill NaN with the mean of the column
match_df['result_winning_team'] = match_df['result_winning_team'].fillna('Tie Match')
match_df['result_winning_team_id'] = match_df['result_winning_team_id'].fillna(0)
#3 Value in the both column is as shown below.
match_df_win_team_id = match_df[['result_winning_team','result_winning_team_id']]
print(match_df_win_team_id.head())
#4. recheck if there is any null value still persist.
columns_cointaing_null_value(match_df)


# In[ ]:


#Inforamtion of datatype of columns of data frame
match_df.info()


# In[ ]:


#data type of different columns with object
def list_of_object_dtype_columns(match_df):
    match_df_object_dtype = match_df.select_dtypes(include=['object']).columns.tolist()
    print(f'list of columns with datatype object = {match_df_object_dtype}'); print_dashes_and_ln();
list_of_object_dtype_columns(match_df)


# In[ ]:


#Converting float type column to int for better visualization.
#There some of columns which values are in float, objects etc but is needed integer type.
match_df['result_winning_team_id'] = match_df['result_winning_team_id'].astype(int)
#match time iso to date time
match_df['match_matchtime_iso']=pd.to_datetime(match_df['match_matchtime_iso'])
# Match Team Id to numeric
match_df['team_0_id']=pd.to_numeric(match_df['team_0_id'])
match_df['team_1_id']=pd.to_numeric(match_df['team_1_id'])

#Printing all object type columns names
list_of_object_dtype_columns(match_df)

#There is no need to further change object to other datatype
match_df.info()


# In[ ]:


# garbage collect (unused) object and delete all references that is no further used
gc.collect()


# ### Data is cleaned and relevent manupulation has been done.
# 
# #### ------------------------------------------------------------------------------------------------------------

# # Visualizing Data and EDA For all the features

# In[ ]:


# create plotting functions
def data_type(variable):
    if variable.dtype == np.int64 or variable.dtype == np.float64:
        return 'numerical'
    elif variable.dtype == 'category':
        return 'categorical'
    
def univariate(variable, stats=True):
    
    if data_type(variable) == 'numerical':
        sns.distplot(variable)
        if stats == True:
            print(variable.describe())
    
    elif data_type(variable) == 'categorical':
        sns.countplot(variable)
        if stats == True:
            print(variable.value_counts())
            
    else:
        print("Invalid variable passed: either pass a numeric variable or a categorical vairable.")
        
def bivariate(var1, var2):
    if data_type(var1) == 'numerical' and data_type(var2) == 'numerical':
        sns.regplot(var1, var2)
    elif (data_type(var1) == 'categorical' and data_type(var2) == 'numerical') or (data_type(var1) == 'numerical' and data_type(var2) == 'categorical'):        
        sns.boxplot(var1, var2)


# In[ ]:


# Checking corelation
def corr_graph(match_df):
    corr = match_df.corr()
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr)
    plt.show()
    return corr
corr_graph(match_df)


# In[ ]:


col_names = match_df.corr().columns.values

for col, row in (match_df.corr().abs() > 0.7).iteritems():
    print(col, col_names[row.values])


# # Adding Dummy Variables

# In[ ]:


# Finding corelated varibale with winning team
a = match_df[match_df.columns[:]].corr()['result_winning_team_id']


# In[ ]:


a1 = a[a >.1]
a2 = a[a < -.1]
a1


# In[ ]:


a2


# In[ ]:


# There is no futher deletion of column need to perform because each columns has its own importance.
# Neither need to add dummy variables.


# In[ ]:


#Univariate analysis in accordance to match_toss_winner
univariate(match_df.match_toss_winner)


# In[ ]:


#Univariate analysis in accordance to team_0_raid_successful
univariate(match_df.result_winning_team_id)


# In[ ]:


# Bivariate analysic with result_winning_team_id and match_toss_winner
bivariate(match_df.result_winning_team_id, match_df.match_toss_winner)


# In[ ]:


# Bivariate analysic with result_winning_team_id and match_toss_winner
bivariate(match_df.team_0_id, match_df.team_1_id)


# In[ ]:


# Bivariate analysic with team_0_score and team_1_score
bivariate(match_df.team_0_score, match_df.team_1_score)


# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1, 3, 1)
plt.title('Winning Team Distribution')
sns.distplot(match_df['result_winning_team_id'],color='green')

# subplot 2
plt.subplot(1, 3, 2)
sns.distplot(match_df['team_0_stats_all_outs'],color='blue')

# subplot 3l
plt.subplot(1, 3, 3)
sns.distplot(match_df['team_1_stats_all_outs'],color='red')


# In[ ]:


fig, ax = plt.subplots(figsize=(200,8))
width = len(match_df['result_winning_team'].unique()) + 6
fig.set_size_inches(width , 8)
ax=sns.countplot(data = match_df, x= 'result_winning_team') 


# In[ ]:


fig, ax = plt.subplots(figsize=(200,8))
width = len(match_df['team_0_raid_successful'].unique()) + 6
fig.set_size_inches(width , 8)
ax=sns.countplot(data = match_df, x= 'team_0_raid_successful') 


# In[ ]:


fig, ax = plt.subplots(figsize=(200,8))
width = len(match_df['team_1_raid_successful'].unique()) + 6
fig.set_size_inches(width , 8)
ax=sns.countplot(data = match_df, x= 'team_1_raid_successful') 


# In[ ]:



fig, ax = plt.subplots(figsize=(200,8))
width = len(match_df['result_winning_method'].unique()) + 6
fig.set_size_inches(width , 8)
ax=sns.countplot(data = match_df, x= 'result_winning_method')


# In[ ]:


season_7_teams = ['Dabang Delhi K.C.',
'Bengal Warriors',
'Haryana Steelers',
'U Mumba',
'Bengaluru Bulls',
'U.P. Yoddha',
'Jaipur Pink Panthers',
'Puneri Paltan',
'Patna Pirates',
'Gujarat Fortunegiants',
'Telugu Titans',
'Tamil Thalaivas']

df_teams_1 = match_df[match_df['team_0_name'].isin(season_7_teams)]
df_teams_2 = match_df[match_df['team_1_name'].isin(season_7_teams)]
df_teams = pd.concat((df_teams_1, df_teams_2))
df_teams.drop_duplicates()
df_teams.head()


# In[ ]:


#New column introduce as winner to decide predictions
df_teams['winner'] = np.where(df_teams.team_0_id==df_teams.result_winning_team_id,1, np.where(df_teams.team_1_id==df_teams.result_winning_team_id, 2,3))


# In[ ]:


df_teams = df_teams[['team_0_name','team_1_name','winner']]
df_teams.rename(columns={'team_0_name':'Team_1','team_1_name':'Team_2'}, inplace=True)
df_teams.head()


# In[ ]:


final_predict = pd.get_dummies(df_teams, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

X = final_predict.drop(['winner'], axis=1)
y = final_predict["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


#Model implemented to predit team data
rf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0) 
rf.fit(X_train, y_train)
score = rf.score(X_train, y_train)
score2 = rf.score(X_test, y_test)
print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))


# In[ ]:


#Extracting rank and total poins related data
#Json file opening, parsing in strin and flattened after that data frame is creating.
f = open('/kaggle/input/pro-kabaddi-2014-2019/stats_team.json')
data = json.load(f)
f.close()
data_flattened = [flatten(d) for d in data]
#print(data_flattened)
rank_df = pd.DataFrame(data_flattened)
#list(rank_df.columns)
rank_df = rank_df[['bio_team_id','bio_team_name','over_all_stats_rank','over_all_stats_points','over_all_stats_success_raids','over_all_stats_success_tackles','over_all_stats_super_raids','over_all_stats_super_tackles','over_all_stats_all_outs','over_all_stats_all_out_points']]
rank_df.head(23)


# In[ ]:


#Total matches will be 12 * 11 = 132
#team_standing = team for teama in season_7_teams for teamb in season_7_teams if teama != teamb
#team_standing = set(season_7_teams) & set(season_7_teams)
team_standing_a = []
team_standing_b = []
for teama in season_7_teams:
    for teamb in season_7_teams:
        if(teama != teamb):
            team_standing_a.append(teama)
            team_standing_b.append(teamb)
team_standing = pd.DataFrame({
    'Team_1': team_standing_a,
    'Team_2': team_standing_b})    
team_standing.head()


# In[ ]:



team_standing.insert(1, 'first_position', team_standing['Team_1'].map(rank_df.set_index('bio_team_name')['over_all_stats_rank']))
team_standing.insert(2, 'second_position', team_standing['Team_2'].map(rank_df.set_index('bio_team_name')['over_all_stats_rank']))
team_standing.tail()


# In[ ]:


pred_set = []
for index, row in team_standing.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winner': None})
    else:
        pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winner': None})
        
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
pred_set.head()


# In[ ]:


pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

missing_cols = set(final_predict.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final_predict.columns]


pred_set = pred_set.drop(['winner'], axis=1)
pred_set.head()


# In[ ]:


predictions = rf.predict(pred_set)
for i in range(team_standing.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 1:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    
    else:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print("")


# In[ ]:


def clean_and_predict(matches, ranking, final, rf):
    positions = []
    for match in matches:
        positions.append(ranking.loc[ranking['bio_team_name'] == match[0],'over_all_stats_rank'].iloc[0])
        positions.append(ranking.loc[ranking['bio_team_name'] == match[1],'over_all_stats_rank'].iloc[0])

    pred_set = []

    i = 0
    j = 0

    while i < len(positions):
        dict1 = {}

        if positions[i] < positions[i + 1]:
            dict1.update({'Team_1': matches[j][0], 'Team_2': matches[j][1]})
        else:
            dict1.update({'Team_1': matches[j][1], 'Team_2': matches[j][0]})

        pred_set.append(dict1)
        i += 2
        j += 1
        
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    pred_set = pred_set.drop(['winner'], axis=1)

    predictions = rf.predict(pred_set)
    for i in range(len(pred_set)):
        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 1:
            print("Winner: " + backup_pred_set.iloc[i, 1])
        else:
            print("Winner: " + backup_pred_set.iloc[i, 0])
        print("")


# In[ ]:


semi = [('Dabang Delhi K.C.', 'Bengal Warriors'),
            ('Haryana Steelers', 'Bengaluru Bulls')]
clean_and_predict(semi, rank_df, final_predict, rf)


# In[ ]:


#From the above prediction winner team is Bengal Warriors
final = [('Bengal Warriors','Haryana Steelers')]
clean_and_predict(final, rank_df, final_predict, rf)


# In[ ]:


# Predict the top team in the points table after the completion of the league matches. 
rank_df[rank_df.over_all_stats_points == max(rank_df.over_all_stats_points)].bio_team_name	


# In[ ]:


# Predict the team with the highest points for successful raids
rank_df[rank_df.over_all_stats_success_raids == max(rank_df.over_all_stats_success_raids)].bio_team_name	


# In[ ]:


# Predict the team with the highest points for successful tackles
rank_df[rank_df.over_all_stats_success_tackles == max(rank_df.over_all_stats_success_tackles)].bio_team_name	


# In[ ]:


# Predict the team with the highest super-performance total.
rank_df['highest_super'] = rank_df.over_all_stats_super_raids + rank_df.over_all_stats_super_tackles + rank_df.over_all_stats_all_outs-rank_df.over_all_stats_all_out_points
rank_df[rank_df.highest_super == max(rank_df.highest_super)].bio_team_name	

