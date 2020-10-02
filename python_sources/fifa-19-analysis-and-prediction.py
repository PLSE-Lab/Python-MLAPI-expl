#!/usr/bin/env python
# coding: utf-8

# This notebook contains the analysis and visualisations for the FIFA 19 dataset.
# Also, the overall score of a player is predicted using various regression algorithms, ensemble algorithms and neural network.
# 
# The kernels from which references have been taken are:
# * https://www.kaggle.com/shriramganesh/awards-section-how-to-fill-nan-values/data
# * https://www.kaggle.com/nitindatta/fifa-in-depth-analysis-with-linear-regression
# * https://www.kaggle.com/rupavj/fifa-17-detailed-analysis/notebook
# 
# **Table of contents**
# 1. Information about the data
# 2. Data cleaning 
# 3. Grouping similar skills together
# 4. Data analysis and visualisation
# 5. Club level analysis
# 6. National level analysis
# 7. Preparing data for modelling
# 8. Modelling

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import time 

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 1. **Information about the data**

# In[ ]:


df = pd.read_csv('../input/data.csv')
print(df.shape)


# In[ ]:


df.head()


# Number of null values in each column

# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# 2. **Data cleaning**

# Function for removing 'lbs' from the 'weight' column

# In[ ]:


def weight_correction(df):
    try:
        value = float(df[:-3])
    except:
        value = 0
    return value
df['Weight'] = df.Weight.apply(weight_correction)


# In[ ]:


df.Weight = pd.to_numeric(df.Weight)


# In[ ]:


df.Weight = df.Weight.replace(0, np.nan)


# Function for converting values in 'value' and 'wage' columns to numeric type

# In[ ]:


def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]
        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

df['Value'] = df['Value'].apply(value_to_int)
df['Wage'] = df['Wage'].apply(value_to_int)

df.Value = df.Value.replace(0, np.nan)
df.Wage = df.Wage.replace(0, np.nan)


# Filling the NaN values

# In[ ]:


df.Weight.isna().sum()


# In[ ]:


df.Weight.mean()


# According to https://www.sportsrec.com/5464445/the-ideal-weight-for-a-soccer-player,
# 
# The normal weight range for a player with a height of 5 feet 9 inches is 136 to 169 pounds.
# Since the mean weight of the player is 165 pounds in our data and it gels with the global data, we could set the mean weight to fill the null values.

# In[ ]:


df['Weight'].fillna(df.Weight.mean(), inplace = True)


# In[ ]:


df.Height.isna().sum()


# In[ ]:


plt.figure(figsize = (20, 10))
sns.countplot(x = 'Height', data = df)
plt.show()


# According to https://www.sportsrec.com/5464445/the-ideal-weight-for-a-soccer-player,
# 
# The average height of players is 5 feet, 11 and 1/2 inches. Also, from the data, we find that most of the players have their height between 5 feet, 9 inches and 6 feet, 1 inch. So, we can fill the null values in the 'height' column with 5 feet, 11 inches.

# In[ ]:


df['Height'].fillna("5'11", inplace = True)


# In[ ]:


wf_missing = df['Weak Foot'].isna()
wf_missing.sum()


# In[ ]:


weak_foot_prob = df['Weak Foot'].value_counts(normalize = True)
weak_foot_prob


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[wf_missing, 'Weak Foot'] = np.random.choice(weak_foot_prob.index, size = wf_missing.sum(), p = weak_foot_prob.values)


# In[ ]:


pf_missing = df['Preferred Foot'].isna()
pf_missing.sum()


# In[ ]:


foot_distribution = df['Preferred Foot'].value_counts(normalize = True)
foot_distribution


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[pf_missing, 'Preferred Foot'] = np.random.choice(foot_distribution.index, size = pf_missing.sum(), p = foot_distribution.values)


# In[ ]:


fp_missing = df.Position.isna()
fp_missing.sum()


# In[ ]:


position_prob = df.Position.value_counts(normalize = True)
position_prob 


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[fp_missing, 'Position'] = np.random.choice(position_prob.index, p = position_prob.values, size = fp_missing.sum())


# In[ ]:


fs_missing = df['Skill Moves'].isna()
fs_missing.sum()


# In[ ]:


skill_moves_prob = df['Skill Moves'].value_counts(normalize=True)
skill_moves_prob


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[fs_missing, 'Skill Moves'] = np.random.choice(skill_moves_prob.index, p = skill_moves_prob.values, size = fs_missing.sum())


# In[ ]:


bt_missing = df['Body Type'].isna()
bt_missing.sum()


# In[ ]:


bt_prob = df['Body Type'].value_counts(normalize = True)
bt_prob


# 'Neymar', 'Messi', 'Shaqiri', 'Akinfenwa', 'Courtois' are definitely not body types but the names of football players. So, we will fill the null values with 'normal' and 'lean'.

# In[ ]:


df.loc[bt_missing, 'Body Type'] = np.random.choice(['Normal', 'Lean'], p = [.63,.37], size = bt_missing.sum())


# In[ ]:


wage_missing = df['Wage'].isna()
wage_missing.sum()


# In[ ]:


wage_prob = df.Wage.value_counts(normalize = True)
wage_prob


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[wage_missing, 'Wage'] = np.random.choice(wage_prob.index, p = wage_prob.values, size = wage_missing.sum())


# In[ ]:


wr_missing = df['Work Rate'].isna()
wr_missing.sum()


# In[ ]:


wr_prob = df['Work Rate'].value_counts(normalize=True)
wr_prob


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[wr_missing, 'Work Rate'] = np.random.choice(wr_prob.index, p = wr_prob.values, size = wr_missing.sum())


# In[ ]:


ir_missing = df['International Reputation'].isna()
ir_missing.sum()


# In[ ]:


ir_prob = df['International Reputation'].value_counts(normalize = True)
ir_prob


# We will fill the null values with the same probability distribution

# In[ ]:


df.loc[ir_missing, 'International Reputation'] = np.random.choice(ir_prob.index, p = ir_prob.values, size = ir_missing.sum())


# We will fill the other null values in the remaining important columns containing float values using the mean.

# In[ ]:


# filling the missing value for the continous variables for proper data visualization

df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True)
df['Volleys'].fillna(df['Volleys'].mean(), inplace = True)
df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)
df['Curve'].fillna(df['Curve'].mean(), inplace = True)
df['FKAccuracy'].fillna(df['FKAccuracy'].mean(), inplace = True)
df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True)
df['BallControl'].fillna(df['BallControl'].mean(), inplace = True)
df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(), inplace = True)
df['Finishing'].fillna(df['Finishing'].mean(), inplace = True)
df['Crossing'].fillna(df['Crossing'].mean(), inplace = True)
df['Acceleration'].fillna(df['Acceleration'].mean(), inplace = True)
df['SprintSpeed'].fillna(df['SprintSpeed'].mean(), inplace = True)
df['Agility'].fillna(df['Agility'].mean(), inplace = True)
df['Reactions'].fillna(df['Reactions'].mean(), inplace = True)
df['Balance'].fillna(df['Balance'].mean(), inplace = True)
df['ShotPower'].fillna(df['ShotPower'].mean(), inplace = True)
df['Jumping'].fillna(df['Jumping'].mean(), inplace = True)
df['Stamina'].fillna(df['Stamina'].mean(), inplace = True)
df['Strength'].fillna(df['Strength'].mean(), inplace = True)
df['LongShots'].fillna(df['LongShots'].mean(), inplace = True)
df['Aggression'].fillna(df['Aggression'].mean(), inplace = True)
df['Interceptions'].fillna(df['Interceptions'].mean(), inplace = True)
df['Positioning'].fillna(df['Positioning'].mean(), inplace = True)
df['Vision'].fillna(df['Vision'].mean(), inplace = True)
df['Penalties'].fillna(df['Penalties'].mean(), inplace = True)
df['Composure'].fillna(df['Composure'].mean(), inplace = True)
df['Marking'].fillna(df['Marking'].mean(), inplace = True)
df['StandingTackle'].fillna(df['StandingTackle'].mean(), inplace = True)
df['SlidingTackle'].fillna(df['SlidingTackle'].mean(), inplace = True)


# We will fill the null values in 'loaned from' as 'none' and in 'club' as 'no club'

# In[ ]:


df['Loaned From'].fillna('None', inplace = True)
df['Club'].fillna('No Club', inplace = True)


# We will fill the null values in the other remaining columns as 0

# In[ ]:


df.fillna(0, inplace = True)


# 3. **Group similar skills together**

# Here, we are grouping the skills together and generalizing them to 8 categories.
# We do this so that we could analyze the players better and positon them accordingly.

# In[ ]:


def defending(df):
    return int(round((df[['Marking', 'StandingTackle', 
                               'SlidingTackle']].mean()).mean()))

def general(df):
    return int(round((df[['HeadingAccuracy', 'Dribbling', 'Curve', 
                               'BallControl']].mean()).mean()))

def mental(df):
    return int(round((df[['Aggression', 'Interceptions', 'Positioning', 
                               'Vision','Composure']].mean()).mean()))

def passing(df):
    return int(round((df[['Crossing', 'ShortPassing', 
                               'LongPassing']].mean()).mean()))

def mobility(df):
    return int(round((df[['Acceleration', 'SprintSpeed', 
                               'Agility','Reactions']].mean()).mean()))
def power(df):
    return int(round((df[['Balance', 'Jumping', 'Stamina', 
                               'Strength']].mean()).mean()))

def rating(df):
    return int(round((df[['Potential', 'Overall']].mean()).mean()))

def shooting(df):
    return int(round((df[['Finishing', 'Volleys', 'FKAccuracy', 
                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))


# Adding these categories to the data

# In[ ]:


df['Defending'] = df.apply(defending, axis = 1)
df['General'] = df.apply(general, axis = 1)
df['Mental'] = df.apply(mental, axis = 1)
df['Passing'] = df.apply(passing, axis = 1)
df['Mobility'] = df.apply(mobility, axis = 1)
df['Power'] = df.apply(power, axis = 1)
df['Rating'] = df.apply(rating, axis = 1)
df['Shooting'] = df.apply(shooting, axis = 1)


# 4. **Data analysis and visualisation**

# Creating a dataframe which will be used later to compare players across different clubs and countries.

# In[ ]:


players = df[['Name', 'Defending', 'General', 'Mental', 'Passing',
                'Mobility', 'Power', 'Rating', 'Shooting', 'Age',
                'Nationality', 'Club']]


# Representing skill moves of players using countplot

# In[ ]:


plt.figure(figsize = (10, 10))
ax = sns.countplot(x = 'Skill Moves', data = df, palette = 'bright')
ax.set_title(label = 'Count of players on the basis of their skill moves', fontsize = 20)
ax.set_xlabel(xlabel = 'Rating of skill moves', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()


# Representing share of weak foot of players using pie chart

# In[ ]:


labels = ['3', '2', '4', '5', '1']
sizes = df['Weak Foot'].value_counts()
plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(sizes, labels = labels)
plt.title('Distribution of players on the basis of their weak foot rating', fontsize = 20)
plt.legend()
plt.show()


# Representing share of preferred foot of players using countplot

# In[ ]:


plt.figure(figsize = (10, 10))
ax = sns.countplot(x = 'Preferred Foot', data = df, palette = 'deep')
ax.set_title(label = 'Count of players on the basis of their preferred foot', fontsize = 20)
ax.set_xlabel(xlabel = 'Preferred foot', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()


# Representing share of international reputation of players using pie chart

# In[ ]:


labels = ['1', '2', '3', '4', '5']
sizes = df['International Reputation'].value_counts()
explode = [0.1, 0.2, 0.3, 0.7, 0.9]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(sizes, labels = labels, explode = explode)
plt.title('Distribution of the international reputation of the players', fontsize = 20)
plt.legend()
plt.show()


# Representing wage range of players using distplot

# In[ ]:


plt.rcParams['figure.figsize'] = (10, 10)
sns.distplot(df['Wage'], color = 'blue')
plt.xlabel('Wage Range for Players', fontsize = 16)
plt.ylabel('Count of the Players', fontsize = 16)
plt.title('Distribution of Wages of Players', fontsize = 20)
plt.show()


# Inference-Most of the players have extremely low wages and only a few of the top players have wages above $100000

# Representing number of players in different positions using countplot

# In[ ]:


plt.figure(figsize = (20, 10))
plt.style.use('fivethirtyeight')
ax = sns.countplot('Position', data = df, palette = 'Reds_r')
ax.set_xlabel(xlabel = 'Different positions in football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
ax.set_title(label = 'Count of players on the basis of position', fontsize = 20)
plt.show()


# Representing distribution of work rate of players using countplot

# In[ ]:


plt.figure(figsize = (20, 10))
plt.style.use('fast')
sns.countplot(x = 'Work Rate', data = df, palette = 'hls')
plt.title('Count of players on the basis of work rate', fontsize = 20)
plt.xlabel('Work rates', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.show()


# Representing speciality scores of players using distplot

# In[ ]:


x = df.Special
plt.figure(figsize = (20, 10))
plt.style.use('tableau-colorblind10')
ax = sns.distplot(x, bins = 50, kde = True, color = 'g')
ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)
ax.set_ylabel(ylabel = 'Count',fontsize = 16)
ax.set_title(label = 'Count of players on the basis of their speciality score', fontsize = 20)
plt.show()


# Representing potential score of players using distplot

# In[ ]:


x = df.Potential
plt.figure(figsize = (20, 10))
plt.style.use('seaborn-paper')
ax = sns.distplot(x, bins = 50, color = 'y')
ax.set_xlabel(xlabel = "Player\'s potential scores", fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
ax.set_title(label = 'Count of players on the basis of potential scores', fontsize = 20)
plt.show()


# Representing overall score of players using distplot

# In[ ]:


sns.set(style = "dark", palette = "deep", color_codes = True)
x = df.Overall
plt.figure(figsize = (20, 10))
plt.style.use('ggplot')
ax = sns.distplot(x, bins = 50, color = 'r')
ax.set_xlabel(xlabel = "Player\'s Scores", fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
ax.set_title(label = 'Count of players on the basis of their overall scores', fontsize = 20)
plt.show()


# Plotting age against potential

# In[ ]:


plt.style.use('fast')
sns.jointplot(x = 'Age', y = 'Potential', data = df)
plt.show()


# Inference-As expected, the potential of players is greatest during their 20s and decrease with increase in age.

# Plotting special against overall

# In[ ]:


sns.jointplot(x = 'Special', y = 'Overall', data = df, joint_kws={'color':'orange'}, marginal_kws={'color':'blue'})
plt.show()


# Inference-Overall rating has a high correlation with special rating.

# Representing number of players from different countries using barplot

# In[ ]:


plt.style.use('dark_background')
df['Nationality'].value_counts().plot.bar(color = 'orange', figsize = (30, 15))
plt.title('Different Nations Participating in FIFA 2019', fontsize = 20)
plt.xlabel('Name of The Country', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 15)
plt.show()


# Inference-The most number of players are from top European and South American footballing nations like England, Germany, Spain, Argentina, France, Brazil etc.

# The top 4 features of players according to different positions

# In[ ]:


player_features = (
    'Acceleration', 'Aggression', 'Agility', 
    'Balance', 'BallControl', 'Composure', 
    'Crossing', 'Dribbling', 'FKAccuracy', 
    'Finishing', 'GKDiving', 'GKHandling', 
    'GKKicking', 'GKPositioning', 'GKReflexes', 
    'HeadingAccuracy', 'Interceptions', 'Jumping', 
    'LongPassing', 'LongShots', 'Marking', 'Penalties', 
    'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
    'Finishing', 'Crossing', 'SprintSpeed', 'Reactions',
    'ShotPower', 'Stamina', 'Strength',
    'Positioning', 'StandingTackle', 'SlidingTackle'
)

from math import pi
idx = 1
plt.style.use('seaborn-bright')
plt.figure(figsize = (25,45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(4))
    
    # number of variable
    categories = top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(9, 3, idx, polar = True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color = 'white', size = 10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color = "white", size = 9)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth = 1, linestyle = 'solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha = 0.1)
    
    plt.title(position_name, size = 10, y = 1.1)
    
    idx += 1


# Radar charts for a few of the top players
# 
# Credit:https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way

# Representing player's abilities on a few important parameters

# In[ ]:


labels = np.array(['Acceleration', 'Strength', 'Finishing',  
    'LongPassing', 'Penalties', 
    'ShortPassing', 'Volleys',
    'Finishing', 'Crossing', 'SprintSpeed',
    'ShotPower'])

stats = df.loc[0, labels].values
stats_1 = df.loc[1, labels].values
stats_2 = df.loc[2, labels].values


# In[ ]:


angles = np.linspace(0, 2 * np.pi, len(labels), endpoint = False)

# close the plot
stats = np.concatenate((stats, [stats[0]]))
stats_1 = np.concatenate((stats_1, [stats_1[0]]))
stats_2 = np.concatenate((stats_2, [stats_2[0]]))

angles = np.concatenate((angles, [angles[0]]))


# Radar chart for Lionel Messi

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, polar = True)
ax.plot(angles, stats, 'o-', linewidth = 2)
ax.fill(angles, stats, alpha = 0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title([df.loc[0, 'Name']], position = (0, 1))# Changing the 1st argument in df.loc will give the radar charts for different players
ax.grid(True)


# Radar chart for Cristiano Ronaldo

# In[ ]:


fig = plt.figure()
ax_1 = fig.add_subplot(111, polar = True)
ax_1.plot(angles, stats_1, 'o-', linewidth = 2)
ax_1.fill(angles, stats_1, alpha = 0.25)
ax_1.set_thetagrids(angles * 180/np.pi, labels)
ax_1.set_title([df.loc[1, 'Name']], position = (0, 1))
ax_1.grid(True)


# Radar chart for Neymar Jr.

# In[ ]:


fig = plt.figure()
ax_2 = fig.add_subplot(111, polar = True)
ax_2.plot(angles, stats_2, 'o-', linewidth = 2)
ax_2.fill(angles, stats_2, alpha = 0.25)
ax_2.set_thetagrids(angles * 180/np.pi, labels)
ax_2.set_title([df.loc[2, 'Name']], position = (0, 1))
ax_2.grid(True)


# Plotting overall rating against international reputation

# In[ ]:


plt.figure(figsize = (10, 10))
plt.style.use('seaborn-darkgrid')
plt.scatter(df['Overall'], df['International Reputation'])
plt.xlabel('Overall Ratings', fontsize = 16)
plt.ylabel('International Reputation', fontsize = 16)
plt.title('Ratings vs Reputation', fontsize = 20)
plt.show()


# There is definitely a positive correlation between international reputation and overall rating but surprisingly there are many players rated above 75 who have an international reputation of only 1.

# Players with maximum potential grouped by position

# In[ ]:


df.loc[df.groupby(df['Position'])['Potential'].idxmax()][['Name', 'Position', 'Overall', 'Potential', 'Age', 'Nationality', 'Club']]


# Players with maximum overall grouped by position

# In[ ]:


df.loc[df.groupby(df['Position'])['Overall'].idxmax()][['Name', 'Position', 'Overall', 'Age', 'Nationality', 'Club']]


# Plotting mean of overall, age-wise

# In[ ]:


df.groupby('Age')['Overall'].mean().plot(figsize = (20, 10))
plt.xlabel('Age', fontsize = 16)
plt.ylabel('Mean', fontsize = 16)
plt.title('Mean of overall, age-wise', fontsize = 20)
plt.show()


# Inference-As expected, the overall is highest during the 20s and starts decreasing with age. However, we see that suddenly it starts increasing around the age of 45, this might be due to outliers and so we will investigate further.

# In[ ]:


df['Age'].value_counts()


# In[ ]:


plt.style.use('seaborn-deep')
plt.figure(figsize = (20, 10))
sns.countplot(x = 'Age', data = df)
plt.xlabel('Age', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.title('Count of players, age-wise', fontsize = 20)
plt.show()


# As we can observe, there are very few players above the age of 40 and this might be the reason for the anomaly observed in the overall mean vs. age plot.

# In[ ]:


df[df['Age'] > 40][['Name', 'Overall', 'Age', 'Nationality']]


# As we can see, the outliers are the reason for the high mean of players aged around 45.

# The 15 youngest players of the dataset

# In[ ]:


df.sort_values('Age', ascending = True)[['Name', 'Age', 'Club', 'Nationality']].head(15)


# The 15 oldest players of the dataset

# In[ ]:


df.sort_values('Age', ascending = False)[['Name', 'Age', 'Club', 'Nationality']].head(15)


# The top 10 left-footed footballers according to overall rating

# In[ ]:


df[df['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)


# The top 10 right-footed footballers according to overall rating

# In[ ]:


df[df['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)


# 5. **Club level analysis**

# The top clubs on the basis of average overall rating

# In[ ]:


d = {'Overall': 'Average_Rating'}
best_overall_club_df = df.groupby('Club').agg({'Overall' : 'mean'}).rename(columns = d)
clubs = best_overall_club_df.Average_Rating.nlargest(5).index
print(clubs)


# The top clubs on the basis of attack

# In[ ]:


attck_list = ['Shooting', 'Power', 'Passing']

best_attack_df = players.groupby('Club')[attck_list].sum().sum(axis = 1)
clubs = best_attack_df.nlargest(6).index
print(clubs)


# 'No Club' is on top as more than 200 players have not been assigned clubs which is much higher than the number of players in any club and hence the sum of attack of those players would be higher than that of any club.

# The top clubs on the basis of defence

# In[ ]:


best_defense_df = players.groupby('Club')['Defending'].sum()
clubs = best_defense_df.nlargest(6).index
print(clubs)


# The same reason as above for 'No Club' being at the top.

# Finding the clubs with the most number of players

# In[ ]:


df['Club'].value_counts().head(15)


# Comparing the top clubs on the basis of a few important parameters

# In[ ]:


some_clubs = ('Manchester United', 'Arsenal', 'Juventus', 'Paris Saint-Germain', 'Napoli', 'Manchester City',
             'Tottenham Hotspur', 'FC Barcelona', 'Inter', 'Chelsea', 'Real Madrid', 'Borussia Dortmund', 'Liverpool', 'Roma', 'Ajax')


# Representing distribution of overall rating in top clubs

# In[ ]:


df_clubs = df.loc[df['Club'].isin(some_clubs) & df['Overall']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = df_clubs['Club'], y = df_clubs['Overall'], palette = 'rocket')
ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)
ax.set_ylabel(ylabel = 'Overall score', fontsize = 16)
ax.set_title(label = 'Distribution of overall score in different popular clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# Representing distribution of age in top clubs

# In[ ]:


df_club = df.loc[df['Club'].isin(some_clubs) & df['Age']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = 'Club', y = 'Age', data = df_club, palette = 'magma')
ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)
ax.set_ylabel(ylabel = 'Distribution', fontsize = 16)
ax.set_title(label = 'Distribution of ages in some popular clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# Representing distribution of wages in top clubs

# In[ ]:


df_club = df.loc[df['Club'].isin(some_clubs) & df['Wage']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = 'Club', y = 'Wage', data = df_club, palette = 'Reds')
ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)
ax.set_ylabel(ylabel = 'Distribution', fontsize = 16)
ax.set_title(label = 'Disstribution of wages in some popular clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# Representing distribution of international reputation in top clubs

# In[ ]:


df_club = df.loc[df['Club'].isin(some_clubs) & df['International Reputation']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = 'Club', y = 'International Reputation', data = df_club, palette = 'bright')
ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)
ax.set_ylabel(ylabel = 'International reputation', fontsize = 16)
ax.set_title(label = 'Distribution of international reputation in some popular clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# Representing distribution of weight in top clubs

# In[ ]:


df_clubs = df.loc[df['Club'].isin(some_clubs) & df['Weight']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = 'Club', y = 'Weight', data = df_clubs, palette = 'rainbow')
ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 16)
ax.set_title(label = 'Distribution of weight in different popular clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# The clubs with players from highest number of different nationalities

# In[ ]:


df.groupby(df['Club'])['Nationality'].nunique().sort_values(ascending = False).head(11)


# The clubs with players from least number of different nationalities

# In[ ]:


df.groupby(df['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10)


# 6. **National level analysis**

# The top nations on the basis of average overall rating

# In[ ]:


d = {'Overall': 'Average_Rating'}
best_overall_country_df = df.groupby('Nationality').agg({'Overall':'mean'}).rename(columns = d)
nations = best_overall_country_df.Average_Rating.nlargest(5).index
print(nations)


# The top nations on the basis of attack

# In[ ]:


best_attack_nation_df = players.groupby('Nationality')[attck_list].sum().mean(axis = 1)
nations = best_attack_nation_df.nlargest(5).index
print(nations)


# The top nations on the basis of defence

# In[ ]:


best_defense_nation_df = players.groupby('Nationality')['Defending'].sum()
nations = best_defense_nation_df.nlargest(5).index
print(nations)


# Finding the nations with the most number of players

# In[ ]:


df['Nationality'].value_counts().head(15)


# Comparing the top nations on the basis of a few important parameters

# In[ ]:


some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Colombia', 'Japan', 'Netherlands')


# Representing distribution of weight in top nations

# In[ ]:


df_countries = df.loc[df['Nationality'].isin(some_countries) & df['Weight']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['Weight'], palette = 'cubehelix')
ax.set_xlabel(xlabel = 'Countries', fontsize = 16)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 16)
ax.set_title(label = 'Distribution of weight of players from different countries', fontsize = 20)
plt.show()


# Representing distribution of overall rating in top nations

# In[ ]:


df_countries = df.loc[df['Nationality'].isin(some_countries) & df['Overall']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['Overall'], palette = 'spring')
ax.set_xlabel(xlabel = 'Countries', fontsize = 16)
ax.set_ylabel(ylabel = 'Overall scores', fontsize = 16)
ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)
plt.show()


# Representing distribution of wages in top nations

# In[ ]:


df_countries = df.loc[df['Nationality'].isin(some_countries) & df['Wage']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['Wage'], palette = 'hot')
ax.set_xlabel(xlabel = 'Countries', fontsize = 16)
ax.set_ylabel(ylabel = 'Wage', fontsize = 16)
ax.set_title(label = 'Distribution of wages of players from different countries', fontsize = 20)
plt.show()


# Representing distribution of international reputation in top nations

# In[ ]:


df_countries = df.loc[df['Nationality'].isin(some_countries) & df['International Reputation']]
plt.rcParams['figure.figsize'] = (20, 10)
ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['International Reputation'], palette = 'autumn')
ax.set_xlabel(xlabel = 'Countries', fontsize = 16)
ax.set_ylabel(ylabel = 'International reputation', fontsize = 16)
ax.set_title(label = 'Distribution of international repuatation of players from different countries', fontsize = 20)
plt.show()


# 7. **Preparing data for modelling**

# We will try to predict the overall rating of a player using various regression algorithms. For determining the important attributes for prediction, we will try to find correlation between them and then decide which attributes to use.

# In[ ]:


selected_columns = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',
                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',
                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',
                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']

df_selected = pd.DataFrame(df, columns = selected_columns)
df_selected.columns


# In[ ]:


df_selected.sample(5)


# Plotting a correlation heatmap

# In[ ]:


plt.rcParams['figure.figsize'] = (30, 30)
sns.heatmap(df_selected[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',
                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',
                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',
                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True)

plt.title('Heatmap of the dataset', fontsize = 30)
plt.show()


# As we can observe, there is an extremely high correlation between the different attributes of a goalkeeper. We will plot a heatmap between these attributes.

# In[ ]:


GK_attributes = df[['GKPositioning','GKKicking','GKHandling','GKReflexes','GKDiving']]

plt.rcParams['figure.figsize'] = (10, 10)
sns.heatmap(GK_attributes.corr(), annot = True)

plt.title('correlations between attributes of goalkeeper', fontsize = 30)
plt.show()


# We will keep just 1 attribute out of these 5 as the other 4 attributes would be redundant for prediction.

# Creating a new dataframe which will be used for prediction

# In[ ]:


dummy_df = df.copy()


# Information about the types of columns in the dataframe.

# In[ ]:


print(dummy_df.keys())
print(dummy_df.shape)
print(dummy_df.select_dtypes(['O']).shape)        
print(dummy_df.select_dtypes([np.number]).shape)


# Because of the high correlation between the attributes of a goalkeeper, we will only keep only 1 attribute out of the 5 as we do not want redundant features for our model.

# In[ ]:


dummy_df.drop(['GKPositioning', 'GKKicking', 'GKHandling', 'GKReflexes'], inplace = True, axis = 1)


# Clearly, we also don't want to have 'rating' as one of our attributes as it is the mean of 'overall' and 'potential' and 'overall' is the variable which we want to predict.

# In[ ]:


dummy_df.drop(['Rating'], inplace = True, axis = 1)


# 'standing tackle', 'sliding tackle', 'marking' and 'interceptions' have extremely high correlation which is understandable since all of them are the attributes of a defender and thus we will only keep 'marking' for our model.

# In[ ]:


dummy_df.drop(['StandingTackle', 'SlidingTackle'], inplace = True, axis = 1)


# In[ ]:


dummy_df.drop(['Interceptions'], inplace = True, axis = 1)


# 'ball control' has a high correlation with 'dribbling' which makes sense since having a high or low rating in either one of them mostly implies the same for the other, implying that both the skills go hand in hand.

# In[ ]:


dummy_df.drop(['BallControl'], inplace = True, axis = 1)


# 'long shots' has a high correlation with 'shot power' since having a high or low rating in either of them automatically implies the same for the other as well in most of the cases.

# In[ ]:


dummy_df.drop(['LongShots'], inplace = True, axis = 1)


# Searching for columns of string type which can be eliminated.

# In[ ]:


string_columns = dummy_df.select_dtypes(['O']).columns           
string_columns


# Clearly, we do not want columns such as 'Photo', 'Flag', 'Club Logo', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Release Clause', 'Preferred Foot', 'Nationality','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW','LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM','CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Body Type'.

# In[ ]:


dummy_df.drop(['Photo', 'Flag', 'Club Logo', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Release Clause'],inplace = True,axis = 1)


# In[ ]:


dummy_df.drop(['Preferred Foot', 'Nationality'], inplace = True, axis = 1)


# 'Work rate' is an extremely important parameter for our model, but, we need to convert it to numeric type from string type since we can only use numeric values for the parameters in our model.

# In[ ]:


dummy_df[['w_r_attack','w_r_defence']] = dummy_df['Work Rate'].str.split('/',expand=True)


# In[ ]:


dummy_df.w_r_attack = dummy_df.w_r_attack.str.strip()
dummy_df.w_r_defence = dummy_df.w_r_defence.str.strip()


# In[ ]:


dummy_df.w_r_defence = dummy_df.w_r_defence.map({'High':3,'Medium':2,'Low':1})
dummy_df.w_r_attack = dummy_df.w_r_attack.map({'High':3,'Medium':2,'Low':1})


# In[ ]:


dummy_df.drop(['Work Rate'],inplace = True,axis = 1)


# In[ ]:


dummy_df.drop(['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'], inplace = True, axis = 1)


# In[ ]:


dummy_df.drop(['Body Type'], inplace = True, axis = 1)


# In[ ]:


string_columns = dummy_df.select_dtypes(['O']).columns           
string_columns


# Although we do not need the 'name' column for our model, we will keep it for referencing players later on and as far as the other 2 columns are concerned, we will encode them using scikit-learn's 'LabelEncoder' function.

# In[ ]:


le = LabelEncoder()
for column in string_columns[1:]:                                    
    dummy_df[column] = le.fit_transform(dummy_df[column])


# In[ ]:


string_columns = dummy_df.select_dtypes(['O']).columns      
string_columns


# Searching for columns of numeric type which can be eliminated.

# In[ ]:


number_columns = dummy_df.select_dtypes([np.number]).columns           
number_columns


# Clearly, we don't want columns such as 'Unnamed: 0', 'Jersey Number', 'ID'.

# In[ ]:


dummy_df.drop(['Unnamed: 0', 'Jersey Number', 'ID'], inplace = True, axis = 1)


# In[ ]:


dummy_df


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# We will create a new dataframe named 'dummy_df_scaled' which will contain the values of 'Value', 'Wage' and 'Special' between 0 and 1, since they are having a much higher range of values than the other columns, so, our model might give more importance to these columns which may result in a lower accuracy.

# In[ ]:


dummy_df_scaled = dummy_df.copy()


# We will also train the linear regression algorithms on the unscaled data so as to compare the difference in performance for both, the unscaled and scaled dataset.

# In[ ]:


output_2 = dummy_df['Overall']
dummy_df.drop(['Overall'], inplace = True, axis = 1)


# Changing the range of the values in all the numeric type columns of the data to (0, 1). 

# In[ ]:


scaling = MinMaxScaler(copy = False).fit(dummy_df_scaled.iloc[:, 1:])
dummy_df_scaled.iloc[:, 1:] = scaling.transform(dummy_df_scaled.iloc[:, 1:])


# In[ ]:


dummy_df_scaled


# 8. **Modelling**

# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score,KFold
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error
from IPython.display import display


# First of all, we will create a function which outputs a model's performance on various metrics and also prints details about the model's parameters.
# 
# The parameters to be passed to the function are:
# 
# 1. model = Model to be used for prediction 
# 
# 2. data = Data to be used for prediction(does not contain the column which is to be predicted)
# 
# 3. output_df = The variable's column which is to be predicted
# 
# The function outputs the following:
# 
# 1. R-squared score 
# 
# 2. Root-mean-square error 
# 
# 3. Cross-validation score
# 
# 4. The error in the predicted value for each player
# 
# 5. Limits of the error in prediction
# 
# 6. Weights and constant of the linear model
# 
# 7. Importance of each feature for the model
# 
# 8. Prediction and residual plots
# 
# For understanding the different linear models such as Linear, Ridge, Lasso and ElasticNet regression, the importance of R-squared score and residual plot, take a look at the following links:
# 
# 1. https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
# 
# 2. https://blog.minitab.com/blog/adventures-in-statistics-2/why-you-need-to-check-your-residual-plots-for-regression-analysis
# 
# 3. https://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
# 
# For more information about the 'PermutationImportance' function:
# 
# 1. https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html

# In[ ]:


def model_performance(model, data, output_df):
    
    # Splitting the data into train and test set
    x_train, x_test, y_train, y_test = train_test_split(data, output_df)
    
    train_names = x_train.iloc[:,0]
    x_train = x_train.iloc[:,1:]
    test_names = x_test.iloc[:,0]
    x_test = x_test.iloc[:,1:]
    
    start = time.time()
    model.fit(x_train, y_train)
    print("fitting time : {}".format(time.time()-start))

    start = time.time()
    y_pred = model.predict(x_test)
    print("\nModel's score is :", model.score(x_test, y_test))          # Returns the coefficient of determination R^2 of the prediction.
    print("testing time : {}".format(time.time() - start))
    
    print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
    
    #Cross-validation score
    crossScore = cross_val_score(model, X = data.iloc[:,1:], y = output_df, cv = KFold(n_splits = 5, shuffle = True)).mean()
    print("\nThe cross-validation score is:", crossScore) 
    
    comparisionRDF = pd.DataFrame(y_test)
    comparisionRDF['predicted'] = y_pred
    comparisionRDF['player Name'] = test_names
    comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
    print("\nThe error in the prediction for each player is:")
    print(comparisionRDF)
    
    #Limits of error
    print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))
    
    model.get_params()              
    weights = model.coef_           # Coefficients of all features/columns
    bias = model.intercept_         # Constant term in a linear line equation
    print("\nweights are:",weights)
    print("Constant is :",bias)
    
    print("\nThe importance of each feature for the model is:")
    perm = PermutationImportance(model).fit(x_test, y_test)
    display(eli5.show_weights(perm, feature_names = x_test.columns.tolist()))
    
    # Visualising the results
    plt.figure(figsize=(20, 10))
    sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
    plt.xlabel('Predictions')
    plt.ylabel('Overall')
    plt.title("Prediction of overall rating")
    plt.show()
    
    # Visualising the residual plot
    plt.figure(figsize=(20, 10))
    sns.scatterplot(y_pred, y_test - y_pred)
    plt.xlabel('Predictions')
    plt.ylabel('Residual')
    plt.title("Residual plot")
    plt.show()


# Removing the 'Overall' column as it is the variable which we want to predict.

# In[ ]:


output = dummy_df_scaled['Overall']
dummy_df_scaled.drop(['Overall'], inplace = True, axis = 1)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression() 
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# The accuracy of the algorithm for both, unscaled and scaled data is similar, the reason for which is explained in the following link:
# https://www.quora.com/Do-I-need-to-do-feature-scaling-for-simple-linear-regression

# Let us try Ridge regression, which applies the L2 penalty.

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


model = Ridge()
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# The residual plot for Ridge regression with the value of alpha as 1 looks better compared to Linear regression for both scaled and unscaled data, probably due to the penalty term which is added in Ridge regression.

# In[ ]:


model = Ridge(alpha = 0.05)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = Ridge(alpha = 0.5)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = Ridge(alpha = 5)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# The optimal value of alpha for Ridge regression is 1 for this problem, since the residual plot is almost the best which we can achieve, while there is not much difference in the accuracy of the models with different values of alpha.

# Now, let us try out Lasso regression, which applies the L1 penalty.

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


model = Lasso(alpha = 0.05)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = Lasso(alpha = 0.05, max_iter = 10000)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = Lasso(alpha = 0.5, max_iter = 10000)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = Lasso(alpha = 5, max_iter = 10000)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# You must have observed that Lasso regression gave an extremely low accuracy for scaled data since according to this algorithm, if the values of the features are small, it will have bigger coefficients which will result in bigger Lasso penalty and therefore the algorithm will assign weights close to 0 to almost all the variables which will result in extremely low accuracy. Also, the accuracy decreased as we increases the value of alpha, the reason being that the penalty for the features will increase with increase in the value of alpha which will result in a higher number of features having weights 0.

# Let us try out ElasticNet regression, which is a combination of Lasso and Ridge regression.

# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


model = ElasticNet(alpha = 1, l1_ratio = 0.5)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = ElasticNet(alpha = 1, l1_ratio = 0.3)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# In[ ]:


model = ElasticNet(alpha = 1, l1_ratio = 0.7)
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# There is not much difference in the performance of the algorithm for different values of l1 ratio, but as was the case with Lasso regression, the weights assigned to the variables are close to 0 for scaled data since ElasticNet regression has some amount of L1 penalty(Lasso regression).

# In[ ]:


from sklearn.svm import LinearSVR


# In[ ]:


model = LinearSVR()
model_performance(model, dummy_df_scaled, output)
model_performance(model, dummy_df, output_2)


# As you might have observed for the case of LinearSVR, the scaled data gives a much better performance than unscaled data, the reason for which is explained in the given link:
# 
# https://stackoverflow.com/questions/15436367/svm-scaling-input-values
# 
# Although the answer is about SVM, the same concept applies to LinearSVR as well.

# Now, we will apply ensemble algorithms over the scaled data which will hopefully give a much better performance than the linear algorithms.
# For understanding the reason due to which ensemble methods mostly work better than individual models, refer the given link:
# 
# https://www.quora.com/How-do-ensemble-methods-work-and-why-are-they-superior-to-individual-models
# 
# To understand the working of the ensemble algorithms, the scikit-learn documentation is an excellent source:
# 
# https://scikit-learn.org/stable/modules/ensemble.html

# We will do a random grid search over the hyperparameters rather than a normal grid search. For understanding the concepts of hyperparameter tuning, refer the following article:
# 
# https://www.oreilly.com/ideas/evaluating-machine-learning-models/page/5/hyperparameter-tuning
# 
# The following paper shows that randomly chosen trials are more efficient for hyper-parameter optimization than trials on a regular grid.
# 
# http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# Splitting the data into train and test set.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(dummy_df_scaled, output)
    
train_names = x_train.iloc[:,0]
x_train = x_train.iloc[:,1:]
test_names = x_test.iloc[:,0]
x_test = x_test.iloc[:,1:]


# Let us first try Random Forest regression. 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model = RandomForestRegressor()
print('Parameters currently in use:\n')
print(model.get_params())


# In[ ]:


# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees in random forest
               'max_features': ['auto', 'sqrt'],# Number of features to consider at every split
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in tree
               'min_samples_split': [2, 5, 10, 20],# Minimum number of samples required to split a node
               'min_samples_leaf': [1, 2, 4, 10, 25],# Minimum number of samples required at each leaf node 
               'bootstrap': [True, False]}# Method of selecting samples for training each tree
print(random_grid)
        
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 5 different combinations
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 2, cv = 3, verbose = 1, n_jobs = 1)
        
# Fit the random search model
model_random.fit(x_train, y_train)

print("\nThe best parameters for the model:", model_random.best_params_)
        
model = model_random.best_estimator_
        
start = time.time()
model.fit(x_train, y_train)
print("fitting time : {}".format(time.time()-start))
        
start = time.time()
y_pred = model.predict(x_test)
print("testing time : {}".format(time.time() - start))

print("\nThe importance of each feature for the model:", model.feature_importances_)


# Although I have only used 2 iterations for the random search for this algorithm and Extra Trees regressor, you can try out more iterations which might increase the acccuracy of the models. The reason being that even with 5 iterations, it took about 45 minutes to find the optimal set of features for these 2 algorithms.

# In[ ]:


print("\nModel's score is :", model.score(x_test, y_test))
print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
        
crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()
print("\nThe cross-validation score is:", crossScore)
        
comparisionRDF = pd.DataFrame(y_test)
comparisionRDF['predicted'] = y_pred
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
print("\nThe error in the prediction for each player is:")
print(comparisionRDF)
    
#Limits of error
print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

print('\nParameters currently in use:\n')
print(model.get_params())
    
#Visualising the results
plt.figure(figsize=(20, 10))
sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Prediction of overall rating")
plt.show()

#Visualising the residual plot
plt.figure(figsize=(20, 10))
sns.scatterplot(y_pred, y_test - y_pred)
plt.xlabel('Predictions')
plt.ylabel('Residual')
plt.title("Residual plot")
plt.show()


# We can see a huge jump in the score and a decrease in RMSE compared to the linear models even though we had only used 2 iterations for the random search of the hyperparameters, this demonstrates the power of ensembling algorithms. Also, the residual plot has drastically improved compared to the linear models.

# Now, let us try Extra Trees regression.

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


model = ExtraTreesRegressor()
print('Parameters currently in use:\n')
print(model.get_params())


# In[ ]:


# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees in random forest
               'max_features': ['auto', 'sqrt'],# Number of features to consider at every split
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in tree
               'min_samples_split': [2, 5, 10, 20],# Minimum number of samples required to split a node
               'min_samples_leaf': [1, 2, 4, 10, 25],# Minimum number of samples required at each leaf node 
               'bootstrap': [True, False]}# Method of selecting samples for training each tree
print(random_grid)
        
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 2, cv = 3, verbose = 1, n_jobs = 1)
        
# Fit the random search model
model_random.fit(x_train, y_train)


print("\nThe best parameters for the model:", model_random.best_params_)
        
model = model_random.best_estimator_
        
start = time.time()
model.fit(x_train, y_train)
print("fitting time : {}".format(time.time()-start))
        
start = time.time()
y_pred = model.predict(x_test)
print("testing time : {}".format(time.time() - start))

print("\nThe importance of each feature for the model:", model.feature_importances_)


# In[ ]:


print("\nModel's score is :", model.score(x_test, y_test))
print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
        
crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()
print("\nThe cross-validation score is:", crossScore)
        
comparisionRDF = pd.DataFrame(y_test)
comparisionRDF['predicted'] = y_pred
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
print("\nThe error in the prediction for each player is:")
print(comparisionRDF)
    
#Limits of error
print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

print('\nParameters currently in use:\n')
print(model.get_params())
    
#Visualising the results
plt.figure(figsize=(20, 10))
sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Prediction of overall rating")
plt.show()

#Visualising the residual plot
plt.figure(figsize=(20, 10))
sns.scatterplot(y_pred, y_test - y_pred)
plt.xlabel('Predictions')
plt.ylabel('Residual')
plt.title("Residual plot")
plt.show()


# Let us use AdaBoost regression

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


model = AdaBoostRegressor()
print('Parameters currently in use:\n')
print(model.get_params())


# In[ ]:


# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)],# Maximum number of trees at which boosting is terminated
               'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],# Learning rate of the algorithm
               'loss': ['linear', 'square', 'exponential']}# Loss function to be used for updating the weights
print(random_grid)
        
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)

# Fit the random search model
model_random.fit(x_train, y_train)
        
print("\nThe best parameters for the model:", model_random.best_params_)
        
model = model_random.best_estimator_
        
start = time.time()
model.fit(x_train, y_train)
print("fitting time : {}".format(time.time()-start))
        
start = time.time()
y_pred = model.predict(x_test)
print("testing time : {}".format(time.time() - start))        
        
print("\nThe weights given to different estimators:", model.estimator_weights_)
        
print("\nThe errors of different estimators:", model.estimator_errors_)


# In[ ]:


print("\nModel's score is :", model.score(x_test, y_test))
print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
        
crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()
print("\nThe cross-validation score is:", crossScore)
        
comparisionRDF = pd.DataFrame(y_test)
comparisionRDF['predicted'] = y_pred
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
print("\nThe error in the prediction for each player is:")
print(comparisionRDF)
    
#Limits of error
print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

print('\nParameters currently in use:\n')
print(model.get_params())
    
#Visualising the results
plt.figure(figsize=(20, 10))
sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Prediction of overall rating")
plt.show()

#Visualising the residual plot
plt.figure(figsize=(20, 10))
sns.scatterplot(y_pred, y_test - y_pred)
plt.xlabel('Predictions')
plt.ylabel('Residual')
plt.title("Residual plot")
plt.show()


# Let us now try Gradient Boosting regression.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


model = GradientBoostingRegressor()
print('Parameters currently in use:\n')
print(model.get_params())


# In[ ]:


# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees 
               'max_features': ['auto', 'sqrt'],# Number of features to consider at every split
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in tree
               'min_samples_split': [2, 5, 10, 20],# Minimum number of samples required to split a node
               'min_samples_leaf': [1, 2, 4, 10, 25]}# Minimum number of samples required at each leaf node
print(random_grid)
        
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)

# Fit the random search model
model_random.fit(x_train, y_train)
        
print("\nThe best parameters for the model:", model_random.best_params_)
        
model = model_random.best_estimator_
        
start = time.time()
model.fit(x_train, y_train)
print("fitting time : {}".format(time.time()-start))
        
start = time.time()
y_pred = model.predict(x_test)
print("testing time : {}".format(time.time() - start))


# In[ ]:


print("\nModel's score is :", model.score(x_test, y_test))
print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
        
crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()
print("\nThe cross-validation score is:", crossScore)
        
comparisionRDF = pd.DataFrame(y_test)
comparisionRDF['predicted'] = y_pred
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
print("\nThe error in the prediction for each player is:")
print(comparisionRDF)
    
#Limits of error
print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

print('\nParameters currently in use:\n')
print(model.get_params())
    
#Visualising the results
plt.figure(figsize=(20, 10))
sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Prediction of overall rating")
plt.show()

#Visualising the residual plot
plt.figure(figsize=(20, 10))
sns.scatterplot(y_pred, y_test - y_pred)
plt.xlabel('Predictions')
plt.ylabel('Residual')
plt.title("Residual plot")
plt.show()


# Let us try the XGBoost algorithm. For a tutorial on hyperparameter tuning of the XGBoost algorithm, refer the following article:
# 
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# 
# For understanding the effectiveness of this algorithm, take a look at the following link:
# 
# https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

# In[ ]:


import xgboost as xgb


# In[ ]:


model = xgb.XGBRegressor()
print('Parameters currently in use:\n')
print(model.get_params())


# In[ ]:


# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees
               'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],# Learning rate of the algorithm
               'min_child_weight': [1, 3, 5, 7, 9],# Minimum sum of instance weight needed in a child
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in a tree
               'colsample_bytree': [0.5, 0.8, 1],# Subsample ratio of columns when constructing each tree,
               'scale_pos_weight': [1, 2, 3, 4, 5]}# Balancing of positive and negative weights
print(random_grid)
        
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)

# Fit the random search model
model_random.fit(x_train, y_train)
        
print("\nThe best parameters for the model:", model_random.best_params_)
        
model = model_random.best_estimator_
        
start = time.time()
model.fit(x_train, y_train)
print("fitting time : {}".format(time.time()-start))
        
start = time.time()
y_pred = model.predict(x_test)
print("testing time : {}".format(time.time() - start))
        
print("\nThe underlying xgboost booster of this model:", model.get_booster())
        
print("\nThe number of xgboost boosting rounds:", model.get_num_boosting_rounds())
    
print("\nXgboost type parameters:", model.get_xgb_params())        


# In[ ]:


print("\nModel's score is :", model.score(x_test, y_test))
print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
        
crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()
print("\nThe cross-validation score is:", crossScore)
        
comparisionRDF = pd.DataFrame(y_test)
comparisionRDF['predicted'] = y_pred
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
print("\nThe error in the prediction for each player is:")
print(comparisionRDF)
    
#Limits of error
print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

print('\nParameters currently in use:\n')
print(model.get_params())
    
#Visualising the results
plt.figure(figsize=(20, 10))
sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Prediction of overall rating")
plt.show()

#Visualising the residual plot
plt.figure(figsize=(20, 10))
sns.scatterplot(y_pred, y_test - y_pred)
plt.xlabel('Predictions')
plt.ylabel('Residual')
plt.title("Residual plot")
plt.show()


# As observed above, all the ensemble algorithms performed much better than the linear algorithms, hence, proving their supremacy.

# Now, let us try k-nearest neighbours regression.

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


# Create the random grid
random_grid = {'n_neighbors': [5, 10, 15, 20],# Number of neighbors
               'weights': ['uniform', 'distance'],# Whether to weigh each point in the neighborhood equally or by the inverse of their distance
               'leaf_size': [20, 30, 40, 50]}# To be passed to the algorithm which will be used to compute the nearest neighbors
print(random_grid)

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)

# Fit the random search model
model_random.fit(x_train, y_train)
        
print("\nThe best parameters for the model:", model_random.best_params_)
        
model = model_random.best_estimator_
        
start = time.time()
model.fit(x_train, y_train)
print("fitting time : {}".format(time.time()-start))
        
start = time.time()
y_pred = model.predict(x_test)
print("testing time : {}".format(time.time() - start))


# In[ ]:


print("\nModel's score is :", model.score(x_test, y_test))
print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))
        
crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()
print("\nThe cross-validation score is:", crossScore)
        
comparisionRDF = pd.DataFrame(y_test)
comparisionRDF['predicted'] = y_pred
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])
print("\nThe error in the prediction for each player is:")
print(comparisionRDF)
    
#Limits of error
print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

print('\nParameters currently in use:\n')
print(model.get_params())
    
#Visualising the results
plt.figure(figsize=(20, 10))
sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Prediction of overall rating")
plt.show()

#Visualising the residual plot
plt.figure(figsize=(20, 10))
sns.scatterplot(y_pred, y_test - y_pred)
plt.xlabel('Predictions')
plt.ylabel('Residual')
plt.title("Residual plot")
plt.show()


# Even k-nearest neighbours performs significantly better than linear models and is at par with ensemble models.

# Finally, let us try to fit a neural network on the scaled data. We will build the neural network using Keras.

# In[ ]:


from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Adam


# Splitting the data into train and test set.

# In[ ]:


x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(dummy_df_scaled, output)
    
train_names_nn = x_train_nn.iloc[:,0]
x_train_nn = x_train_nn.iloc[:,1:]
test_names_nn = x_test_nn.iloc[:,0]
x_test_nn = x_test_nn.iloc[:,1:]


# We will build a five layered neural network with 3 hidden layers, however, this is not based on any theory, rather, it is based on trial and error and since this model gave a great performance, I went ahead with this, you can try building your own architecture and experiment with them. The learning rate was also selected after doing a few experiments, however, there are techniques which can assist in finding an optimal learning rate, one such technique is described here:
# 
# https://arxiv.org/abs/1506.01186

# The loss and the metric, both, are mean squared error since we will compare the performance of the neural network with the other algorithms using this metric.

# In[ ]:


input_layer = Input((dummy_df_scaled.shape[1] - 1,))
y = Dense(64, kernel_initializer = 'he_normal', activation = 'relu')(input_layer)
y = Dense(32, kernel_initializer = 'he_normal', activation = 'relu')(y)
y = Dense(8, kernel_initializer = 'he_normal', activation = 'relu')(y)
y = Dense(1, kernel_initializer = 'he_normal', activation = 'sigmoid')(y)

model = Model(inputs = input_layer, outputs = y)
model.compile(optimizer = Adam(lr = 0.001), loss = 'mse', metrics = ['mean_squared_error'])
model.summary()


# In[ ]:


history = model.fit(x_train_nn, y_train_nn, epochs = 1000, batch_size = 512)


# Plotting the graph of epochs vs. loss

# In[ ]:


plt.plot(history.history['loss'], label = 'train')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()


# In[ ]:


scores = model.evaluate(x_test,y_test)
print("Test Set MSE(loss, metric):",scores)


# Since the loss and the metric are the same, both the outputs from model.evaluate are the same. As observed from the mean squared error, neural network outperforms all the methods tested so far in this kernel and so far, we have not even tried to optimise the neural network. This goes on to show why neural networks have become so popular nowadays, with one of the major reasons being the unprecedented accuracy which they can achieve.

# This was my first kernel, wherein I learned a whole lot of new techniques and concepts. It was a great learning experience for me and I would like to thank the Kaggle community for creating such a wonderful platform where people who are new to data science, like me, can leverage their skills so easily. Any suggestions or comments are welcome. Please consider upvoting this kernel if it was helpful in any way to you, it would be much appreciated.
# 
# Thank you
