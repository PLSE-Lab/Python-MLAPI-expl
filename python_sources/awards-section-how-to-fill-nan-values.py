#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# - [1. Information about the data given](#first-bullet)
# - [2. Importing the packages needed](#second-bullet)
# - [3. Load and prepare the data](#third-bullet)
# - [4. Initial Data Insepction](#fourth-bullet)
# - [5. Data Cleaning](#fifth-bullet)
#     - [5.1 Take out Lbs from Weight](#5.1)
#     - [5.2 Change value and wage to a real number](#5.2)
#     - [5.3 Fill the NaN Values](#5.3)
#     - [5.4 Grouping similar skills together](#5.4)
# - [6. Skills/Position Analysis](#6)
#     - [6.1 Analyzing top 5 features on all skills](#6.1)
#     - [6.2 More stats on skills and special moves](#6.2)
# - [7. Awards Section](#8)
#     - [7.1 Top footballer producing nations](#8.1)
#         - [7.1.1 Player weight distribution in top 5 footballer producing countries](#8.1.1)
#     - [7.2 Club-level Analysis](#8.2)
#         - [7.2.1 Best football clubs ](#8.2.1)
#         - [7.2.2 Clubs that have the best Attack](#8.2.2)
#         - [7.2.3 Clubs that have the best Defense](#8.2.3)
#     - [7.3 Nation-level Analysis](#8.3)
#         - [7.3.1 Best footballing nations](#8.3.1)
#         - [7.3.2 Nations that has the best Attack](#8.3.2)
#         - [7.3.3 Nations that has the best Defense](#8.3.3)
#     - [7.4 Top 5 footballers](#8.4)
# - [8. Dream Team](#7)
# - [9. Wage Analysis](#9)
# - [10. Age Analysis](#10)
# - [11. Next Steps](#11)

# # Introduction
# 
# - This notebook is built to analyze in detail on what all could be done using the data provided about all players.
# - This notebook would have an in-depth analysis of the some of the main data features
# 

# - Interesting Ideas:
#     - come up with dream team (people who are best in all positions)
#     - Top 3 clubs that are good in 'Attacking' and it's top 3 contributers
#     - Top 3 clubs that are good in 'Defense' and it's top 3 contributers
#     - Best club overall  and it's top 3 contributers
#     - Top 3 nations that has best footballers
#     - Next best player according to wages     
# - How wage and players are related
# - Players who aren't performing after age 
# - Come up with Id card that would show person's Skills, team, name, weiht, height, income
# - group subset skills under common skills
# - Which position would a person wanna get trained if he wants to make it quickly?

# ## Information about the data given<a class="anchor" id="first-bullet"></a>

# - Columns
# - row number
# - IDunique id for every player
# - Namename
# - Ageage
# - Photourl to the player's photo
# - Nationalitynationality
# - Flagurl to players's country flag
# - Overalloverall rating
# - Potentialpotential rating
# - Clubcurrent club
# - Club Logourl to club logo
# - Valuecurrent market value
# - Wagecurrent wage
# - Special
# - Preferred Footleft/right
# - International Reputationrating on scale of 5
# - Weak Footrating on scale of 5
# - Skill Movesrating on scale of 5
# - Work Rateattack work rate/defence work rate
# - Body Typebody type of player
# - Real Face
# - Positionposition on the pitch
# - Jersey Numberjersey number
# - Joinedjoined date
# - Loaned Fromclub name if applicable
# - Contract Valid Untilcontract end date
# - Heightheight of the player
# - Weightweight of the player
# - LS rating on scale of 100
# - ST rating on scale of 100
# - RS rating on scale of 100
# - LW rating on scale of 100
# - LF rating on scale of 100
# - CF rating on scale of 100
# - RF rating on scale of 100
# - RW rating on scale of 100
# - LAM rating on scale of 100
# - CAM rating on scale of 100
# - RAM rating on scale of 100
# - LM rating on scale of 100
# - LCM rating on scale of 100
# - CM rating on scale of 100
# - RCM rating on scale of 100
# - RM rating on scale of 100
# - LWB rating on scale of 100
# - LDM rating on scale of 100
# - CDM rating on scale of 100
# - RDM rating on scale of 100
# - RWB rating on scale of 100
# - LB rating on scale of 100
# - LCB rating on scale of 100
# - CB rating on scale of 100
# - RCB rating on scale of 100
# - RB rating on scale of 100
# - Crossing rating on scale of 100
# - Finishing rating on scale of 100
# - HeadingAccuracy rating on scale of 100
# - ShortPassing rating on scale of 100
# - Volleys rating on scale of 100
# - Dribbling rating on scale of 100
# - Curverating on scale of 100
# - FKAccuracy rating on scale of 100
# - LongPassing rating on scale of 100
# - BallControl rating on scale of 100
# - Acceleration rating on scale of 100
# - SprintSpeed rating on scale of 100
# - Agility rating on scale of 100
# - Reactions rating on scale of 100
# - Balance rating on scale of 100
# - ShotPower rating on scale of 100
# - Jumping rating on scale of 100
# - Stamina rating on scale of 100
# - Strength rating on scale of 100
# - LongShots rating on scale of 100
# - Aggression rating on scale of 100
# - Interceptions rating on scale of 100
# - Positioning rating on scale of 100
# - Vision rating on scale of 100
# - Penalties rating on scale of 100
# - Composure rating on scale of 100
# - Marking rating on scale of 100
# - StandingTackle rating on scale of 100
# - SlidingTackle rating on scale of 100
# - GK Diving rating on scale of 100
# - GK Handling rating on scale of 100
# - GK Kicking rating on scale of 100
# - GK Positioning rating on scale of 100
# - GK Reflexes rating on scale of 100
# - Release Clauserelease clause value
# 

# # Importing the packages needed <a class="anchor" id="second-bullet"></a>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from math import pi


# # Load and prepare the data <a class="anchor" id="third-bullet"></a>

# In[ ]:


data_path = "../input/data.csv"
df = pd.read_csv(data_path)


# # Initial Data Inspection<a class="anchor" id="fourth-bullet"></a>

# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# ### Inference
# There are a lot of null values on the data
# 

# # Data Cleaning <a class="anchor" id="fifth-bullet"></a>

# ## Take out Lbs from Weight  <a class="anchor" id="5.1"></a>

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


# ## Change value and wage to a real number<a class="anchor" id="5.2"></a>

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


# ## Fill the Nan Values  <a class="anchor" id="5.3"></a>
# - We will try to fill everything with some logic behind it

# ### Fixing Weight

# In[ ]:


df.Weight.isna().sum()


# In[ ]:


df.Weight.mean()


# According to [livestrong data](https://www.livestrong.com/article/207894-the-ideal-weight-for-a-soccer-player/),
# - The normal weight range for a player 5 feet 9 inches tall is 136 to 169 pounds. 
# - Since the mean weight of the player is 165 pounds and it gels with the global data, we could set the mean weight to fill the null values

# In[ ]:


df['Weight'].fillna(df.Weight.mean(), inplace = True)


# ### Fixing Height

# In[ ]:


df.Height.isna().sum()


# In[ ]:


plt.figure(figsize = (20, 10))
sns.countplot(x='Height', data=df)
plt.show()


# According to [livestrong data](https://www.livestrong.com/article/207894-the-ideal-weight-for-a-soccer-player/),
# 
# - The height of a player varies according to the position he takes
# - Their average height was 5 feet, 11 1/2 inches tall
# - We also find from the data given that most people are between 5.9 to 6-1
# - So we would fill the height to 5'11"

# In[ ]:


df['Height'].fillna("5'11", inplace = True)


# ### Fixing Weak Foot Rating

# In[ ]:


wf_missing = df['Weak Foot'].isna()
wf_missing.sum()


# In[ ]:


weak_foot_prob = df['Weak Foot'].value_counts(normalize=True)
weak_foot_prob


# - From this we could clearly find that majority of the players would have weak foot rating as 3
# - We will also fill the Foot with the same probability distribution

# In[ ]:


df.loc[wf_missing,'Weak Foot'] = np.random.choice(weak_foot_prob.index, size=wf_missing.sum(),p=weak_foot_prob.values)


# ### Fixing Preferred Foot

# In[ ]:


pf_missing = df['Preferred Foot'].isna()
pf_missing.sum()


# In[ ]:


df['Preferred Foot'].value_counts()


# In[ ]:


foot_distribution = df['Preferred Foot'].value_counts(normalize=True)
foot_distribution


# - From the data, it's clear that 77% of people are right footed
# - So we will fill the preferred foot in the same probability distributsion

# In[ ]:


df.loc[pf_missing, 'Preferred Foot'] = np.random.choice(foot_distribution.index, size = pf_missing.sum(), p=foot_distribution.values)


# In[ ]:


df['Preferred Foot'].value_counts()


# ### Filling Position

# In[ ]:


fp_missing = df.Position.isna()
fp_missing.sum()


# In[ ]:


position_prob = df.Position.value_counts(normalize=True)
position_prob 


# In[ ]:


plt.figure(figsize = (20, 10))
sns.countplot(x=df.Position, data=df)
plt.show()


# - From the data, it's clear that many positions have different percentage
# - So we will fill the position in the same probability distributsion

# In[ ]:


df.loc[fp_missing, 'Position'] = np.random.choice(position_prob.index, p=position_prob.values, size=fp_missing.sum())


# ### Filling Skill Moves

# In[ ]:


fs_missing = df['Skill Moves'].isna()
fs_missing.sum()


# In[ ]:


skill_moves_prob = df['Skill Moves'].value_counts(normalize=True)
skill_moves_prob


# - We could fill the nan valeus with the same probability distribution

# In[ ]:


df.loc[fs_missing, 'Skill Moves'] = np.random.choice(skill_moves_prob.index, p=skill_moves_prob.values, size=fs_missing.sum())


# ### Filling Body Type

# In[ ]:


bt_missing = df['Body Type'].isna()
bt_missing.sum()


# In[ ]:


bt_prob = df['Body Type'].value_counts(normalize=True)
bt_prob


# - Not sure what how 'Neymar', 'Messi', 'Shaqiri', 'Akinfenwa', 'Courtois' are listed as body types because they are they names of the football players
# - We fill the body types with the same probability distribution of the 'Normal' and 'Lean'

# In[ ]:


df.loc[bt_missing, 'Body Type'] = np.random.choice(['Normal', 'Lean'], p=[.63,.37], size=bt_missing.sum())


# ### Filling Wages

# In[ ]:


wage_missing = df.Wage.isna()
wage_missing.sum()


# In[ ]:


wage_prob = df.Wage.value_counts(normalize=True)
wage_prob


# - Since all good and recognized players have good wage, the wage wouldn't be filled only for players who aren't famous
# - The wage distribution says that most of the players have very less wages
# - We should not be filling it with mean because the mean would be really high 
# - So the players who are not so famous might get higher wages when compared to others who have the same talent
# - So we would fill the Nan wage columns with the probability distribution of the data

# In[ ]:


df.loc[wage_missing, 'Wage'] = np.random.choice(wage_prob.index, p=wage_prob.values, size=wage_missing.sum())


# ### Filling the rest of the valeus
# 
# - Since all features that have float64 datatypehas continuos values, we will fill it's Nan values with mean
# - Randomly fill 'Contract Valid Until', 'Work Rate', 'International Reputation' , 'Jersey Number', 'Club', 

# In[ ]:



for feature in df.columns:
    if df[feature].dtype == 'float64':
        df[feature].fillna(df[feature].mean(), inplace=True)
    
df['Contract Valid Until'].fillna(np.random.choice(df['Contract Valid Until']), inplace = True)
df['Loaned From'].fillna(np.random.choice(df['Loaned From']), inplace = True)
df['Joined'].fillna(np.random.choice(df['Joined']), inplace = True)
df['Jersey Number'].fillna(np.random.choice(df['Jersey Number']), inplace = True)
df['Club'].fillna(np.random.choice(df.Club), inplace = True)
df['Work Rate'].fillna(np.random.choice(df['Work Rate']), inplace = True)
df['International Reputation'].fillna(np.random.choice(df['International Reputation']), inplace = True)


# ### Fill the rest of the NaN data with 0

# In[ ]:


df.fillna(0, inplace = True)


# ## Grouping similar skills together<a class="anchor" id="5.4"></a>
# 
# - Here, we are grouping the skills together and generalizing it to 8 categories
# - These 8 categories would let us know which position that player would take 
# - We do this because we could analyze the players better and positon them accordingly

# In[ ]:


def defending(data):
    return data[['Marking', 'StandingTackle', 
                               'SlidingTackle']].mean().mean()

def general(data):
    return data[['HeadingAccuracy', 'Dribbling', 'Curve', 
                               'BallControl']].mean().mean()

def mental(data):
    return data[['Aggression', 'Interceptions', 'Positioning', 
                               'Vision','Composure']].mean().mean()

def passing(data):
    return data[['Crossing', 'ShortPassing', 
                               'LongPassing']].mean().mean()

def mobility(data):
    return data[['Acceleration', 'SprintSpeed', 
                               'Agility','Reactions']].mean().mean()
def power(data):
    return data[['Balance', 'Jumping', 'Stamina', 
                               'Strength']].mean().mean()

def rating(data):
    return data[['Potential', 'Overall']].mean().mean()

def shooting(data):
    return data[['Finishing', 'Volleys', 'FKAccuracy', 
                               'ShotPower','LongShots', 'Penalties']].mean().mean()


# In[ ]:


# renaming a column
df.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)

# adding these categories to the data

df['Defending'] = df.apply(defending, axis = 1)
df['General'] = df.apply(general, axis = 1)
df['Mental'] = df.apply(mental, axis = 1)
df['Passing'] = df.apply(passing, axis = 1)
df['Mobility'] = df.apply(mobility, axis = 1)
df['Power'] = df.apply(power, axis = 1)
df['Rating'] = df.apply(rating, axis = 1)
df['Shooting'] = df.apply(shooting, axis = 1)


# In[ ]:


players = df[['Name','Defending','General','Mental','Passing',
                'Mobility','Power','Rating','Shooting','Flag','Age',
                'Nationality', 'Photo', 'Club_Logo', 'Club']]


# # Skills/Position Analysis <a class="anchor" id="6"></a>

# ### Number of footballers available in each position

# In[ ]:


plt.figure(figsize = (20, 10))
ax = sns.countplot(x='Position', data=df, order = df['Position'].value_counts().index)
ax.set_title(label = 'Number of footballers available in each position', fontsize = 20)
plt.show()


# ## Analyzing top 5 features on all skills <a class="anchor" id="6.1"></a>
# 

# In[ ]:


player_features = (
    'Acceleration', 'Aggression', 'Agility', 
    'Balance', 'BallControl', 'Composure', 
    'Crossing', 'Dribbling', 'FKAccuracy', 
    'Finishing', 'GKDiving', 'GKHandling', 
    'GKKicking', 'GKPositioning', 'GKReflexes', 
    'HeadingAccuracy', 'Interceptions', 'Jumping', 
    'LongPassing', 'LongShots', 'Marking', 'Penalties'
)

from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(10, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1


# ### Inference
# - From the [position graph](#6) and the [skill graph](#6.1) above, we get to know the most important skills that are required for each position
# - This helps:
#     - the management to look out for the right set of skills from each players and to buy players with the skills they are looking for
#     - the to-be professionals to understand which skill they need to develop in order to get the position they need

# ## More stats on skills and special moves <a class="anchor" id="6.2"></a>

# In[ ]:


sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
x = df.Special
plt.figure(figsize = (12, 8))
ax = sns.distplot(x, bins = 50, kde = False, color = 'm')
ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)
ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)
plt.show()


# In[ ]:


sns.scatterplot(x = 'Special', y='Wage', data=df)


# ### Inference
# 
# - Skills and wage are not completely positively correlated

# ## Top 5 nations that have the best skilled footballers

# In[ ]:


plt.rcParams['figure.figsize'] = (20, 10)
skill_df = df[df['Skill Moves'] == 5][['Name','Nationality']]
sns.countplot(x='Nationality', data=skill_df, order=skill_df.Nationality.value_counts().iloc[:5].index)


# # Awards Section<a class="anchor" id="8"></a>
# 
# - This section would have the top 5/ top 3 contibutors for many sections

# ## Top footballer producing nations <a class="anchor" id="8.1"></a>

# In[ ]:


import squarify
df.Nationality.value_counts().nlargest(5).plot(kind='bar')


# ### Player weight distribution in top 5 footballer producing countries <a class="anchor" id="8.1.1"></a>

# In[ ]:


countries = df.Nationality.value_counts().nlargest(5).index


# In[ ]:


data_countries = df[df['Nationality'].isin(countries)]


# In[ ]:


plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'colorblind')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)
ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)


# ## Club-level Analysis <a class="anchor" id="8.2"></a>

# In[ ]:


import matplotlib.image as mpimg
import requests
def print_club_flag(clubs):
    fig = plt.figure(figsize=(10,10))
    for index, club in enumerate(clubs):
        logo = df[df['Club'] == club]['Club_Logo'].iloc[0]
        logo_image = "img_club_logo.jpg"
        logo_flag = requests.get(logo).content
        with open(logo_image, 'wb') as handler:
            handler.write(logo_flag)
        img=mpimg.imread(logo_image)
        ax = fig.add_subplot(1, 6, index+1, xticks=[], yticks=[])
        fig.tight_layout()
        ax.imshow(img, interpolation="lanczos")
        ax.set_title("%d. %s" %(index+1, club))
    
def print_national_flag(nations):
    fig = plt.figure(figsize=(10, 10))
    for index, nation in enumerate(nations):
        logo = df[df['Nationality'] == nation]['Flag'].iloc[0]
        logo_image = "img_nation_logo.jpg"
        logo_flag = requests.get(logo).content
        with open(logo_image, 'wb') as handler:
            handler.write(logo_flag)
        img=mpimg.imread(logo_image)
        ax = fig.add_subplot(1, 6, index+1, xticks=[], yticks=[])
        fig.tight_layout()
        ax.imshow(img, interpolation="lanczos")
        ax.set_title("%d. %s" %(index+1, nation))


# ### Best football clubs  <a class="anchor" id="8.2.1"></a>
# 
# - Here are the top 5 football clubs w.r.t their overall rating

# In[ ]:


d = {'Overall': 'Average_Rating'}
best_overall_club_df = df.groupby('Club').agg({'Overall':'mean'}).rename(columns=d)
clubs = best_overall_club_df.Average_Rating.nlargest(5).index
clubs_list = []

print_club_flag(clubs)


# ### Clubs that have the best Attack <a class="anchor" id="8.2.2"></a>
# - Here are the top 5 clubs that specialize in attack

# In[ ]:


attck_list = ['Shooting', 'Power', 'Passing']

best_attack_df = players.groupby('Club')[attck_list].sum().sum(axis=1)
clubs = best_attack_df.nlargest(5).index

print_club_flag(clubs)


# ### Clubs that have the best Defense  <a class="anchor" id="8.2.3"></a>
# - Here are the top 5 clubs that specialize in defense
# - [More on interpolation](https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html)

# In[ ]:



best_defense_df = players.groupby('Club')['Defending'].sum()
clubs = best_defense_df.nlargest(5).index
print_club_flag(clubs)

    


# ## Nation-level Analysis <a class="anchor" id="8.3"></a>

# ### Best footballing nations  <a class="anchor" id="8.3.1"></a>

# In[ ]:


d = {'Overall': 'Average_Rating'}
best_overall_country_df = df.groupby('Nationality').agg({'Overall':'mean'}).rename(columns=d)
nations = best_overall_country_df.Average_Rating.nlargest(5).index
print_national_flag(nations)


# In[ ]:


best_3_uae = df[df['Nationality'] == 'United Arab Emirates']['Overall'].nlargest(3)
print(best_3_uae)
uae_df = df[df['Nationality'] == 'United Arab Emirates']
uae_df[uae_df['Overall'].isin(best_3_uae)]['Name']


# ### Nations that has the best Attack<a class="anchor" id="8.3.2"></a>

# In[ ]:


best_attack_nation_df = players.groupby('Nationality')[attck_list].sum().sum(axis=1)
nations = best_attack_nation_df.nlargest(5).index
print_national_flag(nations)


# ### Nations that has the best Defense<a class="anchor" id="8.3.3"></a>

# In[ ]:


best_defense_nation_df = players.groupby('Nationality')['Defending'].sum()
nations = best_defense_nation_df.nlargest(5).index
print_national_flag(nations)


# In[ ]:


import requests
import random
from math import pi

import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)

def details(row, title, image, age, nationality, photo, logo, club):
    
    flag_image = "img_flag.jpg"
    player_image = "img_player.jpg"
    logo_image = "img_club_logo.jpg"
        
    img_flag = requests.get(image).content
    with open(flag_image, 'wb') as handler:
        handler.write(img_flag)
    
    player_img = requests.get(photo).content
    with open(player_image, 'wb') as handler:
        handler.write(player_img)
     
    logo_img = requests.get(logo).content
    with open(logo_image, 'wb') as handler:
        handler.write(logo_img)
        
    r = lambda: random.randint(0,255)
    colorRandom = '#%02X%02X%02X' % (r(),r(),r())
    
    if colorRandom == '#ffffff':colorRandom = '#a5d6a7'
    
    basic_color = '#37474f'
    color_annotate = '#01579b'
    
    img = mpimg.imread(flag_image)
    #flg_img = mpimg.imread(logo_image)
    
    plt.figure(figsize=(15,8))
    categories=list(players)[1:]
    coulumnDontUseGraph = ['Flag', 'Age', 'Nationality', 'Photo', 'Logo', 'Club']
    N = len(categories) - len(coulumnDontUseGraph)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color= 'black', size=17)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75,100], ["25","50","75","100"], color= basic_color, size= 10)
    plt.ylim(0,100)
    
    values = players.loc[row].drop('Name').values.flatten().tolist() 
    valuesDontUseGraph = [image, age, nationality, photo, logo, club]
    values = [e for e in values if e not in (valuesDontUseGraph)]
    values += values[:1]
    
    ax.plot(angles, values, color= basic_color, linewidth=1, linestyle='solid')
    ax.fill(angles, values, color= colorRandom, alpha=0.5)
    axes_coords = [0, 0, 1, 1]
    ax_image = plt.gcf().add_axes(axes_coords,zorder= -1)
    ax_image.imshow(img,alpha=0.5)
    ax_image.axis('off')
    
    ax.annotate('Nationality: ' + nationality.upper(), xy=(10,10), xytext=(103, 138),
                fontsize= 12,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})
                      
    ax.annotate('Age: ' + str(age), xy=(10,10), xytext=(43, 180),
                fontsize= 15,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})
    
    ax.annotate('Team: ' + club.upper(), xy=(10,10), xytext=(92, 168),
                fontsize= 12,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})

    arr_img_player = plt.imread(player_image, format='jpg')

    imagebox_player = OffsetImage(arr_img_player)
    imagebox_player.image.axes = ax
    abPlayer = AnnotationBbox(imagebox_player, (0.5, 0.7),
                        xybox=(313, 223),
                        xycoords='data',
                        boxcoords="offset points"
                        )
    arr_img_logo = plt.imread(logo_image, format='jpg')

    imagebox_logo = OffsetImage(arr_img_logo)
    imagebox_logo.image.axes = ax
    abLogo = AnnotationBbox(imagebox_logo, (0.5, 0.7),
                        xybox=(-320, -226),
                        xycoords='data',
                        boxcoords="offset points"
                        )

    ax.add_artist(abPlayer)
    ax.add_artist(abLogo)

    plt.title(title, size=50, color= basic_color)


# In[ ]:


# defining a polar graph

def get_id_card(id = 0):
    if 0 <= id < len(df.ID):
        details(row = players.index[id], 
                title = players['Name'][id], 
                age = players['Age'][id], 
                photo = players['Photo'][id],
                nationality = players['Nationality'][id],
                image = players['Flag'][id], 
                logo = players['Club_Logo'][id], 
                club = players['Club'][id])
    else:
        print('The base has 17917 players. You can put positive numbers from 0 to 17917')


# ## Top 5 footballers <a class="anchor" id="8.4"></a>
# 
# - This gives a pictorial representation of the top 5 footballers
# - Thanks [Roshan sharma](https://www.kaggle.com/roshansharma/fifa-2019-data-analysis-and-visualization) for the ID card code. Really well done!!!

# In[ ]:


best_footballers = df['Overall'].nlargest(5)
for index in best_footballers.index:
    get_id_card(index)


# # Dream Team  <a class="anchor" id="7"></a>
# 
# - Ever dreamt of a team which would have all your favourite players?
# - This team below has the best players in all positions :) 

# In[ ]:


df.loc[df.groupby(df['Position'])['Potential'].idxmax()][['Name', 'Position', 'Overall', 'Age', 'Nationality', 'Club']]


# # Wage Analysis  <a class="anchor" id="9"></a>

# In[ ]:


#### sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
x = df.Wage
plt.figure(figsize = (12, 8))
ax = sns.distplot(x, bins = 50, kde = False, color = 'm')
ax.set_xlabel(xlabel = 'Player Wage', fontsize = 16)
ax.set_ylabel(ylabel = 'Player Count',fontsize = 16)
ax.set_title(label = 'Histogram that shows the wage of the Players', fontsize = 20)
plt.show()


# In[ ]:


df[df['Wage']>300000][['Name','Age','Wage']]


# ### Inference
# - Looks like the wage is highly skewed
# - Only a handful of people get more than 300,000 Euros

# In[ ]:


df.groupby('Wage')['Overall'].mean().plot()


# ### Inference
# - It's very evident that the wage is getting higher only for star performers

# # Age Analysis  <a class="anchor" id="10"></a>

# In[ ]:


df.groupby('Age')['Overall'].mean().plot()


# ### Inference
# - The overall performance of the players dips after 30
# - Let us look at why it has gone up after 43

# In[ ]:


sns.countplot(x='Age', data=df)


# In[ ]:


df[df['Age']>40][['Name','Overall','Age','Nationality']]


# ### Inference
# - There is only a handful of people are there > 40
# - Mr.Perez is surely an outlier and a Mexico's pride!!!

# In[ ]:


new_wage = df[df['Wage']>10000]
new_wage['age_group'] = pd.cut(new_wage.Age, bins=4)
ax = new_wage.boxplot(column='Wage', by='age_group', showmeans=True)
ax.set_xlabel(xlabel = 'Age Group', fontsize = 20)
ax.set_ylabel(ylabel = 'Wage', fontsize = 20)


# ### Inference
# 
# - Players Are In High Demand In Their Mid-20s

# # Next Steps  <a class="anchor" id="11"></a>

# - This section would involve some more analysis predictions like:
#     - Who would be the next big star?
#     - What all would contribute to get a better salary?
#     - Please comment on which predctions/analysis would you need
