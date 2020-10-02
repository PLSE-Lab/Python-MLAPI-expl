#!/usr/bin/env python
# coding: utf-8

# # FIFA 19 Complete Player Analysis

# <img src = "https://cdn02.nintendo-europe.com/media/images/10_share_images/games_15/nintendo_switch_4/H2x1_NSwitch_EASportsFifa19_image1600w.jpg">

# FIFA 19 is a football simulation video game developed by EA Vancouver as part of Electronic Arts' FIFA series. It is the 26th installment in the FIFA series, and was released on 28 September 2018 for PlayStation 3, PlayStation 4, Xbox 360, Xbox One, Nintendo Switch, and Microsoft Windows.The game features the UEFA club competitions for the first time, including the UEFA Champions League and UEFA Europa League and the UEFA Super Cup as well.
# 
# As with FIFA 18, Cristiano Ronaldo featured as the cover athlete of the regular edition: however, following his unanticipated transfer from Spanish club Real Madrid to Italian side Juventus, new cover art was released. He also appeared with Neymar in the cover of the Champions edition. From February 2019, an updated version featured Neymar, Kevin De Bruyne and Paulo Dybala on the cover of the regular edition.
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')

from pylab import rcParams

from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


fifa = pd.read_csv('/kaggle/input/fifa19/data.csv')
fifa


# FIFA 19 Player Dataset consists of 18207 players and their field attributes distributed over 89 fields.

# In[ ]:


fifa.head()


# These are the top 5 players in FIFA 19

# In[ ]:


fifa.info()


# In[ ]:


fifa.describe()


# In[ ]:


fifa.columns


# Some fields like Photo, Flag, Club logo are not relevent so I will drop them.

# In[ ]:


fifa.drop(['Unnamed: 0','Photo','Flag','Club Logo'], axis = 1, inplace = True)


# In[ ]:


fifa.columns


# In[ ]:


fifa.shape


# In[ ]:


fifa.nunique()


# In[ ]:


#Looking for missing values
fifa.isnull().any()


# In[ ]:


#Missing value count in each field
fifa.isnull().sum()


# In[ ]:


#Heat map of missing values
plt.rcParams['figure.figsize'] = (20, 15)
sb.heatmap(pd.isnull(fifa))


# From the heatmap I can conclude that 'Loaned From' has large amount of missing data and position based records also have a significant amount of missing data.

# ## Correlation Heatmap of Player Stats

# In[ ]:


plt.rcParams['figure.figsize'] = (20,15)
sb.heatmap(fifa[['Age', 'Overall', 'Potential', 'Value', 'Wage',
                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 
                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 
                'HeadingAccuracy', 'Interceptions','International Reputation',
                'Joined', 'Jumping', 'LongPassing', 'LongShots',
                'Marking', 'Penalties', 'Position', 'Positioning',
                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',
                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',
                'Volleys']].corr(), annot = True,  cmap = 'Blues')
plt.title('Correlation Heatmap', size = 20)


# ## Age Analysis

# In[ ]:


sb.countplot(fifa['Age'], color = 'r')
plt.title('Count of players', size = 20)


# From the Age analysis I can that most of the players are aged between 19 to 28 Years. 

# In[ ]:


#Eldest Player
eldest = fifa.sort_values('Age', ascending = False)[['Name', 'Nationality', 'Age','Overall']]
eldest.set_index('Name', inplace = True)
eldest.head(10)


# These are the top 10 eldest players. The eldest player is 45 years old

# In[ ]:


#Youngest Player
youngest = fifa.sort_values('Age', ascending = True)[['Name', 'Nationality', 'Age','Overall']]
youngest.set_index('Name', inplace = True)
youngest.head(10)


# These are the top 10 young players. The youngest player is 16 years old

# ## Position Analysis

# In[ ]:


sb.countplot(fifa['Position'], color = 'r')
plt.title('Count of Players in each Position', size = 20)


# In[ ]:


# Top players for each position
tp = fifa.loc[fifa.groupby(fifa['Position'])['Overall'].idxmax()]
tp1 = pd.DataFrame(tp, columns = ['Name', 'Position', 'Overall'])
tp1


# ## Comparison between Age and Overall

# In[ ]:


sb.scatterplot(fifa['Age'], fifa['Overall'])
plt.title('Scatterplot of Players Age and Overall', size = 20)


# ## Country & Club Analysis

# In[ ]:


#Total number of countries
print('Total number of countries : {}' .format(fifa['Nationality'].nunique()))


# In[ ]:


#Count of players from each country
fifa['Nationality'].value_counts()


# In[ ]:


fifa['Nationality'].value_counts().head(10)


# These are the top 10 countries that have the maximum number of players.

# ### Wordcloud for Countries

# In[ ]:


wordcloud = WordCloud(background_color = 'black', width = 1920, height = 1080).generate(" ".join(fifa.Nationality))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


#Total number of clubs
print('Total number of clubs: {}' .format(fifa['Club'].nunique()))


# In[ ]:


fifa['Club'].value_counts()


# Highest number of players the club has is 33 and the least number of players the club has is 18.

# In[ ]:


fifa['Club'].value_counts().head(100)


# More than 100 clubs have the squad of above 30 players.

# ## Preferred Foot Analysis

# In[ ]:


fifa['Preferred Foot'].value_counts()


# In[ ]:


sb.countplot(fifa['Preferred Foot'])


# From the above chart I can say Right Footed players are more in number when compared to Left Footed i.e almost 3 times the number of Left Footed.

# In[ ]:


#Top 5 Left Footed Players
left = fifa[fifa['Preferred Foot'] == 'Left'][['Name','Overall', 'Club','Nationality']].head(10)
left


# In[ ]:


wordcloud = WordCloud(background_color = 'black', width = 1920, height = 1080).generate(" ".join(left.Name))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


#Top 5 Right Footed Players
right = fifa[fifa['Preferred Foot'] == 'Right'][['Name','Overall', 'Club', 'Nationality']].head(10)
right


# In[ ]:


wordcloud = WordCloud(background_color = 'black', width = 1920, height = 1080).generate(" ".join(right.Name))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ### Best Players

# In[ ]:


columns=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
i=0
while i < len(columns):
    print('Best {0} : {1}'.format(columns[i],fifa.loc[fifa[columns[i]].idxmax()][1]))
    i += 1


# In[ ]:


i=0
best = []
while i < len(columns):
    best.append(fifa.loc[fifa[columns[i]].idxmax()][1])
    i +=1
    
best


# ### WordCloud for Best Players

# In[ ]:


wordcloud = WordCloud(background_color = 'black', width = 1920, height = 1080).generate(" ".join(best))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ## Overall Analysis for Club & Country

# In[ ]:


#Top 10 Countries
fifa.groupby(['Nationality'])['Overall'].max().sort_values(ascending = False).head(10)


# In[ ]:


nations = ('Brazil', 'Argentina', 'Portugal', 'England', 'Germany', 'France', 'Belgium', 'Spain')
fifa_nations = fifa.loc[fifa['Nationality'].isin(nations) & fifa['Overall']]

sb.barplot(x = fifa_nations['Nationality'], y = fifa_nations['Overall'], palette = 'rocket')
plt.title(label='Player Overall distribution in Top Nations', size=20)


# In[ ]:


#Top 10 Clubs
fifa.groupby(['Club'])['Overall'].max().sort_values(ascending = False).head(10)


# In[ ]:


clubs = ('Juventus', 'Real Madrid', 'FC Barcelona', 'Chelsea', 'Manchester United', 'Paris Saint-Germain', 'Manchester City')
fifa_club = fifa.loc[fifa['Club'].isin(clubs) & fifa['Overall']]

sb.barplot(x = fifa_club['Club'], y = fifa_club['Overall'], palette = 'rocket')
plt.title(label='Player Overall distribution in Top Clubs', size=20)


# # Modelling

# In[ ]:


fifa = pd.read_csv('/kaggle/input/fifa19/data.csv')
fifa.head()


# In[ ]:


#Dropping unnecessary values 
drop_cols = fifa.columns[28:54]
fifa = fifa.drop(drop_cols, axis = 1)
fifa = fifa.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Joined','Special','Loaned From','Body Type', 'Release Clause',
               'Weight','Height','Contract Valid Until','Wage','Value','Name','Club'], axis = 1)
fifa = fifa.dropna()
fifa.head()


# ### Converting values for modelling

# In[ ]:


#Converting Real Face into values
def face_to_num(fifa):
    if(fifa['Real Face'] == 'Yes'):
        return 1
    else:
        return 0


# In[ ]:


#Converting Preferred foot record to values
def foot(fifa):
    if(fifa['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0


# In[ ]:


#Creating a simplified position variable by combining diff positions
def simple_position(fifa):
    if (fifa['Position'] == 'GK'):
        return 'GK'
    elif ((fifa['Position'] == 'RB') or (fifa['Position'] == 'LB') or (fifa['Position'] == 'CB') or (fifa['Position'] == 'LCB') or (fifa['Position'] == 'RCB') or (fifa['Position'] == 'RWB') or (fifa['Position'] == 'LWB')):
        return 'DF'
    elif ((fifa['Position'] == 'LDM') or (fifa['Position'] == 'CDM') or (fifa['Position'] == 'RDM')):
        return 'DM'
    elif ((fifa['Position'] == 'LM') or (fifa['Position'] == 'LCM') or (fifa['Position'] == 'CM') or (fifa['Position'] == 'RCM') or (fifa['Position'] == 'RM')):
        return 'MF'
    elif ((fifa['Position'] == 'LAM') or (fifa['Position'] == 'CAM') or (fifa['Position'] == 'RAM') or (fifa['Position'] == 'LW') or (fifa['Position'] == 'RW')):
        return 'AM'
    elif ((fifa['Position'] == 'RS') or (fifa['Position'] == 'ST') or (fifa['Position'] == 'LS') or (fifa['Position'] == 'CF') or (fifa['Position'] == 'LF') or (fifa['Position'] == 'RF')):
        return 'ST'
    else:
        return fifa.Position


# In[ ]:


#Making list of those nations with more then 250 players
nat_count = fifa.Nationality.value_counts()
nat_list = nat_count[nat_count > 250].index.tolist()

def major_nation(fifa):
    if(fifa.Nationality in nat_list):
        return 1
    else:
        return 0


# In[ ]:


#Creating a copy to avoid indexing error
fifa1 = fifa.copy()


# In[ ]:


#Applying changes to dataset to create new columns
fifa1['Real_Face'] = fifa1.apply(face_to_num, axis = 1)
fifa1['Right_Footed'] = fifa1.apply(foot, axis = 1)
fifa1['Simple_Position'] = fifa1.apply(simple_position, axis = 1)
fifa1['Nation'] = fifa1.apply(major_nation, axis = 1)


# In[ ]:


#Splitting the Team Work column
tempwork = fifa1['Work Rate'].str.split('/', n = 1, expand = True)

fifa1['Workrate1'] = tempwork[0]
fifa1['Workrate2'] = tempwork[1]


# In[ ]:


#Droping the original columns
fifa1 = fifa1.drop(['Work Rate', 'Preferred Foot', 'Real Face', 'Position', 'Nationality'], axis = 1)
fifa1.head()


# In[ ]:


#Spliting ID
target = fifa1.Overall
fifa2 = fifa1.drop(['Overall'], axis = 1)


# In[ ]:


#Splitting dataset for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fifa2, target, test_size = 0.2, random_state = 0)

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor()
random.fit(x_train, y_train)
y_pred = random.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score
print('R2 score: '+str(r2_score(y_test, y_pred)))

