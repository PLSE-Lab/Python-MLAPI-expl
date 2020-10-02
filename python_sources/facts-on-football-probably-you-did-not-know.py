#!/usr/bin/env python
# coding: utf-8

# # ** Well, Most People either love to play or watch Football. **
# 
# ![](http://www.elitecolumn.com/wp-content/uploads/2017/09/Andrea-Pirlo-Quotes-4.jpg)
# 
# ###  "Football is played with the head. Your feet are just tools." - Andrea Pirlo.
# 
# That's some wise words, from one of the best professionals to have played this game for his life.  
# 
# As a fan of Football, the heated atmosphere enveloping the stadium packed with fans(*EPL houses on an average of 38,297 people* ) from all over the world to support their club/team by singing phrases in unision to motivate the team, Shouting in joy when a Goal is scored or When the Goalkeeper saves a stunner or Even sometimes a Nail biting finish to win the league and secure the title (AAAAGGGUUUEEERRROOO !!! )  can run chills down the spine. It's more than a game, its a part of the Family to many!!
# 
# I shall be analysing the dataset containing the detailed information about the players in the game. This dataset shall help us attain certain facts which you might not know.

# In[ ]:


## Import all the necessary packages and libraries

import numpy as np 
import pandas as pd
import re
from math import pi
import os

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


## Import the dataset 
data = pd.read_csv('../input/data.csv')
data.head()


# In[ ]:


## Print all the available columns in the dataset provided

data.columns


# It is not easy to work with the entire set of data, I have selected a few columns from the total existing number of columns. This is an area of choice, feel free to work around a bit.

# In[ ]:


## I have handpicked a few columns, that i consider important. Feel free to alter this input data.

columns = [
    'Name',
    'Age',
    'Nationality',
    'Overall',
    'Potential',
    'Preferred Foot',
    'Acceleration',
    'Aggression',
    'Agility',
    'Balance',
    'BallControl',
    'Composure',
    'Crossing',
    'Curve',
    'Club',
    'Dribbling',
    'FKAccuracy',
    'Finishing',
    'GKDiving',
    'GKHandling',
    'GKKicking',
    'GKPositioning',
    'GKReflexes',
    'HeadingAccuracy',
    'Interceptions',
    'International Reputation',
    'Jumping',
    'LongPassing',
    'LongShots',
    'Marking',
    'Position',
    'Positioning',
    'Reactions',
    'ShortPassing',
    'ShotPower',
    'Skill Moves',
    'SlidingTackle',
    'SprintSpeed',
    'Stamina',
    'StandingTackle',
    'Strength',
    'Value',
    'Vision',
    'Volleys',
    'Wage',
    'Weak Foot',
    'Work Rate'
]


# In[ ]:


## Final dataset after removing unnecessary columns.

df = pd.DataFrame(data, columns = columns)
df.head(10)


# # 1.  Only 23% of the Footballers are left footed. 

# In[ ]:


left_or_right_footed = df["Preferred Foot"].value_counts()
print(left_or_right_footed)

percent_left = (4211)/(13948 + 4211) * 100

print("\nOnly {}% of the professionals playing football are left footed". format(percent_left))

plt.figure(figsize=(5,10))
sns.set_style("whitegrid");
sns.countplot(df["Preferred Foot"])

plt.title("Number of players ")
plt.show()


# # 2. Most of the Footballers between the age of 23 - 28 have the Highest Work Rate throughout the game.

# In[ ]:


## Lets plot a graph to test whether age is actually a factor that affects an individuals professional Career.

plt.figure(figsize = (15,10))
sns.boxplot(x = "Work Rate", y = "Age", data = df)
plt.show()


# # 3.  Number of Strikers (ST) ~ Number of Goalkeepers (GK)
# 
# ## Well, I didnt know there were as many goalkeepers as the number of Strikers professionally. Looks like these 2 positions are the most preferred by the lot.
# 
# ## There are 94 Goalkeepers for every set of 100 Strikers. Interesting !

# In[ ]:


## This block of code prints the number of footballers for each position.

total_players_per_position = [df["Position"].value_counts()]
total_players_per_position


# In[ ]:


print(float(2025)/2152)


# In[ ]:


## To plot the graph and visualise the number of players in each position.

plt.figure(figsize=(15,10))
sns.set_style("whitegrid");
sns.countplot(df["Position"])

plt.title("Number of players per Position")
plt.show()


# # 4.  Overall Performance in the game starts deteriorating after the age 32 in most cases. Unless you are Zlatan Ibrahimovic, "*I think I am like wine. The older I get. The better I get*." 
# 
# ## This plot below shows the Overall Perfomance of a Footballer with respect to his age.

# ![](https://www.trollfootball.me/upload/full/2016/03/10/theres-only-one-zlatan.jpg)

# In[ ]:


## Lets visualise the Overall Ratings per player

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Overall", size = 10).map(plt.scatter,"Age","Overall").add_legend();
plt.show()


# ## The next generation of Footballers seem to have great potentials. Many Footballers between the age 17 - 24 have a really high predicted Potential.

# In[ ]:


## Lets visualise the Talented Potentials

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Potential", size = 10).map(plt.scatter,"Age","Potential").add_legend();
plt.show()


# In[ ]:


## Lets create a new column based on Potential and present Overall

df["Ratings"] = (df["Potential"] + df["Overall"])/2


# # 5. The best 4 Qualities for playing a particular position :
# 
# ![Kevin De Bruyne](https://i.dailymail.co.uk/i/pix/2017/12/17/22/476464B700000578-5188919-image-a-22_1513548491192.jpg)
# 

# In[ ]:


player_features = (
    'Acceleration', 'Aggression', 'Agility', 
    'Balance', 'BallControl', 'Composure', 
    'Crossing', 'Dribbling', 'FKAccuracy', 
    'Finishing', 'GKDiving', 'GKHandling', 
    'GKKicking', 'GKPositioning', 'GKReflexes', 
    'HeadingAccuracy', 'Interceptions', 'Jumping', 
    'LongPassing', 'LongShots', 'Marking'
)

# Top five features per position
for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))


# # 6. Well, an Intelligent Pass at the right moment to the right person is as good as a Fancy dribble. 
# 
# 
# Well both Kevin De Bruyne and Luka Modric are on a par when it comes down to picking the best Midfielder. The latter has more valuable experience. Winning the Champions League 3 years in a row. Earning the Highly Esteemed Ballon D'or Award. While, the former is yet to firmly showcase the ace up his sleeve. 
# 
# A perfect team of talented youngsters as his teammates under the abled guidance of Josep "Pep" Guardiola at Manchester City, Kevin De Bruyne could really set the bar high for the next generation of midfielders. Only time should be able to decide....
# 
# ![](https://i.makeagif.com/media/6-21-2018/CnTh3e.gif)

# In[ ]:


mid_data = df.loc[(df["Acceleration"] > 85) & (df["Agility"] > 85) & (df["Vision"] > 85) & (df["BallControl"] > 85) & (df["Dribbling"] > 85) & (df["LongShots"] > 85) & (df["HeadingAccuracy"] > 85) & (df["Finishing"] >85)]


# In[ ]:


# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

data = [
    go.Scatterpolar(
        r = [87, 90, 94, 96, 97, 87],
        theta = ['LongPassing','ShortPassing','Vision', 'BallControl', 'Dribbling', 'LongPassing'],
        name = "L. Messi",
        mode = 'lines',
        line =  dict(
            color = 'orangered'
        )
    ),
    go.Scatterpolar(
        r = [91, 92, 94, 91, 86, 91],
        theta = ['LongPassing','ShortPassing','Vision', 'BallControl', 'Dribbling', 'LongPassing'],
        name = 'Kevin De Bruyne',
        mode = 'lines',
        line =  dict(
            color = 'peru'
        )
    ),
    go.Scatterpolar(
        r = [88, 93, 92, 93, 90, 88],
        theta = ['LongPassing','ShortPassing','Vision', 'BallControl', 'Dribbling', 'LongPassing'],
        name = 'L. Modric',
        mode = 'lines',
        line =  dict(
            color = 'darkviolet'
        )
    ),
    go.Scatterpolar(
        r = [87, 93, 92, 94, 89, 87],
        theta = ['LongPassing','ShortPassing','Vision', 'BallControl', 'Dribbling', 'LongPassing'],
        name = 'David Silva',
        mode = 'lines',
        line =  dict(
            color = 'deepskyblue'
        )
    ),
    go.Scatterpolar(
        r = [90, 87, 86, 90, 87, 90],
        theta = ['LongPassing','ShortPassing','Vision', 'BallControl', 'Dribbling', 'LongPassing'],
        name = 'Paul Pogba',
        mode = 'lines',
        line =  dict(
            color = 'green'
        )
    )
]

layout = go.Layout(
        title='Best MIDFIELDER',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "radar/basic")


# # 7. Best forwards in the world. (Lionel Messi v/s Cristiano Ronaldo)
# 
# ![](http://messivsronaldo.net/wp-content/uploads/2017/09/messivsronaldo.png)

# In[ ]:


data = [
    go.Scatterpolar(
        r = [91, 91, 59, 72, 95, 86, 68, 91],
        theta = ['Acceleration','Agility', 'Strength', 'Stamina', 'Balance', 'SprintSpeed', 'Jumping', 'Acceleration'],
        name = "L. Messi",
        mode = 'lines',
        line =  dict(
            color = 'violet'
        )
    ),
    go.Scatterpolar(
        r = [89, 87, 79, 88, 70, 91, 95, 89],
        theta = ['Acceleration','Agility', 'Strength', 'Stamina', 'Balance', 'SprintSpeed', 'Jumping', 'Acceleration'],
        name = 'Cristiano Ronaldo',
        mode = 'lines',
        line =  dict(
            color = 'peru'
        )
    )
]

layout = go.Layout(
        title='Best Forward - Physicality',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "radar/basic")


# In[ ]:


data = [
    go.Scatterpolar(
        r = [96, 95, 94, 96],
        theta = ['Composure', 'Finishing', 'Vision', 'Composure'],
        name = "L. Messi",
        mode = 'lines',
        line =  dict(
            color = 'violet'
        )
    ),
    go.Scatterpolar(
        r = [95, 94, 82, 95],
        theta = ['Composure', 'Finishing', 'Vision', 'Composure'],
        name = 'Cristiano Ronaldo',
        mode = 'lines',
        line =  dict(
            color = 'peru'
        )
    )
]

layout = go.Layout(
        title='Best Forward - Mentality',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "radar/basic")


# In[ ]:


data = [
    go.Scatterpolar(
        r = [93, 94, 94, 85, 86, 70, 93],
        theta = ['Curve', 'FKAccuracy', 'LongShots', 'ShotPower', 'Volleys', 'HeadingAccuracy', 'Curve'],
        name = "L. Messi",
        mode = 'lines',
        line =  dict(
            color = 'violet'
        )
    ),
    go.Scatterpolar(
        r = [81, 76, 93, 95, 87, 89, 81],
        theta = ['Curve', 'FKAccuracy', 'LongShots', 'ShotPower', 'Volleys', 'HeadingAccuracy', 'Curve'],
        name = 'Cristiano Ronaldo',
        mode = 'lines',
        line =  dict(
            color = 'peru'
        )
    )
]

layout = go.Layout(
        title='Best Forward - Shots & Set Pieces',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "Forward3")


# # 8. Country with the best average Overall Ratings of an Individual

# In[ ]:


no_players = df["Nationality"].value_counts() 
no_players[0:20]


# In[ ]:


from collections import Counter as counter

plt.figure(1 , figsize = (15 , 7))
countries = []
c = counter(df['Nationality']).most_common()[:11]
for n in range(11):
    countries.append(c[n][0])

sns.countplot(x  = 'Nationality' ,
              data = df[df['Nationality'].isin(countries)] ,
              order  = df[df['Nationality'].isin(countries)]['Nationality'].value_counts().index , 
             palette = 'rocket') 
plt.xticks(rotation = 90)
plt.title('Maximum number footballers belong to which country' )
plt.show()


# ## England has the Highest number of individuals involved in practising Football all over the world, followed by Germany and Spain

# In[ ]:


best_dict = {}
for nation in df['Nationality'].unique():
    overall_rating = df['Overall'][df['Nationality'] == nation].sum()
    best_dict[nation] = overall_rating


# In[ ]:


best_average_ovr = {}

countries = list(best_dict.keys())

for country in countries:
    if best_dict[country] > 9000 :
        best_average_ovr[country] = best_dict[country]/float(no_players[country])
    else:
        continue


# In[ ]:


best_average_ovr_sorted = sorted(best_average_ovr, key=best_average_ovr.get, reverse = True)


# In[ ]:


avg_ovr = []

for i in best_average_ovr_sorted:
    avg_ovr.append(best_average_ovr[i])


# In[ ]:


final_avg = avg_ovr[0:15]


# In[ ]:


final_countries = best_average_ovr_sorted[0:15]


# In[ ]:


plt.figure(figsize=(20,10))

plt.bar(final_countries, final_avg, align='center', alpha=0.5)
plt.ylabel('Average Overall Rating of an Individual')
plt.xlabel('Countries')
plt.title('Average Overall Rating of an individual in a country')
 
plt.show()


# ## Although England has the Highest number of Individuals playing Football. The quality of Players from Portugal, Brazil & Uruguay is better than most of the Footballing Nations.

# # 9. Club with the best average Overall Ratings of an Individual

# In[ ]:


no_players_club = df["Club"].value_counts() 
no_players_club[0:20]


# In[ ]:


best_dict2 = {}
for clu in df['Club'].unique():
    overall_rating = df['Overall'][df['Club'] == clu].sum()
    best_dict2[clu] = overall_rating


# In[ ]:


best_average_ovr2 = {}

clubss = list(best_dict2.keys())

for j in clubss:
    if best_dict2[j] > 1000 :
        best_average_ovr2[j] = best_dict2[j]/float(no_players_club[j])
    else:
        continue


# In[ ]:


best_average_ovr2_sorted = sorted(best_average_ovr2, key=best_average_ovr2.get, reverse = True)


# In[ ]:


avg_ovr2 = []

for i in best_average_ovr2_sorted:
    avg_ovr2.append(best_average_ovr2[i])


# In[ ]:


final_avg2 = avg_ovr2[0:15]


# In[ ]:


final_clubs = best_average_ovr2_sorted[0:15]


# In[ ]:


plt.figure(figsize=(30,15))

plt.bar(final_clubs, final_avg2, align='center', alpha=0.5)
plt.ylabel('Average Overall Rating of an Individual')
plt.xlabel('Clubs')
plt.title('Average Overall Rating of an individual in a Club')
 
plt.show()


# ## a) Juventus has the best individual Overall Rating of more than 80% . Followed by Napoli and Inter Milan. 
# 
# ## b) 4 out the top 5 teams are Italian sides. Looks like there is huge talent resting in the Italian sides.

# In[ ]:





# In[ ]:





# In[ ]:




