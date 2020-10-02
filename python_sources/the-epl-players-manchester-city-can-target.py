#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#STANDARD IMPORT PACKAGES

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import sklearn

from matplotlib import pyplot as plt
from scipy.stats import norm, skew
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neural_network
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings('ignore')

My favourite club is Manchester City, so the business problem I am trying to solve is which players to buy if I were the manager?Examining the dataset
# In[ ]:


dataset = pd.read_csv('../input/epldata_final.csv')
dataset.head()

Assumptions:
1. The position given in the dataset is the best position of the player, we can't assume the player to be versatile and play in multiple positions
2. We are not given any money to spend on players, we have to sell players to get new players
3. There has to be at least 5 English players
4. Here the player is judged by their market value rather than FPL Points as a player can be injured and miss some games of the season and not get FPL Points, but he is still a good player
5. Try to avoid signing or selling newly signed players
6. Try not to target too many people from the same club
7. Because we don't have data for other clubs.. All transfers can be made within Premier LeagueBased on Assumptions, we will remove columns that are not needed
# In[ ]:


dataset = dataset.drop(['position_cat','page_views','fpl_value','fpl_sel','fpl_points','region','new_foreign','club_id'],axis = 1)
dataset.shape

Part 1: Look at Man City and analyze their weak points
# In[ ]:


#dataset['club'].value_counts()
man_city = dataset[dataset['club'] == "Manchester+City"]


# In[ ]:


man_city.head()

Visual 1 - Split by position and get counts of each position, add market value as a stacked chart
and color code by nationality (wait.. lets make this England and Overseas)
# In[ ]:


man_city['England_Flag'] =  np.where(man_city['nationality']=='England', 'England', 'Overseas')


# In[ ]:


#Graph 1 - Getting the colors
plot_df = man_city.groupby(['position','England_Flag']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 2 - Getting the counts
man_city['Rank'] = man_city.sort_values(['England_Flag','market_value'], ascending=[True,False])              .groupby(['position'])              .cumcount() + 1
plot_df2 = man_city.groupby(['position','Rank']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 3 - Superimposing the two graphs
fig, ax = plt.subplots()
ax2 = ax.twiny()
plot_df.plot(kind='bar',width = 1, stacked=True,ax=ax2,alpha = 0.5)
plot_df2.plot(kind='bar',width = 1,stacked=True, edgecolor='black',legend = False, color = 'white',ax=ax)
plt.show()  

NOTE THAT IN THE ABOVE PLOT:
1. Each box denotes a Man City player and his market valueSo we know that from the 20 English players that we have to buy one English player at the expense of selling a Foreign playerWe definitely need another similar plot that emphasizes on Age buckets rather than the nationality
# In[ ]:


#Graph 4 - Getting the colors
plot_df3 = man_city.groupby(['position','age_cat']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 5 - Getting the counts
man_city['Rank'] = man_city.sort_values(['age_cat','market_value'], ascending=[True,False])              .groupby(['position'])              .cumcount() + 1
plot_df4 = man_city.groupby(['position','Rank']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Correcting the positions
def position_groups(series):
    if series == 'SS':
        return 'CF'
    elif series == 'LM':
        return 'LW'
    elif series == 'RM':
        return 'RW'
    else:
        return series

dataset['position'] = dataset['position'].apply(position_groups)


# In[ ]:


#Graph 6 - Getting the market price per position from the top 6 teams
top_6 = dataset[dataset['big_club'] == 1]
top_6 = top_6.groupby(['position','club']).agg({'market_value':'sum'})
top_6 = top_6.groupby(['position']).agg({'market_value':'mean'})
top_6.reset_index(level=0, inplace=True)


# In[ ]:


#Graph 7 - Superimposing the three graphs
col = sns.color_palette("YlOrRd")
fig, ax = plt.subplots()
ax2 = ax.twiny()
plot_df3.plot(kind='bar',width = 1, stacked=True,ax=ax2,alpha = 0.5,color = col)
plot_df4.plot(kind='bar',width = 1,stacked=True, edgecolor='black',legend = False, color = 'white',ax=ax)
plt.scatter(top_6['position'],top_6['market_value'],color = 'darkblue')
plt.legend(loc=1, prop={'size': 5})

NOTE THAT IN THE ABOVE PLOT:
1. Each box denotes a Man City player and his market value
2. The color palette denotes the age group
3. The blue dots denote the average market value of the top 6 clubsIdeally you would like two people in each position so that enough cover is there for each position.
# In[ ]:


#ATTACKING MIDFIELDER:
man_city_am = man_city[man_city['position'] == "AM"]
man_city_am

Selling David Silva makes sense, and we can get someone cheaper and younger as a replacement.
# In[ ]:


dataset_am = dataset[(dataset['position'] == "AM") & (dataset['age'] < 31) & (dataset['market_value'] < 30)
                    & (dataset['new_signing'] == 0)]
dataset_am

Kasey Palmer and Manuel Lanzini will make it into the shortlist for Man City as Silva's replacement. What favours Lanzini is he is a better backup than Palmer if De Bruyne is injured and what favours Palmer is that as De Bruyne is too good freeing up funds elsewhere makes sense.
# In[ ]:


#CENTER BACK:
man_city_cb = man_city[man_city['position'] == "CB"]
man_city_cb

Let's see what options do we have
# In[ ]:


dataset_cb = dataset[(dataset['position'] == "CB") & (dataset['age'] < 29) & (dataset['market_value'] > 20)
                    & (dataset['new_signing'] == 0)]
dataset_cb

As there are plenty of options, selling Kompany and Otamendi makes sense and you only need one member as a replacement. As Stones is young you would need someone experienced to partner him and at the same time he shouldn't be that old and pricy like Alderweireld so Van Dijk is your best bet. 
# In[ ]:


#CENTER FORWARD:
man_city_cf = man_city[man_city['position'] == "CF"]
man_city_cf

Even though Aguero is 29 years old, it doesn't make sense to sell him as Man City does not have other premium center forwards. Gabriel Jesus has a good market value at 20 years old so his market value will improve over time. Kelechi Iheanacho seems to be an extra back up so he can be sold without buying other center forwards.
# In[ ]:


#CENTER MIDFIELDER
man_city_cm = man_city[man_city['position'] == "CM"]
man_city_cm


# In[ ]:


dataset_cm = dataset[(dataset['position'] == "CM") & (dataset['age'] < 30) & (dataset['market_value'] > 20)
                    & (dataset['new_signing'] == 0)]
dataset_cm

As far as Center Midfielders are concerned, we only one solid player, so we can afford to keep Delph as a backup to Gundogan and need not splurge cash on CMs. We therefore only need to sell Yaya Toure. We can't sell Gundogan because he is a new signing.
# In[ ]:


#DEFENSIVE MIDFIELDER
man_city_dm = man_city[man_city['position'] == "DM"]
man_city_dm

Ok, this is bad. Both of them need to leave. So two new DMs are required.
# In[ ]:


dataset_dm = dataset[(dataset['position'] == "DM") & (dataset['age'] < 30) & (dataset['market_value'] > 10)
                    & (dataset['new_signing'] == 0)]
dataset_dm

Dier will be the first pick as he is young and has a decent market value. A backup option will be Imbula as he is young and comes at the lowest price is the list. And a Stoke player would love to get opportunities in a big club.
# In[ ]:


#GOALKEEPERS
man_city_gk = man_city[man_city['position'] == "GK"]
man_city_gk

As Bravo is a new signing we cannot sell him.
# In[ ]:


#LEFT BACK
man_city_lb = man_city[man_city['position'] == "LB"]
man_city_lb

Alright, we can sell Kolarov, let's see options
# In[ ]:


dataset_lb = dataset[(dataset['position'] == "LB") & (dataset['age'] < 30)
                    & (dataset['new_signing'] == 0)]
dataset_lb

Ben Davies is a viable option as he plays for a big club and is just a backup to Danny Rose (as Danny Rose has a higher market value to Danny Rose) so he will be willing to sign for Man City as a first choice. The second option we can go for Ben Chilwell as a budget pick
# In[ ]:


#LEFT WING
man_city_lw = man_city[man_city['position'] == "LW"]
man_city_lw


# In[ ]:


#Let's see if we can get greedy and go for one good pick and one backup pick
dataset_lw = dataset[(dataset['position'] == "LW") & (dataset['age'] < 30) & (dataset['market_value'] > 15)
                    & (dataset['new_signing'] == 0)]
dataset_lw

Sanchez and Hazard are the only people more expensive than Sterling and we can't sell Sane as he is a new signing for Man CityChecking potential targets:
# In[ ]:


dataset_marquee = dataset[(dataset['age'] < 30) & (dataset['market_value'] > 40)
                    & (dataset['new_signing'] == 0)]
dataset_marquee


# In[ ]:


#RIGHT BACK
man_city_rb = man_city[man_city['position'] == "RB"]
man_city_rb

We need another RB option as a backup
# In[ ]:


dataset_rb = dataset[(dataset['position'] == "RB") & (dataset['age'] < 25) & (dataset['market_value'] < 20)
                    & (dataset['new_signing'] == 0)]
dataset_rb

Yep, let's go for Trent Alexander Arnold
# In[ ]:


#RIGHT WINGER
man_city_rw = man_city[man_city['position'] == "RW"]
man_city_rw

Bernardo is great, we can scout premium picks though, just to be greedy
# In[ ]:


dataset_rw = dataset[(dataset['position'] == "RW") & (dataset['age'] < 30) & (dataset['market_value'] >= 40)
                    & (dataset['new_signing'] == 0)]
dataset_rw

Nope, he is pretty much our best option under the age of 30 now to scout a backup
# In[ ]:


dataset_rw = dataset[(dataset['position'] == "RW") & (dataset['age'] < 25) & (dataset['market_value'] < 20)
                    & (dataset['new_signing'] == 0)]
dataset_rw

We can go with Jordon Ibe as he has a good market value at 21 years old and Zaha is expensive for a backup option.
# In[ ]:


#MAN CITY TEAM VALUE BEFORE TRANSFERS
team_value = man_city['market_value'].sum()
team_value


# In[ ]:


#TRANSFERS DRAFT 1
man_city_new = man_city[~man_city.name.isin(['David Silva','Nicolas Otamendi',
                                             'Vincent Kompany','Yaya Toure','Kelechi Iheanacho',
                                            'Fernandinho','Fernando',
                                            'Aleksandar Kolarov'])]


# In[ ]:


dataset_man_city = dataset[dataset.name.isin(['Manuel Lanzini','Virgil van Dijk','Eric Dier','Trent Alexander-Arnold',
                                             'Jordon Ibe','Ben Davies','Ben Chilwell',
                                             'Giannelli Imbula'])]


# In[ ]:


team_value_new = man_city_new['market_value'].sum() + dataset_man_city['market_value'].sum()
team_value_new

Okay, so we need to replot the graph to check where we can spend more money
# In[ ]:


man_city_2 = man_city_new.append(dataset_man_city)
man_city_2 = man_city_2.drop(['England_Flag','Rank'],axis = 1)
man_city_2


# In[ ]:


#Graph 8 - Getting the colors
plot_df5 = man_city_2.groupby(['position','age_cat']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 9 - Getting the counts
man_city_2['Rank'] = man_city_2.sort_values(['age_cat','market_value'], ascending=[True,False])              .groupby(['position'])              .cumcount() + 1
plot_df6 = man_city_2.groupby(['position','Rank']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 10 - Superimposing the three graphs
col = sns.color_palette("YlOrRd")
fig, ax = plt.subplots()
ax2 = ax.twiny()
plot_df5.plot(kind='bar',width = 1, stacked=True,ax=ax2,alpha = 0.5,color = col)
plot_df6.plot(kind='bar',width = 1,stacked=True, edgecolor='black',legend = False, color = 'white',ax=ax)
plt.scatter(top_6['position'],top_6['market_value'],color = 'darkblue')
plt.legend(loc=1, prop={'size': 5})


# In[ ]:


dataset_cb = dataset[(dataset['position'] == "CB") & (dataset['age'] < 25) & (dataset['market_value'] < 6)
                    & (dataset['new_signing'] == 0)]
dataset_cb

Retrospecting the transfers to accomodate the mean market value of the Top 6
1. Upgrading Ben Chilwell to Luke Shaw
2. Upgrading Fabian Delph to Aaron Ramsey
3. Downgrading Raheem Sterling to Son Heung Min
4. Upgrading Eric Dier to Granit Xhaka (Otherwise there will be three Tottenham signings)
5. Add Mason Holgate as the third center back as he comes at 5 for an age of 20
6. Jan Bednarek can be bought as the fourth center back for the remaining 0.5 balance cash we have
# In[ ]:


#TRANSFERS DRAFT 2
man_city_new_2 = man_city_2[~man_city_2.name.isin(['Ben Chilwell','Fabian Delph','Raheem Sterling','Eric Dier'])]
dataset_man_city_2 = dataset[dataset.name.isin(['Luke Shaw','Aaron Ramsey','Son Heung-min','Granit Xhaka','Mason Holgate','Jan Bednarek'])]


# In[ ]:


team_value_new = man_city_new_2['market_value'].sum() + dataset_man_city_2['market_value'].sum()
team_value_new


# In[ ]:


man_city_3 = man_city_new_2.append(dataset_man_city_2)
man_city_3 = man_city_3.drop(['Rank'],axis = 1)
man_city_3


# In[ ]:


#Graph 11 - Getting the colors
plot_df7 = man_city_3.groupby(['position','age_cat']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 12 - Getting the counts
man_city_3['Rank'] = man_city_3.sort_values(['age_cat','market_value'], ascending=[True,False])              .groupby(['position'])              .cumcount() + 1
plot_df8 = man_city_3.groupby(['position','Rank']).agg({'market_value':'sum'}).unstack()


# In[ ]:


#Graph 13 - Superimposing the three graphs
col = sns.color_palette("YlOrRd")
fig, ax = plt.subplots()
ax2 = ax.twiny()
plot_df7.plot(kind='bar',width = 1, stacked=True,ax=ax2,alpha = 0.5,color = col)
plot_df8.plot(kind='bar',width = 1,stacked=True, edgecolor='black',legend = False, color = 'white',ax=ax)
plt.scatter(top_6['position'],top_6['market_value'],color = 'darkblue')
plt.legend(loc=1, prop={'size': 5})

This is the best Man City can do to have a well balanced squad with a good mix of youngsters and experience with the current budget.
All the assumptions are satisfied:
1. Five English players
2. Not more than two players from the same club etc.MANCHESTER CITY NEW SQUAD (4-3-3)

                                  Ederson/Bravo
                                  
      Walker/Trent     Van Dijk/Holgate             Stones/Bednarek       Shaw/Davies
         
                                  Xhaka/Imbula

                    Gundogan/Ramsey
                          
                                          De Bruyne/Lanzini
                                                                   
         Bernardo/Ibe                                       Sane/Heung-min Son
         
                                   Aguero/Jesus
                                          
 