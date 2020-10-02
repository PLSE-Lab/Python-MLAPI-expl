#!/usr/bin/env python
# coding: utf-8

#  **Caption:**<br>
# &emsp; Here is the code that I worked on the dataset 'appstore_games.csv' for providing insights in April 2020. @Tristan #Kaggle
#     <br>
#  **Description:**<br>
# &emsp; This code is effective and helps to strategize by giving insights on different types of apps. I am excited to share my work on the dataset provided by @Tristan #Kaggle.

# In[ ]:


import numpy as np
import pandas as pd
import datetime


# In[ ]:


gamestats = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

gamestats.columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon_URL',                      'Average_User_Rating', 'User_Rating_Count', 'Price',                      'In_App_Purchases', 'Description', 'Developer',                      'Age_Rating', 'Languages', 'Size', 'Primary_Genre',                      'Genres', 'Original_Release_Date', 'Current_Version_Release_Date']


# In[ ]:


gamestats.dropna(subset=['Average_User_Rating'], how='any', inplace = True)
gamestats


# In[ ]:


low_user_count = gamestats[gamestats['User_Rating_Count'] < 200].index

gamestats.loc[:, 'Current_Version_Release_Date'] = pd.to_datetime(gamestats['Current_Version_Release_Date'])
gamestats.loc[:, 'Original_Release_Date'] = pd.to_datetime(gamestats['Original_Release_Date'])
gamestats.loc[:, 'Last_Updated'] = gamestats['Current_Version_Release_Date'] - gamestats['Original_Release_Date']

gamestats.drop((low_user_count & gamestats[gamestats.Last_Updated < datetime.timedelta(days=175)].index), inplace = True)
gamestats


# In[ ]:


gamestats['Genres'] = gamestats.Genres.str.replace('Games', '').str.replace('Entertainment', '').str.replace('Strategy', '').str.replace(',', '')

gamestats.loc[gamestats['Genres'].str.contains('Puzzle'),'Genres'] = 'Puzzle'
gamestats.loc[gamestats['Genres'].str.contains('Board'),'Genres'] = 'Puzzle'
gamestats.loc[gamestats['Genres'].str.contains('Adventure'),'Genres'] = 'Adventure'
gamestats.loc[gamestats['Genres'].str.contains('Role'),'Genres'] = 'Adventure'
gamestats.loc[gamestats['Genres'].str.contains('Action'),'Genres'] = 'Action'
gamestats.loc[gamestats['Genres'].str.contains('Family'),'Genres'] = 'Family'
gamestats.loc[gamestats['Genres'].str.contains('Education'),'Genres'] = 'Family'

gamestats.drop(gamestats[~gamestats.Genres.str.contains('Puzzle')                        & ~gamestats.Genres.str.contains('Adventure')                        & ~gamestats.Genres.str.contains('Action')                        & ~gamestats.Genres.str.contains('Family')].index, inplace = True)


# In[ ]:


gamestats.drop(['URL', 'ID', 'Subtitle', 'Icon_URL', 'Description', 'Developer', 'Primary_Genre'], axis=1, inplace = True)
gamestats


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = ['Puzzle', 'Adventure', 'Action', 'Family']
y = [gamestats.Average_User_Rating[(gamestats['Genres']=='Puzzle')].mean(),     gamestats.Average_User_Rating[(gamestats['Genres']=='Adventure')].mean(),     gamestats.Average_User_Rating[(gamestats['Genres']=='Action')].mean(),     gamestats.Average_User_Rating[(gamestats['Genres']=='Family')].mean()]

plot1 = sns.barplot(x, y)
plot1.set(xlabel = 'Genres', ylabel = 'Average User Rating', ylim = (3.5, 5), title = 'Average User Ratings for a Genre')
for p in plot1.patches:
    plot1.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 20),textcoords='offset points')


# From the above graph, the following is the order of **Genres** having higher Average User Ratings from top to bottom:
# 1. Family - 4.24
# 2. Action - 4.22
# 3. Adventure - 4.15
# 4. Puzzle - 4.04

# In[ ]:


gamestats.Genres.value_counts()


# In[ ]:


x = ['Puzzle', 'Adventure', 'Action', 'Family']
y = [gamestats.Genres[gamestats['Genres']=='Puzzle'].count(),     gamestats.Genres[gamestats['Genres']=='Adventure'].count(),     gamestats.Genres[gamestats['Genres']=='Action'].count(),     gamestats.Genres[gamestats['Genres']=='Family'].count()]

plot2 = sns.barplot(x, y)
plot2.set(xlabel = 'Genres', ylabel = 'Number of Games', title = 'Genre Grouping')
for p in plot2.patches:
    plot2.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 20),textcoords='offset points')


# There are more number of Games with Genre **Puzzle**

# In[ ]:


gamestats['Size_in_MB'] = round(gamestats['Size']/1000000,1)
no_of_games_below_250MB = gamestats.Size[gamestats['Size_in_MB'] < 250]
no_of_games_between_250MB_and_1GB = gamestats.Size[(gamestats['Size_in_MB'] >= 250) & gamestats['Size_in_MB'] < 1000]
no_of_games_above_1GB = gamestats.Size[gamestats['Size_in_MB'] >= 1000]

print('Number of Games below 250 MB:', no_of_games_below_250MB.count())
print('Number of Games between 250 MB and 1 GB:', no_of_games_between_250MB_and_1GB.count())
print('Number of Games above 1 GB:', no_of_games_above_1GB.count())


# In[ ]:


x = ['Size < 250MB', '250MB <= Size < 1GB', 'Size >= 1GB']
y = [no_of_games_below_250MB.count(), no_of_games_between_250MB_and_1GB.count(), no_of_games_above_1GB.count()]
plot3 = sns.barplot(x, y)
plot3.set(ylabel = 'Number of Games')
for p in plot3.patches:
    plot3.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 20),textcoords='offset points')


# There are more number of Games which have size between 250MB and 1GB

# In[ ]:


f, axes = plt.subplots (2, 3, figsize=(18,12))

below_250MB = gamestats[gamestats['Size_in_MB'] < 250]
between_250MB_and_1GB = gamestats[(gamestats['Size_in_MB'] >= 250) & (gamestats['Size_in_MB'] < 1000)]
above_1GB = gamestats[gamestats['Size_in_MB'] >= 1000]

plot4 = sns.boxplot(data = below_250MB, x = 'Average_User_Rating', y = 'Size_in_MB',  ax=axes[1,0])
plot5 = sns.boxplot(data = between_250MB_and_1GB, x = 'Average_User_Rating', y = 'Size_in_MB', ax=axes[1,1])
plot6 = sns.boxplot(data = above_1GB, x = 'Average_User_Rating', y = 'Size_in_MB', ax=axes[1,2],)

plot4.set(xlabel='User Rating',ylabel='Game Size in MB')
plot5.set(xlabel='User Rating',ylabel='Game Size in MB')
plot6.set(xlabel='User Rating',ylabel='Game Size in MB')

plot7 = sns.distplot(below_250MB.Size_in_MB, bins= 10, kde=False, ax=axes[0,0])
plot8 = sns.distplot(between_250MB_and_1GB.Size_in_MB, bins= 10, kde=False, ax=axes[0,1])
plot9 = sns.distplot(above_1GB.Size_in_MB, bins= 10, kde=False, ax=axes[0,2])

plot7.set(xlabel='Game Size in MB',ylabel='Number of Games')
plot8.set(xlabel='Game Size in MB',ylabel='Number of Games')
plot9.set(xlabel='Game Size in MB',ylabel='Number of Games')

axes[0, 0].set_title('No. of Games Below 250MB')
axes[0, 1].set_title('No. of Games Between 250MB and 1GB')
axes[0, 2].set_title('No. of Games Above 1GB')
axes[1, 0].set_title('Games Below 250MB vs User Rating')
axes[1, 1].set_title('Games Between 250MB and 1GB vs User Rating')
axes[1, 2].set_title('Games Above 1GB vs User Rating')

plot7.set(ylim=(0, 800))
plot8.set(ylim=(0, 800))
plot9.set(ylim=(0, 800))


# In[ ]:


plot10 = sns.stripplot(y='Size_in_MB' , x='Genres', data=gamestats,                hue='Average_User_Rating',dodge=True, size=4)

plt.legend(title='User Rating', bbox_to_anchor=(1.29, 1))

plot10.set(ylim=(0, 1000))
plot10.set(xlabel='User Rating', ylabel='Game Size in MB')


# There are many games above 250MB with better User Rating

# In[ ]:


gamestats.loc[:, 'Year'] = gamestats['Original_Release_Date'].astype(str)


# In[ ]:


for i in (list(gamestats.index.values)):
    gamestats.loc[i, 'Year']=gamestats.loc[i, 'Year'][:4]
    
f, axes = plt.subplots (3, 2, figsize=(18,20))

plot11 = sns.lineplot(x=gamestats.Year, y=gamestats.Size_in_MB, data=gamestats, ax=axes[0][0])
plot11.set(ylim=(0, 600))
plot12 = sns.lineplot(x=gamestats.Year, y=gamestats.Size_in_MB, hue=gamestats.Genres, err_style=None, marker='o', ax=axes[0][1])
plot12.set(ylim=(0, 600))

axes[0][0].set_title('Size of Games in each Year')
axes[0][1].set_title('Size of Games in each Year by Genre')

plot11.set(xlabel='Year', ylabel='Game Size in MB')
plot12.set(xlabel='Year', ylabel='Game Size in MB')

plot13 = sns.lineplot(x=gamestats.Year, y=gamestats.User_Rating_Count, data=gamestats, ax=axes[1][0])

plot14 = sns.lineplot(x=gamestats.Year, y=gamestats.User_Rating_Count, hue=gamestats.Genres, err_style=None, marker='o', ax=axes[1][1])

axes[1][0].set_title('User Rating Count in each Year')
axes[1][1].set_title('User Rating Count in each Year by Genre')

plot13.set(xlabel='Year', ylabel='User Rating Count')
plot14.set(xlabel='Year', ylabel='User Rating Count')

plot15 = sns.lineplot(x=gamestats.Year, y=gamestats.Average_User_Rating, data=gamestats, ax=axes[2][0])

plot16 = sns.lineplot(x=gamestats.Year, y=gamestats.Average_User_Rating, hue=gamestats.Genres, err_style=None, marker='o', ax=axes[2][1])

axes[2][0].set_title('Average User Rating in each Year')
axes[2][1].set_title('Average User Rating in each Year by Genre')

plot15.set(xlabel='Year', ylabel='Average User Rating')
plot16.set(xlabel='Year', ylabel='Average User Rating')


# In[ ]:


for i in (list(gamestats.index.values)):
    gamestats.loc[i, 'Last_Updated'] = gamestats.loc[i, 'Last_Updated'].days


# In[ ]:


x=['Puzzle','Adventure','Action','Family']
y = [gamestats.Last_Updated[(gamestats['Genres']=='Puzzle')].mean(),     gamestats.Last_Updated[(gamestats['Genres']=='Adventure')].mean(),     gamestats.Last_Updated[(gamestats['Genres']=='Action')].mean(),     gamestats.Last_Updated[(gamestats['Genres']=='Family')].mean()]

plot17 = sns.barplot(x,y)
plot17.set(xlabel = 'Genre', ylabel = 'Mean of Last Updated (In Days)', title = 'Last Updated factor on User Ratings')

for p in plot17.patches:
             plot17.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height() - p.get_height()),
                 ha='center', va='center', fontsize=11, color='white', xytext=(0, 20),
                 textcoords='offset points')


# The Average no. of days last updated is more in Puzzle than other Genres.

# In[ ]:


gamestats.Price.value_counts()


# In[ ]:


plot18 = gamestats.Price.value_counts().plot(kind='bar')


# There are many Free Games.

# In[ ]:


plot19 = sns.lineplot(x = gamestats.Price, y = gamestats.Average_User_Rating, data = gamestats)
plot19.set(title = 'User Rating Count VS Price', xlabel = 'Price (In $)', ylabel = 'User Rating Count')


# The User Ratings are better for paid games than free games.
# The User Ratings are less on Games when the Price of Games is more than 12.5 dollars.

# In[ ]:


x=['Puzzle','Adventure','Action','Family']
y = [gamestats.Price[(gamestats['Genres']=='Puzzle')].mean(),     gamestats.Price[(gamestats['Genres']=='Adventure')].mean(),     gamestats.Price[(gamestats['Genres']=='Action')].mean(),     gamestats.Price[(gamestats['Genres']=='Family')].mean()]

plot20 = sns.barplot(x,y)
plot20.set(xlabel = 'Genre', ylabel = 'Average Price', title = 'Average Price of Games in Each Genre')

for p in plot20.patches:
             plot20.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height() - p.get_height()),
                 ha='center', va='center', fontsize=11, color='white', xytext=(0, 20),
                 textcoords='offset points')


# The Average Price of Games is more in Puzzle.

# In[ ]:


plot21 = sns.lineplot(x = gamestats.Price, y = gamestats.User_Rating_Count, data = gamestats)
plot21.set(title = 'User Rating Count VS Price', xlabel = 'Price (In $)', ylabel = 'User Rating Count')


# In[ ]:


plot22 = sns.jointplot(x = 'Price', y = 'Average_User_Rating', data = gamestats)


# In[ ]:


plot23 = sns.lineplot(x=gamestats.Price, y=gamestats.Average_User_Rating, hue=gamestats.Genres, err_style=None, marker='o')
plot23.set(title = 'User Rating VS Price (For Each Genre)', xlabel = 'Price (In $)', ylabel = 'User Rating', ylim = (3,5))


# In[ ]:


gamestats['In_App_Purchases'] = gamestats['In_App_Purchases'].str.replace(',', '') 
gamestats.In_App_Purchases = gamestats.In_App_Purchases.fillna(0)

import statistics

for i in (list(gamestats.index.values)):
    if gamestats.In_App_Purchases[i] == 0:
        gamestats.loc[i, 'In_App_Purchases'] = 0.01
    elif gamestats.In_App_Purchases[i] != 0:
        A = str.split(gamestats.In_App_Purchases[i])
        A = [float(i) for i in A]
        gamestats.loc[i, 'In_App_Purchases']=statistics.mean(A)
plot24 = sns.stripplot(y=gamestats.In_App_Purchases , x=gamestats.Price, data=gamestats, hue=gamestats.Genres, dodge = True, size = 5)
plot24.set(xlim=(-1, 12))
plot24.set(xlabel = 'Price (In $)', ylabel = 'Average In App Purchases (In $)')


# There are many games with less than 1 dollar.

# In[ ]:


gamestats.Age_Rating.value_counts()


# In[ ]:


plot25 = gamestats.Age_Rating.value_counts().plot(kind='bar')


# In[ ]:


gamestats.Age_Rating[gamestats.Genres=='Puzzle'].value_counts()


# In[ ]:


gamestats.Age_Rating[gamestats.Genres=='Adventure'].value_counts()


# In[ ]:


gamestats.Age_Rating[gamestats.Genres=='Action'].value_counts()


# In[ ]:


gamestats.Age_Rating[gamestats.Genres=='Family'].value_counts()


# In[ ]:


f, axes = plt.subplots (2, 2, figsize=(15,15))

plot26 = gamestats.Age_Rating[gamestats.Genres=='Puzzle'].value_counts().plot(kind='bar', ax = axes[0][0], color = '#E41A7F')
plot26.set(title = 'Puzzle', xlabel = 'Age Rating', ylabel = 'Count')
plot27 = gamestats.Age_Rating[gamestats.Genres=='Adventure'].value_counts().plot(kind='bar', ax = axes[0][1], color = '#123456')
plot27.set(title = 'Adventure', xlabel = 'Age Rating', ylabel = 'Count')
plot28 = gamestats.Age_Rating[gamestats.Genres=='Action'].value_counts().plot(kind='bar', ax = axes[1][0], color = '#095730')
plot28.set(title = 'Action', xlabel = 'Age Rating', ylabel = 'Count')
plot29 = gamestats.Age_Rating[gamestats.Genres=='Family'].value_counts().plot(kind='bar', ax = axes[1][1], color = 'orange')
plot29.set(title = 'Family', xlabel = 'Age Rating', ylabel = 'Count')


# In[ ]:


plot30 = gamestats.Age_Rating[gamestats.Genres=='Puzzle'].value_counts().plot(kind='line', marker = 'o', label = 'Puzzle')
plot31 = gamestats.Age_Rating[gamestats.Genres=='Adventure'].value_counts().plot(kind='line', marker = 'o', label = 'Adventure')
plot32 = gamestats.Age_Rating[gamestats.Genres=='Action'].value_counts().plot(kind='line', marker = 'o', label = 'Action')
plot33 = gamestats.Age_Rating[gamestats.Genres=='Family'].value_counts().plot(kind='line', marker = 'o', label = 'Family')
plt.legend(title = 'Genres')
plot33.set(title = 'Age Rating Factor on No_of_games', xlabel = 'Age Rating', ylabel = 'Number of Games')


# In[ ]:


plot34 = sns.boxplot(x = gamestats.Age_Rating, y = gamestats.Average_User_Rating, data = gamestats, hue = gamestats.Genres, dodge = True)
plot34.set(xlabel = 'Age Rating', ylabel = 'User Rating', title = 'Age Rating Factor on User Ratings')
plt.legend(title = 'Genre', bbox_to_anchor = (1.3, 1))


# There are high User Ratings when Age Rating is 17+.

# In[ ]:


gamestats.dtypes


# In[ ]:


gamestats


# My Inferences:
# 
#     1. Family Games have high Average User Ratings.
#     2. It is better to develop games with size between 250MB and 1GB.
#     3. Puzzle Games should be less than 500MB.
#     4. Puzzle and Action Games are good for competition.
#     5. There are many Puzzle Games in the industry.
#     6. Puzzle Games can have less game updates.
#     7. As the year increases, the size of games are increases and also the Average User Rating.
#     8. Paid Games have better Ratings when they are priced below 12.5 dollars.
#     9. In-App Purchases should be less than 2 dollars for better User Rating.
#     10. Games with Age Rating 17+ have good User Rating.
#     11. The following order of Genres have high Average User Ratings from top to bottom:
#         a. Family - 4.24
#         b. Action - 4.22
#         c. Adventure - 4.15
#         d. Puzzle - 4.04
