#!/usr/bin/env python
# coding: utf-8

# # EDA App Store Games

# #### To expose the best combination for strategy games available in the appstore in order to get a good user rating (4.0/5.0 and above) 

# Import the numpy, pandas and data visualization libraries.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Read the csv file ('appstore_games.csv') with the appstore games data.

# In[ ]:


games = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv', index_col='Name')


# Lets view the data using .head() and .info().

# In[ ]:


games.head()


# In[ ]:


games.info()


# Dropping all the worthless columns(almost all the columns with object data type).

# In[ ]:


games.drop(['URL', 'ID', 'Subtitle', 'Icon URL', 'In-app Purchases', 'Description', 'Developer', 'Languages', 'Primary Genre', 'Original Release Date'],axis = 1, inplace = True)


# Remove missing values from the dataset.

# In[ ]:


games.dropna(inplace = True)


# Cleaning the data further.

# In[ ]:


games = games[games['User Rating Count'] > 500]


# In[ ]:


games = games[games['Price']<100]


# In[ ]:


games['Age Rating'] = games['Age Rating'].str.rstrip('+').astype('int')


# Converting String to Tuple in Genres column.

# In[ ]:


games['Genres'] = games.Genres.apply(lambda x: tuple(sorted(x.split(', '))))


# Converting the date(string) to tuple so that it can be used in analysis(like month and year of the latest update).

# In[ ]:


games['Current Version Release Date'] = games['Current Version Release Date'].apply(lambda x: x[3:]).apply(lambda x: tuple(sorted(x.split('-'))))


# Scaling down the columns - User Rating Count, Size.

# In[ ]:


games['User Rating Count'] = games['User Rating Count'].apply(lambda x: x/1000000)
games['Size'] = games['Size'].apply(lambda x: x/1048576)


# In[ ]:


games.info()


# In[ ]:


games.head()


# So, here we are done with the data preprocessing. Let's start analysing our data.

# In[ ]:


sns.jointplot('Average User Rating', 'Size', games)


# In[ ]:


sns.jointplot('Average User Rating', 'Price', games)


# Grouping the data by Genres to find out the best genre. 1. Sum the User Rating Count 2. Average the other Columns.

# In[ ]:


count = games.drop(['Price','Size','Age Rating','Average User Rating'], 1).groupby('Genres').sum()


# In[ ]:


games_genre = games.drop('User Rating Count',1).groupby('Genres').mean()


# In[ ]:


Genre_Groups = games_genre.join(count,on = 'Genres').sort_values('User Rating Count',0,False)


# Cleaning the data further(removing insignificant Genres).

# In[ ]:


Genre_Groups = Genre_Groups[Genre_Groups['User Rating Count']>0.1]
Genre_Groups = Genre_Groups[Genre_Groups['Average User Rating']>4].reset_index()


# In[ ]:


Genre_Groups


# ###### From this data, We can see that 'Action, Entertainment, Games, Strategy' is the most popular Genre with 4.3 Avg User Rating.

# In[ ]:


plt.figure(figsize=(10,6))
sns.stripplot('Average User Rating','Genres',  data = Genre_Groups)


# ###### Also, the top five Genres have significantly good Avg User Ratings around 4.3 to 4.5.

# In[ ]:


sns.jointplot('Price', 'Average User Rating', games)


# In[ ]:


sns.jointplot('Price', 'User Rating Count', games)


# ###### from the graph above, we can observe that free or unpaid games are most popular with high ratings.

# ###### So we can conclude that:
# 1. Action Strategy games are the most popular. Simulation and Role-Playing Strategy games are also some popular genres with significant popularity.
# 2. The above genres have good Average User Ratings around 4.3 to 4.5.
# 3. Average size of the games of most popular genres is around 200MB.
# 4. All the popular games are free. The average price of the above genres is less than a dollar.
