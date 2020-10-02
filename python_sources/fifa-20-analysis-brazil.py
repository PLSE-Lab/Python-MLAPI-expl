#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Let's import the dataset

# In[ ]:


data = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")


# Check the dataset

# In[ ]:


data.head(10)


# let's check the shape of the dataframe

# In[ ]:


data.shape 


# In[ ]:


for col in data.columns: 
    print(col)


# Let's check the Brazilian Players list

# ### Data cleaning

# In[ ]:


# filling the missing value for the continous variables for proper data visualization
data['release_clause_eur'].fillna(0,inplace=True)
data['player_tags'].fillna('#Team Player',inplace=True)
data['team_position'].fillna('Not Decided',inplace=True)                                  
data['team_jersey_number'].fillna(0,inplace=True)
data['loaned_from'].fillna('Disclosed',inplace=True)
data['joined'].fillna('Disclosed',inplace=True)
data['contract_valid_until'].fillna('Disclosed',inplace=True)
data['nation_position'].fillna('Not Decided',inplace=True)
data['nation_jersey_number'].fillna(0,inplace=True)
data['pace'].fillna(data['pace'].mean(),inplace=True)
data['shooting'].fillna(data['shooting'].mean(),inplace=True)
data['passing'].fillna(data['passing'].mean(),inplace=True)
data['dribbling'].fillna(data['dribbling'].mean(),inplace=True)
data['defending'].fillna(data['defending'].mean(),inplace=True)
data['physic'].fillna(data['physic'].mean(),inplace=True)
data['gk_diving'].fillna(data['gk_diving'].mean(),inplace=True)
data['gk_handling'].fillna(data['gk_handling'].mean(),inplace=True)
data['gk_kicking'].fillna(data['gk_kicking'].mean(),inplace=True)
data['gk_reflexes'].fillna(data['gk_reflexes'].mean(),inplace=True)
data['gk_speed'].fillna(data['gk_speed'].mean(),inplace=True)
data['gk_positioning'].fillna(data['gk_positioning'].mean(),inplace=True)
data['player_traits'].fillna('Not Analyzed',inplace=True)
data['ls'].fillna('Not Analyzed',inplace=True)
data['st'].fillna('Not Analyzed',inplace=True)
data['rs'].fillna('Not Analyzed',inplace=True)
data['lw'].fillna('Not Analyzed',inplace=True)
data['lf'].fillna('Not Analyzed',inplace=True)
data['cf'].fillna('Not Analyzed',inplace=True)
data['rf'].fillna('Not Analyzed',inplace=True)
data['rw'].fillna('Not Analyzed',inplace=True)
data['lam'].fillna('Not Analyzed',inplace=True)
data['cam'].fillna('Not Analyzed',inplace=True)
data['ram'].fillna('Not Analyzed',inplace=True)
data['lm'].fillna('Not Analyzed',inplace=True)
data['lcm'].fillna('Not Analyzed',inplace=True)
data['cm'].fillna('Not Analyzed',inplace=True)
data['rcm'].fillna('Not Analyzed',inplace=True)
data['rm'].fillna('Not Analyzed',inplace=True)
data['lwb'].fillna('Not Analyzed',inplace=True)
data['ldm'].fillna('Not Analyzed',inplace=True)
data['cdm'].fillna('Not Analyzed',inplace=True)
data['rdm'].fillna('Not Analyzed',inplace=True)
data['rwb'].fillna('Not Analyzed',inplace=True)
data['lb'].fillna('Not Analyzed',inplace=True)
data['lcb'].fillna('Not Analyzed',inplace=True)
data['cb'].fillna('Not Analyzed',inplace=True)
data['rcb'].fillna('Not Analyzed',inplace=True)
data['rb'].fillna('Not Analyzed',inplace=True)


# Drop useless columns

# In[ ]:


data.drop(['sofifa_id', 'player_url', 'dob'], axis=1, inplace=True)
data.head()


# ## **Analyzing Brazil**

# In[ ]:


brazil_data = data['nationality']=='Brazil'
print(brazil_data.head())


# In[ ]:


brazil_data = data[brazil_data]
print(brazil_data.shape)


# In[ ]:


is_brazil.head(10)


# In[ ]:


brazil_data['overall'].describe()


# ### let's see the age disribution

# In[ ]:


# Age Distribution 
plt.figure(figsize=(18,10))
plt.title('Age Distribution in Brazil')
sb.distplot(a=brazil_data['age'], kde=False, bins=10)


# As expected, the plot reveals that we have a lot of youngsters at the team, which is good long-term for the Country.
# 
# 

# Now let's check the overall and the potential, excluding players who have already hit their potential mark

# In[ ]:


brazil_data[(brazil_data['overall'] != brazil_data['potential']) 
            & (brazil_data['age'] <= 25)].sort_values(by='potential', 
            ascending=False)[['short_name', 'age', 'player_positions','overall', 'potential']]


# Team Manager should keep an eye on these players. Because these player are worth investing for long term for the future of the team.

# Now let's check the older players of the country

# In[ ]:


brazil_data[brazil_data['overall'] == brazil_data['potential']][['short_name', 'age', 'overall' ,'value_eur', 'wage_eur']].sort_values(by='age', ascending=False)


# Height and Weight Distribution of the team:

# In[ ]:


# Height Distribution 
plt.figure(figsize=(18,10))
plt.title('Height Distribution in Club')
sb.distplot(a=brazil_data['height_cm'], kde=False)


# In[ ]:


brazil_data['height_cm'].mean()


# In[ ]:


# Weight Distribution 
plt.figure(figsize=(18,10))
plt.title('Weight Distribution in Club')
sb.distplot(a=brazil_data['weight_kg'], kde=False)


# In[ ]:


# mean weight
brazil_data['weight_kg'].mean()


# What is the relationship between Age and Potential and Overall Rating of a Player?

# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(18,10))
sb.regplot(x=brazil_data['age'], y=brazil_data['overall'], ax=ax[0])
sb.regplot(x=brazil_data['age'], y=brazil_data['potential'], ax=ax[1])
ax[0].set_title('Age vs Overall')
ax[1].set_title('Age vs Potential')


# Overall Rating seems to improve with age while Potential Rating reduces with increasing age.

# In[ ]:


# Top 10 left footed footballers

brazil_data[brazil_data['preferred_foot'] == 'Left'][['short_name', 'age', 'club', 'nationality']].head(10)


# In[ ]:


# Top 10 right footed footballers

brazil_data[brazil_data['preferred_foot'] == 'Right'][['short_name', 'age', 'club', 'nationality']].head(10)


# In[ ]:


# comparing the performance of left-footed and right-footed footballers
# ballcontrol vs dribbing

sb.lmplot(x = 'skill_ball_control', y = 'skill_dribbling', data = brazil_data, col = 'preferred_foot')
plt.show()


# In[ ]:





# # **Summery**

# We didn't explore the database with something specific in mind. But we have found some interesting data while exploring Brazil team.
# We have found that Brazil team has more young players.
# Average height of 181 cm.
# Average weight 76 kg.
# We have learned that the more age increses their overall increses but potentiality decreses. 
# We have seen that their are more Right footed players than Left footed players.

# In[ ]:




