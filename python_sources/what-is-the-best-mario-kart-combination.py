#!/usr/bin/env python
# coding: utf-8

# ## Hello and welcome everybody! :D
# 
# ## Today, we'll explore the Mario Kart Datasets: 
# * Karts, 
# * Characters
# * Tires
# * Gliders
# 
# ## Here you can learn a little bit of data manipulation, working with pandas, EDA (Exploratory Data Analysis) and, finally, answer the question:
# 
# # What is the best kart combination for a determined playstyle?
# 
# ![fig](http://s2.glbimg.com/J0c0C48qmT7IIlKhlgI6QBIO7eY=/695x0/s.glbimg.com/po/tt2/f/original/2017/03/10/000303.jpg)

# In[ ]:


#Importing our packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Reading our datasets

characters = pd.read_csv('/kaggle/input/characters.csv')
tires = pd.read_csv('/kaggle/input/tires.csv')
bodies = pd.read_csv('/kaggle/input/bodies.csv')
gliders = pd.read_csv('/kaggle/input/gliders.csv')

#Inspecting
characters.head()


# In[ ]:


tires.head()


# In[ ]:


bodies.head()


# In[ ]:


gliders.head()


# Since we have different speeds for different terrains, my first idea was to summarize them by the mean function but I thought that this wouldn't be fair, after all, there are much more 'normal ground' than water ou air. 
# #### Let's investigate this behavior of different speeds

# In[ ]:


characters[characters['Speed'] != characters['Speed (Air)']]
#characters[characters['Speed'] != characters['Speed (Water)']]   ----
#characters[characters['Speed'] != characters['Speed (Ground)']]  ---- These lines generated the same results 


# - only 5 cars have different speeds ^^

# In[ ]:


tires[tires['Speed'] != tires['Speed (Air)']]


# In[ ]:


tires[tires['Speed'] != tires['Speed (Water)']]


# In[ ]:


tires[tires['Speed'] != tires['Speed (Ground)']]


# **Ok, so with tires we have more issues**
# 
# Unfortunately, I don't know the exact distribution ratio of terrain types per track, so I will arbitrarily assign the following values:
# 
# * Speed          = 0.75
# * Speed (Water)	 = 0.1
# * Speed (Air)	 = 0.05
# * Speed (Ground) = 0.1
# 

# In[ ]:


def new_speed(row):
    return row['Speed'] * 0.75 + row['Speed (Water)'] * 0.1 + row['Speed (Air)'] * 0.05 + row['Speed (Ground)'] * 0.1

characters['new_speed'] = characters.apply(new_speed, axis=1)
characters.drop(['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)'], axis=1, inplace=True)

characters.head()


# In[ ]:


#Now let's do similar to the Handling factor:
def new_handling (row):
    return row['Handling'] * 0.75 + row['Handling (Water)'] * 0.1 + row['Handling (Air)'] * 0.05 + row['Handling(Ground)'] * 0.1

characters['new_handling'] = characters.apply(new_handling, axis=1)
characters.drop(['Handling', 'Handling (Water)', 'Handling (Air)', 'Handling(Ground)'], axis=1, inplace=True)

characters.head()


# Ok, now we have our datasets more simple
# 
# # Before we start doing some combinations of cars, tires and everything else, let's evaluate only the characters:

# In[ ]:


#fig = go.Figure(data=go.Heatmap(
#                   z=characters.groupby('Class').mean().values,
#                   x=characters.groupby('Class').mean().columns,
#                   y=characters.groupby('Class').mean().index,
#                   colorscale='Viridis'
#))
#
#fig.show()

# --- Just left this scrap here because it could be useful another time


# In[ ]:


# Evaluating the characters by class:

plt.figure(figsize=(17,7))
sns.set(font_scale=1.4)
sns.heatmap(characters.groupby('Class').mean(),
           linewidths=1,
           annot=True,
           fmt=".1f",
           cmap='viridis')
plt.title('Heatmap of Classes')


# Well, as expected, Light have better handling and acceleration, while Heavy wins in speed. Medium are medium, hehehe

# In[ ]:


#how speed, acceleration and the class are distribuited?
fig = px.scatter(characters, x="new_speed", y="Acceleration", color="Class",hover_data=['Character'], size='Mini Turbo')
fig.show()


# There is a direct inverse-relation between speed and acceleration (only Pink gold Peach and Metal Mario aren't part of this rule, because of reasons, hehehe)
# 
# Other interesting fact,
# # Even with all 28 characters, we are only choosing between 7 groups of stats values
# 
# (I choose luigi anyway 'cause he looks badass)

# In[ ]:


# Who are the five best of each factor?

factors = ['Acceleration', 'Weight', 'Traction', 'Mini Turbo', 'new_speed', 'new_handling']
for factor in factors:
    print('')
    print('Factor: {}'.format(factor))
    print('')
    print(characters.sort_values(by=[factor], ascending=False).head()['Character'])


# too much to read... 
# 
# # We want some value that summarize the overall effectiveness of the kart.
# 
# **This time I'll use my 'just for fun' gameplay in Mario Kart to assign the following weights:**
# 
# * new_speed      = 0.3
# * new_handling	 = 0.1
# * Mini Turbo	 = 0.3
# * Traction       = 0.1
# * Acceleration   = 0.2
# 
# **Ps.: As you can see, I really like using Mini Turbos... But feel free to commit and change this values according to your playstyle **

# In[ ]:


def effectiveness(row):
    return row['new_speed'] * 0.3 + row['new_handling'] * 0.1 + row['Mini Turbo'] * 0.3 + row['Traction'] * 0.1 + row['Acceleration'] * 0.2

characters['effectiveness'] = characters.apply(effectiveness, axis=1)

characters.head()


# In[ ]:


characters.sort_values(by=['effectiveness'], ascending=False)


# **My favorite kart performed so badly! :(**
# 
# Think I should change some parameters, hehe
# 
# **Here we can visualize again the 7 groups of characters.
# **
# 
# Let's aggregate them to make more easy the next steps, but first, let's reload the original datasets, because we will need them to do all the possible combinations

# In[ ]:


characters = pd.read_csv('/kaggle/input/characters.csv')

characters.drop_duplicates(['Class', 'Speed', 'Speed (Water)', 'Speed (Air)',
       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',
       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',
       'Mini Turbo'])


# In[ ]:


#Okay, let's think in a name for these runners
names =['Babies', 'Toad & Friends', 'Peach/Daisy/Yoshi', "Marios", 'DK/Rosa/Waluig', 'Metal/Gold', 'Heavy Heavies']

characters = characters.drop_duplicates(['Class', 'Speed', 'Speed (Water)', 'Speed (Air)',
       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',
       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',
       'Mini Turbo'])
characters['Character'] = names
characters


# In[ ]:


#Now let's do the same for the components

gliders.drop_duplicates(['Type', 'Speed', 'Speed (Water)', 'Speed (Air)',
       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',
       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',
       'Mini Turbo'], inplace=True)
gliders.drop('Body', axis=1, inplace=True)
gliders 
#Only two actual changes here...


# In[ ]:


tires.drop_duplicates(['Speed', 'Speed (Water)', 'Speed (Air)',
       'Speed (Ground)', 'Acceleration', 'Weight', 'Handling',
       'Handling (Water)', 'Handling (Air)', 'Handling(Ground)', 'Traction',
       'Mini Turbo'], inplace=True)
tires
#7 here...


# In[ ]:


bodies.drop_duplicates(['Speed', 'Acceleration', 'Weight', 'Handling',
       'Traction', 'Mini Turbo'], inplace=True)
bodies
#18 here...


# # Here's the hard part: Combine all datasets 
# 
# **(I would love to receive some tips of how to perform this)**

# In[ ]:


cols = ['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)',
       'Acceleration', 'Weight', 'Handling', 'Handling (Water)',
       'Handling (Air)', 'Handling(Ground)', 'Traction', 'Mini Turbo']

df_fim = pd.DataFrame()
for index, row in gliders.iterrows():
    df_temp = characters.copy()
    df_temp['gliders'] = row['Type']
    for col in cols:
        df_temp[col] = df_temp[col] + row[col]
    df_fim = df_fim.append(df_temp)    


# In[ ]:


aux = df_fim.copy()
df_fim = pd.DataFrame()
for index, row in tires.iterrows():
    df_temp = aux.copy()
    df_temp['tires'] = row['Body']
    for col in cols:
        df_temp[col] = df_temp[col] + row[col]
    df_fim = df_fim.append(df_temp)   


# In[ ]:


cols = ['Speed', 'Acceleration', 'Weight', 'Handling', 'Traction', 'Mini Turbo']
aux = df_fim.copy()
df_fim = pd.DataFrame()
for index, row in bodies.iterrows():
    df_temp = aux.copy()
    df_temp['Vehicle'] = row['Vehicle']
    for col in cols:
        df_temp[col] = df_temp[col] + row[col]
    df_fim = df_fim.append(df_temp)   


# In[ ]:


df_fim.head(10)


# In[ ]:


len(df_fim)


# # We have 1764 possible combinations. O.o
# 
# But we still wanna summarize everything. Let's apply our 3 functions (for speed, handling and effectiveness)

# In[ ]:


df_fim['new_speed'] = df_fim.apply(new_speed, axis=1)
df_fim.drop(['Speed', 'Speed (Water)', 'Speed (Air)', 'Speed (Ground)'], axis=1, inplace=True)

df_fim['new_handling'] = df_fim.apply(new_handling, axis=1)
df_fim.drop(['Handling', 'Handling (Water)', 'Handling (Air)', 'Handling(Ground)'], axis=1, inplace=True)

df_fim['effectiveness'] = df_fim.apply(effectiveness, axis=1)


# # Now the EDA that we all love

# In[ ]:


df_fim.describe()


# In[ ]:


plt.figure(figsize=(17,7))
sns.boxplot(x="Character", y="effectiveness", data=df_fim, palette="Set3")
plt.title('Effectiveness by Characters')


# In[ ]:


sns.pairplot(df_fim, hue="Character")


# In[ ]:


plt.figure(figsize=(17,7))
sns.violinplot(x="tires", y="new_handling", data=df_fim)
plt.title('Tires by handling')


# In[ ]:


plt.figure(figsize=(17,7))
sns.boxplot(x="Character", y="new_speed", hue="gliders", data=df_fim)
plt.title('new_speed and gliders by Character')


# In[ ]:


plt.figure(figsize=(17,7))
sns.violinplot(x="Character", y="new_handling", hue="gliders", data=df_fim)
plt.title('new_handling and gliders by Character')


# In[ ]:


plt.figure(figsize=(17,7))
sns.set(font_scale=1.4)
sns.heatmap(df_fim.corr(),
           linewidths=1,
           annot=True,
           fmt=".1f")
plt.title('Correlation heatmap')


# # And, finally, lets see (according to my parameters), which one are the best and worst karts combinations**

# In[ ]:


#TOP 10
df_fim.sort_values(by=['effectiveness'], ascending=False).head(10)


# In[ ]:


# BOT 10
df_fim.sort_values(by=['effectiveness'], ascending=False).tail(10)


# #### To conclude: an interactive plot of Speed, handling and Acceleration | effectiveness x Turbo

# In[ ]:


fig = px.scatter(df_fim, x="new_speed", y="Acceleration", color="Class",hover_data=['Character', 'tires', 'Vehicle'], size='new_handling')
fig.show()


# In[ ]:


fig = px.scatter(df_fim, x="effectiveness", y="Mini Turbo", color="Class",hover_data=['Character', 'tires', 'Vehicle'], size='new_handling')
fig.show()


# **Looks like my Luigi isn't the best choice for me...**
# 
# **That's explain why I lose every time. hehehe**
# 
# # Thanks for reading, and plis, upvote if you liked! :D
# 
# ![luigi](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d739185e-99c8-4a81-a984-46e93d545928/ddbjpwg-1a6257e9-d8ce-42e4-a4f5-eccd3c8d1d27.png/v1/fill/w_774,h_1032,strp/luigi___full_body_artwork_by_bluetyphoon17_ddbjpwg-pre.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NDAzMiIsInBhdGgiOiJcL2ZcL2Q3MzkxODVlLTk5YzgtNGE4MS1hOTg0LTQ2ZTkzZDU0NTkyOFwvZGRianB3Zy0xYTYyNTdlOS1kOGNlLTQyZTQtYTRmNS1lY2NkM2M4ZDFkMjcucG5nIiwid2lkdGgiOiI8PTMwMjQifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.m1GeCU2NKd29r9DXCxs5RN4HMHu1iDOcVnUTxCxrEcU)[](http://)
