#!/usr/bin/env python
# coding: utf-8

# The FIFA 19 dataset contains over 18,000 rows with 89 columns. Dataset can be found here - https://www.kaggle.com/karangadiya/fifa19

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from IPython.display import HTML


# Creating the dataframe

# In[ ]:


df = pd.read_csv('/kaggle/input/fifa19/data.csv')
df.columns


# In[ ]:


df.info()


# In[ ]:


df.describe() #describes all the numeric values in the dataset


# In[ ]:


df.head()


# In[ ]:


# No. of unique variables in dataset
df.nunique()


# In[ ]:


df.isnull().sum()


# In[ ]:


col = [ 'Name', 'Age', 'Nationality',
       'Overall', 'Club', 'Value', 'Potential','Wage', 'Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Position',
       'Jersey Number', 'Joined',
       'Height', 'Weight', 'Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']


# In[ ]:


#Creating an updated dataframe with required columns
df = pd.DataFrame(df, columns = col)


# In[ ]:


df.head()


# # Data Visualisation

# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(50,40)
plt.show()


# In[ ]:


# Correlation heatmap
plt.rcParams['figure.figsize']=(25,16)
hm=sns.heatmap(df[['Name', 'Age',
       'Overall', 'Potential','Wage', 'Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Position',
       'Jersey Number', 'Joined',
       'Height', 'Weight', 'Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']].corr(), annot = True, linewidths=.5)
hm.set_title(label='Heatmap of dataset', fontsize=20)
hm;

Player distribution across country
# In[ ]:


df_nations = df.groupby(by='Nationality').size().reset_index()
df_nations.columns = ['Nation', 'Count']


# In[ ]:


df_nations[(df_nations['Nation'] == 'England') | (df_nations['Nation'] == 'Wales') 
           | (df_nations['Nation'] == 'Scotland') | (df_nations['Nation'] == 'Northern Ireland') ]


# In[ ]:


df_temp = pd.DataFrame(data= [['United Kingdom', 2148]], columns=['Nation', 'Count'])
df_nations = df_nations.append(df_temp, ignore_index=True)
df_nations.tail()


# Adding values of England, Northern Ireland, Scotland and Wales under United Kingdom as our choropleth map considers the following countries as a whole which is included in the United Kingdom.
# 
# Hover over the map to confirm the same.

# In[ ]:


trace2 = dict(type='choropleth',
              locations=df_nations['Nation'],
              z=df_nations['Count'],
              locationmode='country names',
              colorscale='Portland'
             )

layout = go.Layout(title='<b>Number of Players in each Country</b>',
                   geo=dict(showocean=True,
                            oceancolor='#AEDFDF',
                            projection=dict(type='natural earth'),
                        )
                  )

fig = go.Figure(data=[trace2], layout=layout)
py.iplot(fig)


# Top 5 countries are -
# 
# Engalnd - 1657
# Germany - 1195
# Spain - 1071
# Argentina - 936
# France - 911

# In[ ]:


# Histogram: number of players's age
sns.set(style ="dark", color_codes=True)
x = df.Age
plt.figure(figsize=(12,8))
ax = sns.distplot(x, bins = 58, kde = False, color='g')
ax.set_xlabel(xlabel="Player\'s age", fontsize=16)
ax.set_ylabel(ylabel='Number of players', fontsize=16)
ax.set_title(label='Histogram of players age', fontsize=20)
plt.show()


# Most players age lie in the range of 19 to 29.

# In[ ]:


df['Position'].unique()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Position'].value_counts().plot.pie(explode=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Player Position Distribution')
ax[0].set_ylabel('')
sns.countplot('Position',data=df,ax=ax[1])
ax[1].set_title('Countplot of Player Position')
plt.show()


# Positions which leads are:
# ST - 11.9%
# GK - 11.2%
# CB - 9.8%

# In[ ]:


# The best player per position

display(HTML(df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Name', 'Position']].to_html(index=False)))


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.countplot('Height', data = df, ax=ax[0])
ax[0].set_title('Player Height in Foot')
ax[0].set_ylabel('')
sns.countplot('Weight',data=df,ax=ax[1])
ax[1].set_title('Weight of player in Pounds')
plt.show()


# Most of the players are of height ranging from 5'9" to 6'2"
# while the weight ranges between 150lbs to 174lbs.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Preferred Foot'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Preferred Foot')
ax[0].set_ylabel('')
sns.countplot('Preferred Foot',data=df,ax=ax[1])
ax[1].set_title('Countplot of Preferred Foot by the players')
plt.show()


# Majority of the players are Right footed.

# In[ ]:


df['Work Rate'].unique()


# Relation between preferred and work rate:

# In[ ]:


g = sns.FacetGrid(df, col='Work Rate')
g.map(plt.hist, 'Preferred Foot', bins=20)
plt.figure(figsize = (50,60))


# In[ ]:


# Top five the most expensive clubs
df.groupby(['Club'])['Value'].sum().sort_values(ascending = False).head(5)


# In[ ]:


# Top five the less expensive clubs
df.groupby(['Club'])['Value'].sum().sort_values().head(5)


# In[ ]:


# Top five teams with the best players
df.groupby(['Club'])['Overall'].max().sort_values(ascending = False).head()


# In[ ]:


#Scatter plot between Ball Control and Acceleration
data = pd.concat([df['BallControl'], df['Acceleration']], axis=1)
data.plot.scatter(x='BallControl', y='Acceleration', ylim=(0,50));


# In[ ]:


#Scatter plot between Age and Strength
data = pd.concat([df['Age'], df['Strength']], axis=1)
data.plot.scatter(x='Age', y='Strength', ylim=(0,50));


# In[ ]:


#Scatter plot between Stamina and Positioning
data = pd.concat([df['Stamina'], df['Positioning']], axis=1)
data.plot.scatter(x='Stamina', y='Positioning', ylim=(0,50));


# In[ ]:


#Scatter plot between Age and Potential
data = pd.concat([df['Age'], df['Potential']], axis=1)
data.plot.scatter(x='Age', y='Potential');


# Thank you for reading it till the end. Your feedback is valuable, kindly let me know if there is any kind of changes which can be made to improve the analysis.
