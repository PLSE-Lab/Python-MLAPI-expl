#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# In[ ]:


data.corr()>0.5


# In[ ]:


import plotly.express as px


# In[ ]:


fig = px.imshow(data.corr(),x = data.corr().index, y = data.corr().index)


# In[ ]:


fig


# In[ ]:


px.scatter(data, "Defense", "Speed", color = "Legendary", size = "HP")


# In[ ]:


fig = px.density_heatmap(data, x="Defense", y="Attack", marginal_x="histogram", marginal_y="histogram")
fig.show()


# In[ ]:


px.histogram(data, "Defense", color = "Generation")


# In[ ]:


px.histogram(data, "Speed", color = "Generation")


# In[ ]:


data.head()


# In[ ]:


data["GenesHP"] = data.groupby("Generation")["HP"].transform("mean")


# In[ ]:


data.groupby(["Type 1", "Type 2"])[["HP", "Attack", "Speed"]].agg(["mean", "min", "max", "count"])


# In[ ]:


data.head()


# In[ ]:


data["GenesHP"].value_counts()


# In[ ]:


data[data['Defense']>190]


# In[ ]:


data[ (data['Defense']>200) & (data['Attack']>100)]


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data['Type 1'].value_counts()


# In[ ]:


data['Type 1'].nunique()


# In[ ]:


data['Type 2'].nunique()


# In[ ]:


data['Type 2'].value_counts()


# In[ ]:


data.groupby("Type 1")["Name"].count().sort_values(ascending= False).to_frame()


# In[ ]:


px.histogram(data, "Type 1")


# In[ ]:


px.histogram(data, "HP")


# In[ ]:


px.box(data, x = "Generation", y = "Defense", points = "all")


# In[ ]:


px.box(data, x = "Generation", y = "Attack", points = "all")


# In[ ]:


data.dtypes


# In[ ]:


data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')


# In[ ]:





# In[ ]:


data.isna().sum()


# In[ ]:


data.shape


# In[ ]:


data["Type 2"].fillna('unknown', inplace = True)


# In[ ]:


data.isna().sum()


# In[ ]:


data.dropna(inplace = True)


# In[ ]:


data.shape


# In[ ]:


px.line(data, y= "Defense", x = "Name")


# In[ ]:


data.groupby(["Type 1", "Type 2"]).count().dropna().unstack(-1)


# In[ ]:


data[data["Legendary"] == True]


# In[ ]:


#finding the missing pokemon
data.isna().sum()


# In[ ]:


pokemon = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')


# In[ ]:


pokemon.describe()


# In[ ]:


pokemon.isna().sum()


# In[ ]:


pokemon[pokemon['Name'].isnull()]


# In[ ]:


pokemon.loc[62, "Name"] = 'Primeape'


# In[ ]:


pokemon.iloc[62:67]


# In[ ]:


combats = pd.read_csv('/kaggle/input/pokemon-challenge/combats.csv')


# In[ ]:


combats.head()


# In[ ]:


combats.info()


# In[ ]:


# How often did the first_pokemon win?
combats[combats['First_pokemon'] == combats['Winner']].shape


# In[ ]:


# How often did the second_pokemon win?
combats[combats['Second_pokemon'] == combats['Winner']].shape


# In[ ]:


# How many winners were there?
wcomb = combats.groupby("Winner").count()


# In[ ]:


# How many unique first_poke winners were there?
fcomb = combats.groupby("First_pokemon").count()


# In[ ]:


# How many unique second_poke winners were there?
scomb = combats.groupby("Second_pokemon").count()


# In[ ]:


the_losing_pokemon = np.setdiff1d(fcomb.index.values, wcomb.index.values)-1


# In[ ]:


pokemon.iloc[the_losing_pokemon[0], ]


# In[ ]:


pokemon[pokemon["Name"] == "Pikachu"]


# In[ ]:


# FEATURE ENGINEERING, WIN PERCENTAGE CALCULATION
fcomb


# In[ ]:


wcomb.sort_index()


# In[ ]:


combats['First_pokemon'].nunique()


# In[ ]:


fcomb['Winner']


# In[ ]:


scomb['Winner']


# In[ ]:


wcomb['Total Fights' ] = fcomb['Winner']+ scomb['Winner']


# In[ ]:


wcomb['Win percentage'] = wcomb['First_pokemon']/wcomb['Total Fights']


# In[ ]:


combats.groupby('Winner')['Winner'].count()


# In[ ]:


combats.loc[combats['Winner']== 1]


# In[ ]:


combats.loc[combats['Winner']== 1].shape


# In[ ]:


combats.loc[ (combats['First_pokemon']== 1) | (combats['Second_pokemon']== 1) ]


# In[ ]:


combats.loc[ (combats['First_pokemon']== 1) | (combats['Second_pokemon']== 1) ].shape


# In[ ]:


pokemon.loc[pokemon['#']== 1]


# In[ ]:


fcomb


# In[ ]:


combats_winner = combats.groupby('Winner').count()


# In[ ]:


combats_first = combats.groupby('First_pokemon').count()


# In[ ]:


combats_second = combats.groupby('Second_pokemon').count()
#pokemon.iloc[297]


# In[ ]:


combats['Total_Matches'] = combats_first['Winner']+ combats_second['Winner']


# In[ ]:


#combats.drop(['Win percentage(first)'], axis =1, inplace = True )


# In[ ]:


combats['Win percentage'] = (combats_winner['First_pokemon'] /combats['Total_Matches'] )*100


# In[ ]:


combats[combats['Total_Matches'].isnull()]


# In[ ]:


res = pd.merge(pokemon, combats, right_index = True, left_on = '#')


# In[ ]:


res[res['Win percentage'].isnull()]


# In[ ]:


res.sort_values(by = 'Win percentage', ascending = False).head()


# In[ ]:


px.scatter(res, "Attack", "Win percentage", color = "Legendary", trendline="ols")


# In[ ]:


px.scatter(res, "Defense", "Win percentage", color = "Legendary", trendline="ols")


# In[ ]:


px.scatter(res, "Speed", "Win percentage", color = "Legendary", trendline="ols")


# In[ ]:


winnings = res.groupby('Type 1')['Win percentage'].mean().reset_index().sort_values(by = 'Win percentage' ,ascending = False)


# In[ ]:


px.bar(winnings, x = "Type 1", y = "Win percentage")


# In[ ]:


res.info()


# In[ ]:


data = dict(
    character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
    value=[10, 14, 12, 10, 2, 6, 6, 4, 4])
pd.DataFrame.from_dict(data)


# In[ ]:


res.sample(20).groupby("Type 1")[["Attack", "Defense", "Speed", "Win percentage"]].mean()


# In[ ]:


res.head()


# In[ ]:


# What makes a good pokemon?
good_pokemon = res[res['Win percentage']>85][['Attack', 'Defense', 'Speed']].mean().to_frame()


# In[ ]:


px.pie(good_pokemon, values = 0, names = good_pokemon.index)


# In[ ]:


sliced = res.sample(10)


# In[ ]:


sliced


# In[ ]:


melted = pd.melt(sliced, id_vars = ['Name', 'Win percentage'], value_vars = ['Attack', 'Defense'])


# In[ ]:


melted


# In[ ]:


melted.pivot(index = "Name", columns = 'variable', values = 'value')


# In[ ]:


res.shape


# In[ ]:


pokemon.shape


# In[ ]:


temp = res[res['Type 2'].isnull()]


# In[ ]:


temp


# In[ ]:


res['Type 2'] = np.where(res['Type 2'].isnull(), res['Type 1'], res['Type 2'])


# In[ ]:


res.sample(20)


# In[ ]:


res.isnull().sum()


# In[ ]:


res.fillna(0)


# In[ ]:


res[res['Total_Matches'].isnull()]


# In[ ]:


res['Total_Matches'].fillna(0)


# In[ ]:


res[res['Total_Matches'].isnull()]


# In[ ]:


res['Total_Matches'] = np.where(res['Total_Matches'].isnull(), 0, res['Total_Matches'])


# In[ ]:


res.info()


# In[ ]:


res['Win percentage'] = np.where(res['Win percentage'].isnull(), 0, res['Win percentage'])


# In[ ]:


res[res['Win percentage']== 0]


# In[ ]:


res.info()


# In[ ]:


res1 = res.sample(100)


# In[ ]:


px.sunburst(res, path = ['Type 1', 'Type 2', 'Name'], values = 'HP',color= 'Win percentage', hover_data=['Attack', 'Defense'],
                  color_continuous_scale= 'Inferno',
                  color_continuous_midpoint=np.average(res['Win percentage']))


# In[ ]:


df = px.data.tips()
#fig = px.sunburst(df, path=['day', 'time', 'sex'], values='total_bill')


# In[ ]:


df


# In[ ]:




