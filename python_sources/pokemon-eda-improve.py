#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pokemon=pd.read_csv('..//input/Pokemon.csv')


# In[ ]:




pokemon.head(2)


# In[ ]:


pokemon.drop(columns='#',inplace=True)


# In[ ]:


pokemon.head()


# In[ ]:


pokemon_sorted=pokemon.sort_values(by=['Total','HP'],ascending=False)


# In[ ]:


def ranker(df):
    df['Rank']=np.arange(len(df))+1
    return df


# In[ ]:


pokemon_ranked=pokemon_sorted.groupby('Type 1').apply(ranker)


# In[ ]:


pokemon_ranked.head(5)


# In[ ]:


types=pokemon_ranked['Type 1'].unique()


# In[ ]:


fig = go.Figure()
fig


# In[ ]:


def finalplot(df,color):
    trace=go.Scatter(
        x=np.arange(1,5),
        y=df.Total,
        text=df.Name,
        mode='lines+markers'     
    )
    return trace


# In[ ]:


for item in types:
    pokemonBytype=pokemon_ranked[pokemon_ranked['Type 1']== item ].iloc[0:4,:]
    fig.add_trace(finalplot(pokemonBytype,'red'))


# In[ ]:


fig.show()


# In[ ]:




