#!/usr/bin/env python
# coding: utf-8

# In the following script I will be using PCA on ballot box level data from past Israeli elections.
# In the high-dimensional data, each party is represented by a very long vector of features, as every ballot box in the data is a column (once the data is transposed). We can use PCA and take the first principal component in order to reduce the dimensionality of the data and find the axis along which parties vary the most (in terms of voting patterns).
# Under some assumptions (such as that living proximity is correlated with political proximity), this is the "real" axis of the Israeli politics. It would be interesting to see how much this axis corresponds with the classical right-left divide that characterizes the very polarized politics in Israel.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
import plotly.offline as py
import matplotlib.pyplot as plt
from tqdm import tqdm

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/israeli_elections_results_1996_to_2015.csv', encoding='iso-8859-1')
df.head()


# In[ ]:


parties_color_dict = {'Balad': '#f46d43',
     'Centre Party': '#fddaec',
     'Hadash-Balad': '#d53e4f',
     'Hadash-Taal': '#d53e4f',
     'Hadash': '#d53e4f',
     'Hatnua': '#decbe4',
     'Joint List': '#d53e4f',
     'Kadima': '#fddaec',
     'Kulanu': '#fddaec',
     'Labour Party': '#fbb4ae',
     'Likud Beitenu': '#b3cde3',
     'Likud': '#b3cde3',
     'Madaa-Raam': '#fdae61',
     'Meretz': '#ccebc5',
     'Moledet': '#ffffcc',
     'National Religious Party': '#fed9a6',
     'National Union': '#b3cde3',
     'One Nation': '#fbb4ae',
     'Raam-Taal': '#fdae61',
     'Raam': '#fdae61',
     'Senior Citizens Party': '#ffffcc',
     'Shas': '#999999',
     'Shinui': '#f1e2cc',
     'The Jewish Home': '#fed9a6',
     'Third Way': '#ffffcc',
     'United Torah Judaism': '#4d4d4d',
     'Yachad': '#ffffcc',
     'Otzma LeYisrael': '#ffffcc',
     'Yesh Atid': '#f1e2cc',
     'Yisrael Baaliya': '#8dd3c7',
     'Yisrael Beitenu': '#bebada',
     'Yisrael Beiteinu': '#bebada',
     'Zionist Union': '#fbb4ae'}


# In[ ]:


def get_fig_per_year(year, df):
    temp_df = df[df.year == year]

    parties = temp_df.transpose()[9:]
    parties = parties[parties.sum(axis=1) > 50000]

    parties_dist = pd.DataFrame()
    parties_dist['parties'] = parties.sum(axis=1).index
    parties_dist['votes'] = parties.sum(axis=1).values


    for col in parties.columns:
        try:
            parties[col] = parties[col]/parties[col].sum()
        except ZeroDivisionError:                        
            pass

    parties_dist['var'] = parties.std(axis=1).values
    parties_dist['relative'] = parties_dist['var']/parties.sum(axis=1).values

    for i,party in enumerate(parties.index):
         parties.iloc[i] = (parties.iloc[i] - parties.iloc[i].mean())/(parties.iloc[i].sum())
        
    pca = PCA(n_components=1)
    transformed = pca.fit(parties).transform(parties)
    parties_dist['location'] = [100000*val[0] for val in transformed]
    try:
        parties_dist['location'] = parties_dist['location'] - parties_dist[parties_dist.parties == 'Likud'].location.values[0]
    except IndexError:
        parties_dist['location'] = parties_dist['location'] - parties_dist[parties_dist.parties == 'Likud Beitenu'].location.values[0]
    if parties_dist[parties_dist.parties == 'Shas'].location.values[0] < parties_dist[parties_dist.parties == 'Meretz'].location.values[0]:
        parties_dist['location'] = -parties_dist['location']
    data = []
    all_votes = []
    all_bins = []
    
    for i, t in enumerate(parties_dist.iterrows()):
        temp_gauss = np.random.normal(parties_dist.iloc[i]['location'],
                                         0.001/parties_dist.iloc[i]['relative'],
                                         int(parties_dist.iloc[i]['votes']))
        all_votes = all_votes + temp_gauss.tolist()
        h=np.histogram(temp_gauss, bins=100)
        all_bins.append(h[1][:-1])
        trace = go.Scatter(
                    x=h[1][:-1],
                    y=h[0],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(width=0.5,
                          color=parties_color_dict[parties_dist.iloc[i]['parties']]),
                    name=parties_dist.iloc[i]['parties'])
        data.append(trace)

    return data


# In the following graph, each bell-shaped curve represents a party. The center of the curve represents the location on the main principal component, the height of the curve corresponds with the number of seats the party gain in the relevant elections cycle, and the width is correlated with the heterogeneity of the party voters, very roughly estimated based on the number of seats and distribution of votes along the ballot boxes. You can use the slider in order to toggle between election years.

# In[ ]:


figure = {
    'data': [],
    'layout': {},
}

figure['layout']['hovermode'] = 'closest'


sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 3000, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

data = []
total_len = 0
for i, year in tqdm(enumerate(sorted(df.year.unique()))):
    temp_data = get_fig_per_year(year, df)
    data = data + temp_data
    total_len+= len(temp_data)


visible_array = [False]*total_len
start = 0
for i, year in tqdm(enumerate(sorted(df.year.unique()))):
    temp_data_len = len(get_fig_per_year(year, df))
    end = start + temp_data_len
    slider_step = {'method': 'restyle',
            'label': str(year),
            'args': ['visible',[False]*start + [True]*temp_data_len + [False]*(total_len - start - temp_data_len)]}
    sliders_dict['steps'].append(slider_step)
    start = end


figure['layout']['sliders'] = [sliders_dict]
figure['data'] = data

py.iplot(figure)


# It is very noticeable that the Arab parties, as well as the ultra-orthodox parties are very far away from the rest of the parties. Let's try to zoom in and look at the political center:

# In[ ]:


figure['layout']['yaxis'] = dict(range = [0, 38000],showgrid=False)
figure['layout']['xaxis'] = dict(range = [-500, 500], showgrid=False, zeroline=False,)
py.iplot(figure)


# In[ ]:




