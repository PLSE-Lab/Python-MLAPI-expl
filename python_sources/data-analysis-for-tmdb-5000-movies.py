#!/usr/bin/env python
# coding: utf-8

# It's my first time to publish a kernel in Kaggle. I'll try to dig some intresting information in the notebook as more as I can and visualize them.

# ## 1. Data preparation

# In[ ]:


# import necessary modules
import numpy as np
import pandas as pd
import networkx as nx
# import matplotlib as mpl
# import cufflinks as cf
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as pyg
import plotly.tools as pyt
from wordcloud import WordCloud

import multiprocessing as mp
from time import time, strftime
import calendar
import json
import warnings
warnings.filterwarnings(action='ignore')


# In[ ]:


# set display options
np.set_printoptions(suppress=True)
pd.options.display.float_format = lambda x: ('%0.6f' % x)
get_ipython().run_line_magic('matplotlib', 'inline')
pyo.init_notebook_mode(connected=True)
# cf.set_config_file(offline=False, world_readable=True, theme='ggplot')


# In[ ]:


# list data files we got.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# use pandas to load data
credits_file, movies_file = '../input/tmdb_5000_credits.csv', '../input/tmdb_5000_movies.csv'
credits_df = pd.read_csv(credits_file, encoding='utf-8')
movies_df = pd.read_csv(movies_file, parse_dates=['release_date'])


# Let's see what we got.

# In[ ]:


credits_df.info()


# In[ ]:


credits_df.head(3)


# In[ ]:


movies_df.info()


# In[ ]:


movies_df.head(3)


# Some columns could be converted to original Python object.

# In[ ]:


json_cols = 'cast', 'crew'
for c in json_cols:
    credits_df[c] = credits_df[c].map(json.loads)
json_cols = 'genres','keywords','production_companies','production_countries','spoken_languages'
for c in json_cols:
    movies_df[c] = movies_df[c].map(json.loads)


# The two dataframes could be mergered to be one single dataframe.

# In[ ]:


data_all_df = pd.merge(left=credits_df,     right=movies_df.drop('title', axis=1, inplace=False),     how='left', left_on='movie_id', right_on='id').drop('id', axis=1)
data_all_df.info()


# The following codes is to extract structurized data from json columns.

# In[ ]:


def value_update(s, d, idxes):
    for idx in idxes:
        d[idx] = s[idx]

def calculate_json_col(row, src_col, kept_cols):
    df = None
    if row[src_col]:
        df = pd.DataFrame.from_records(row[src_col])
    else:
        df = pd.DataFrame()
    if kept_cols:
        value_update(row, df, kept_cols)
    return df

def collect_json_col(src_df, src_col, kept_cols):
    return pd.concat(                   map(lambda x: calculate_json_col(src_df.loc[x],                                               src_col, kept_cols),                                               src_df.index),                   axis=0, ignore_index=True)
kept_cols = ['title', 'movie_id']


# Here is cast data.

# In[ ]:


# bg = time()
cast_df = collect_json_col(data_all_df, 'cast', kept_cols)
# print('It took %0.2f seconds.' % (time() - bg))
cast_df.info()


# Here is crew data.

# In[ ]:


crew_df =     collect_json_col(data_all_df, 'crew', kept_cols)
crew_df.info()


# Here is genres data.

# In[ ]:


genres_df = collect_json_col(data_all_df, 'genres', kept_cols)
genres_df.info()


# Here is key words data.

# In[ ]:


keywords_df = collect_json_col(data_all_df, 'keywords', kept_cols)
keywords_df.info()


# Here is production data.

# In[ ]:


production_companies_df =     collect_json_col(data_all_df, 'production_companies', kept_cols)
production_companies_df.info()


# Here is spoken languages data.

# In[ ]:


spoken_languages_df =     collect_json_col(data_all_df, 'spoken_languages', kept_cols)
spoken_languages_df.info()


# Here is production countries data.

# In[ ]:


production_countries_df =     collect_json_col(data_all_df, 'production_countries', kept_cols)
production_countries_df.info()


# Now we can remove json columns from the old dataframe.

# In[ ]:


json_cols = ['production_countries', 'spoken_languages', 'production_companies', 'keywords', 'genres', 'crew', 'cast']
data_all_df.drop(json_cols, axis=1, inplace=True)


# # 2. Data Analysis and Visualization

# In cast information datafram, we find 1000 actors or actress have played part of themselves. And we found that Donald Trump did it 5 times. The most common name of character is Paul. The most common job is reporter and doctor.

# In[ ]:


def merge_by_movie(left, right_cols):
    return pd.merge(left=left, right=data_all_df[right_cols + ['movie_id']],                     how='left', left_on='movie_id', right_on='movie_id')


# In[ ]:


cast_df.character.value_counts().head(20)


# In[ ]:


play_self = cast_df[(cast_df.character=='Himself') | (cast_df.character=='Herself')]['name']
print("%d actors or actress have played part of themselves." % (play_self.nunique()))
play_self.value_counts().head()


# In[ ]:


cast_df.name.value_counts().head(20)


# If we suppose that every actor or actress as first three characters shares same weight on the vote of every movie, average vote score of every actor of actresss could be calculated as following. Similarly, we can get vote score of directors and editor.

# In[ ]:


def calculate_vote(df):
    votes = merge_by_movie(left=df[['movie_id', 'name']], right_cols=['vote_average','vote_count'])
    act_movie_cnt = votes.name.value_counts()
    votes = votes.groupby('name').apply(lambda sub:                                         (sub.vote_average * sub.vote_count).sum() / (sub.vote_count.sum()+1))
    votes_df = pd.DataFrame(index=votes.index)
    votes_df.loc[votes.index, 'votes'] = votes.values
    votes_df.loc[act_movie_cnt.index, 'movie_cnt'] = act_movie_cnt.values
    votes_df = votes_df[votes_df.movie_cnt>1]
    votes_df.sort_values(ascending=False, inplace=True, by=['votes', 'movie_cnt'])
    return votes_df


# In[ ]:


main_characters = cast_df[cast_df.order<2]
act_vote_df = calculate_vote(main_characters)
act_vote_df.head(15)


# In[ ]:


act_vote_df.describe()


# In[ ]:


directors_df = crew_df[crew_df.job=='Director']
directors_vote_df = calculate_vote(directors_df)
directors_vote_df.head(15)


# In[ ]:


editors_df = crew_df[crew_df.job=='Editor']
editors_vote_df = calculate_vote(editors_df)
editors_vote_df.head(15)


# Let's take a look at the movie genres distribution. Drama, comedy, thriller and action movies take more than a half, which are the top 4 movie genres.

# In[ ]:


genres_catl = genres_df['name'].value_counts()
genres_catl = genres_catl / genres_catl.sum()
others = 0.01
genres_catl_ = genres_catl[genres_catl>=others]
genres_catl_['Other'] = genres_catl[genres_catl<others].sum()
explode = (genres_catl_ <= 0.02) / 20 + 0.05
genres_catl_.plot(kind='pie', label='', startangle=10, shadow=False,                  figsize=(7, 7), autopct="%1.1f%%", explode=explode)


# Here is production countries and spoken language distribution.

# In[ ]:


ct_catl = production_countries_df['name'].value_counts()
ct_catl = ct_catl / ct_catl.sum()
others = 0.04
ct_catl_ = ct_catl[ct_catl>=others]
ct_catl_['Other'] = ct_catl[ct_catl<others].sum()
# explode = (ct_catl_ <= 0.02) / 100
explode = np.zeros(len(ct_catl_)) + 0.05
ct_catl_.plot(kind='pie', explode=explode, autopct="%1.1f%%",         figsize=(7, 7), label='', startangle=120, shadow=False)


# In[ ]:


sp_catl = spoken_languages_df['name'].value_counts()
sp_catl = sp_catl / sp_catl.sum()
others = 0.02
sp_catl_ = sp_catl[sp_catl>=others]
sp_catl_['Other'] = sp_catl[sp_catl<others].sum()
# explode = (sp_catl_ <= 0.02) / 100
explode = np.zeros(len(sp_catl_)) + 0.05
sp_catl_.plot(kind='pie', explode=explode, autopct="%1.1f%%",         figsize=(7, 7), label='', startangle=120, shadow=False)


# The following code will analysis these dataset by release date. The release years of these movies are from 1916 to 2017 which covers about 100 years.

# In[ ]:


# get release year and month
data_all_df['release_year'] = data_all_df['release_date'].dt.year
data_all_df['release_month'] = data_all_df['release_date'].dt.month.fillna(0)
# data_all_df['release_month'] = data_all_df['release_date'].dt.month.fillna(0).astype(np.int8)\
#     .map(lambda m: np.nan if m==0 else calendar.month_name[m][:3])
data_all_df['release_year'].describe()


# In[ ]:


years_genres_change = merge_by_movie(genres_df, ['release_year']).groupby(['release_year', 'name']).apply(len)
years_genres_idx = pd.MultiIndex.from_product([sorted(data_all_df.release_year.unique()),                                               genres_df.name.unique()], 
                                             names=['release_year', 'genre'])
years_genres_change_df = pd.DataFrame(index=years_genres_idx, columns=['amount'], data=0)
years_genres_change_df.loc[years_genres_change.index, 'amount'] = years_genres_change.values
# years_genres_change = pd.DataFrame(data=years_genres_change.values, index=years_genres_change.index, columns=['number'])
years_genres_change_df.reset_index(level=['release_year'], inplace=True)
# years_genres_change_df.head()
traces = []
# marker = {'symbol': 'star', 'size': 5}
for g in np.unique(years_genres_change_df.index):
    trc = pyg.Bar(x=years_genres_change_df.loc[g,'release_year'],                       y=years_genres_change_df.loc[g,'amount'], text=g, name=g)
    traces.append(trc)
layout_comp = pyg.Layout(title='Movies amount per genres distribution by year', hovermode='closest', barmode='stack',                    xaxis={'title': 'year', 'gridwidth': 1},                    yaxis={'title': 'amount', 'gridwidth': 2})
fig_comp = pyg.Figure(data=pyg.Data(traces), layout=layout_comp)
pyo.iplot(fig_comp)


# The following code is to show key words of these movies as a word cloud.

# In[ ]:


wc = WordCloud(background_color='white', max_words=2000, random_state=1).     generate_from_frequencies(keywords_df['name'].value_counts().to_dict())
plt.figure(figsize=(16, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In the following code, I'll try to shows some movies (release after 2000) with a part of actors or actresses in one network graph.

# In[ ]:


main_casts = merge_by_movie(main_characters,                             right_cols=['release_year', 'release_month', 'vote_average'])
main_casts['release_year'] =     main_casts.apply(lambda r: r.release_year + r.release_month/100, axis=1)
main_casts = pd.merge(left=main_casts[['name', 'title', 'release_year', 'vote_average']],                       right=act_vote_df, how='left', left_on='name', right_index=True)
main_casts['title'] = main_casts.apply(lambda r: ("%s(%.0f)" % (r['title'], r['release_year'])), axis=1)
active_year = main_casts.groupby('name')['release_year'].mean()
main_casts['active_year'] = main_casts['name'].map(lambda n: active_year[n])
main_casts = main_casts[(main_casts.release_year>2000) & (main_casts.movie_cnt>=20) & (main_casts.vote_average>0)]
main_casts = main_casts.sample(n=80, random_state=5)
main_casts_g = nx.from_pandas_dataframe(main_casts, 'title', 'name', create_using=nx.DiGraph(),                                        edge_attr=['release_year','vote_average','votes','active_year','movie_cnt'])
edge_trace = pyg.Scatter(
    x=[],
    y=[],
    line=dict(width=0.7,color='#807'),
    hoverinfo='none',
    mode='lines')

node_trace = pyg.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        opacity=0.6,
        symbol=[],
        color=[],
        size=[]))
for ed in main_casts_g.edges():
    ed_dt = main_casts_g[ed[0]][ed[1]]
    xm = ed_dt['release_year']
    ym = ed_dt['vote_average']
    xc = ed_dt['active_year']
    yc = ed_dt['votes']
    edge_trace['x'].extend([xm, xc, None])
    edge_trace['y'].extend([ym, yc, None])
    node_trace['x'].extend([xm, xc])
    node_trace['y'].extend([ym, yc])
    node_trace['text'].extend([ed[0], ed[1]+': Played %d movies' % ed_dt['movie_cnt']])
    node_trace['marker']['color'].extend(['blue', 'red'])
    node_trace['marker']['symbol'].extend(['point', 'star'])
    node_trace['marker']['size'].extend([12, 12])

layout = dict(title='Network graph of actors/actress and movies',               hovermode='closest', showlegend=False,              xaxis=dict(title='active/release year'),              yaxis=dict(title='vote'))
fig = pyg.Figure(data=[edge_trace, node_trace], layout=layout)
pyo.iplot(fig)


# In[ ]:




