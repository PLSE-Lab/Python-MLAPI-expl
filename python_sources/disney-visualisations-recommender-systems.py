#!/usr/bin/env python
# coding: utf-8

# <h1 style='text-align:center; font-size:50px; font-weight:bold;'>Disney+ Visualisations & Recommender Systems</h1>

# <h1 style='text-align:center; font-size:30px; font-weight:none;'>Table of Contents</h1>

# <h2 style='text-align:center; font-size:20px; font-weight:none;'>1. Exploratory Data Analysis</h2>
# <h2 style='text-align:center; font-size:20px; font-weight:none;'>2. Content-Based Recommender Systems</h2>

# In[ ]:


import numpy as np 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import wordcloud

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/disney-plus-shows/disney_plus_shows.csv')


# In[ ]:


from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# <h1 style='text-align:center; font-size:50px; font-weight:bold;'>Exploratory Data Analysis</h1>

# <h1 style='text-align:center; font-size:30px'>IMBD Ratings</h1>

# In[ ]:


fig = px.histogram(df['imdb_rating'],nbins=40,labels={'value':'imbd_rating'})
fig.update_layout(title='Distribution of IMBD Ratings',title_x=0.5)


# In[ ]:


fig = px.box(df,y='imdb_rating')
fig.update_layout(title='Box Plot of IMBD Ratings',title_x=0.5)


# In[ ]:


df_top10 = df.sort_values('imdb_rating',ascending=False).head(10)
fig = go.Figure(data=[go.Bar(
            x=df_top10['title'], y=df_top10['imdb_rating'],
            text=df_top10['imdb_rating'],
            textposition='auto',
        )])
fig.update_layout(title='Top 10 shows/movies/episodes with the Highest IMDB Ratings',title_x=0.5)


# In[ ]:


top10_m = df[df['type'] == 'movie'].sort_values('imdb_rating',ascending=False).head(10)

fig = px.bar(top10_m,top10_m['title'],top10_m['imdb_rating'],text=top10_m['imdb_rating'])
fig.update_layout(title='Top 10 Movies with highest IMDB Rating',title_x=0.5)
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Scatter(
    x=top10_m['title'],
    y=top10_m['imdb_rating'],
    mode='markers',
    marker=dict(
        color=10+np.random.randn(200),

        size=top10_m['imdb_rating']*5,
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Movies with the Highest IMDB Ratings',
    title_x=0.5,
    xaxis_title="name",
    yaxis_title="IMDB Rating",
        template='plotly_white'

)
fig.show()


# In[ ]:


top10_m = df[df['type'] == 'series'].sort_values('imdb_rating',ascending=False).head(10)

fig = px.bar(top10_m,top10_m['title'],top10_m['imdb_rating'],text=top10_m['imdb_rating'])
fig.update_layout(title='Top 10 Series with highest IMDB Rating',title_x=0.5)
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Scatter(
    x=top10_m['title'],
    y=top10_m['imdb_rating'],
    mode='markers',
    marker=dict(
        color=10+np.random.randn(200),

        size=top10_m['imdb_rating']*5,
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Series with the Highest IMDB Ratings',
    title_x=0.5,
    xaxis_title="name",
    yaxis_title="IMDB Rating",
        template='plotly_white'

)
fig.show()


# In[ ]:


top10_m = df[df['type'] == 'episode'].sort_values('imdb_rating',ascending=False).head(10)

fig = px.bar(top10_m,top10_m['title'],top10_m['imdb_rating'],text=top10_m['imdb_rating'])
fig.update_layout(title='Top 10 Episodes with highest IMDB Rating',title_x=0.5)
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Scatter(
    x=top10_m['title'],
    y=top10_m['imdb_rating'],
    mode='markers',
    marker=dict(
        color=10+np.random.randn(200),

        size=top10_m['imdb_rating']*5,
        showscale=True
        )
)])
fig.update_layout(
    title='Top 10 Episodes with the Highest IMDB Ratings',
    title_x=0.5,
    xaxis_title="name",
    yaxis_title="IMDB Rating",
        template='plotly_white'

)
fig.show()


# <h1 style='text-align:center; font-size:30px'>IMBD Votes</h1>

# In[ ]:


fig = px.histogram(df['imdb_votes'])
fig.update_layout(title='Distribution of IMBD Votes',title_x=0.5)


# In[ ]:


df_top10_v = df.sort_values('imdb_votes',ascending=False).head(10)
df_top10_v['imdb_votes'] = df_top10_v['imdb_votes'].apply(lambda x: x.replace(',','.')).astype(float)

fig = go.Figure(data=[go.Bar(
            x=df_top10_v['title'], y=df_top10_v['imdb_votes'],
            text=df_top10_v['imdb_votes'],
            textposition='auto',
        )])

fig.update_layout(title='Top 10 shows/movies/episodes with the most IMDB Votes',title_x=0.5)
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=df_top10_v['genre'], y=df_top10_v['imdb_votes'],
            text=round(df_top10_v['imdb_votes'],0),
            textposition='auto',
        )])

fig.update_layout(title='Top 10 Genres with the most IMDB Votes',title_x=0.5)
fig.show()


# In[ ]:


top10_m = df[df['type'] == 'movie'].sort_values('imdb_votes',ascending=False).head(10)
top10_m['imdb_votes'] = top10_m['imdb_votes'].apply(lambda x: x.replace(',','.')).astype(float)


fig = px.bar(top10_m,top10_m['title'],top10_m['imdb_votes'],text=top10_m['imdb_votes'])
fig.update_layout(title='Top 10 Movies with highest IMDB Votes',title_x=0.5)
fig.show()


# In[ ]:


top10_m = df[df['type'] == 'series'].sort_values('imdb_votes',ascending=False).head(10)
top10_m['imdb_votes'] = top10_m['imdb_votes'].apply(lambda x: x.replace(',','.')).astype(float)

fig = px.bar(top10_m,top10_m['title'],top10_m['imdb_rating'],text=top10_m['imdb_rating'])
fig.update_layout(title='Top 10 Series with highest IMDB Votes',title_x=0.5)
fig.show()


# In[ ]:


top10_m = df[df['type'] == 'episode'].sort_values('imdb_votes',ascending=False).head(10)
top10_m['imdb_votes'] = top10_m['imdb_votes'].apply(lambda x: x.replace(',','.')).astype(float)

fig = px.bar(top10_m,top10_m['title'],top10_m['imdb_rating'],text=top10_m['imdb_rating'])
fig.update_layout(title='Top 10 Episodes with highest IMDB Votes',title_x=0.5)
fig.show()


# <h1 style='text-align:center; font-size:30px'>Title,Plot,Actors & Directors</h1>

# In[ ]:


stopwords = ['dtype','Name','object','Length'] + list(STOPWORDS)


genre = wordcloud.WordCloud(max_words=6000,stopwords=stopwords,background_color='white')

genre.generate(str(df['title']))

plt.figure(figsize=(15,15))
plt.imshow(genre,interpolation='bilinear')
plt.axis('off')
plt.show()


# <h1 style='text-align:center; font-size:25px'>WordCloud of the most frequent words in the Titles</h1>

# In[ ]:


stopwords = ['plot','an','In'] + list(STOPWORDS)

genre = wordcloud.WordCloud(max_words=6000,stopwords=stopwords,background_color='white')

genre.generate(str(df['plot']))

plt.figure(figsize=(15,15))
plt.imshow(genre,interpolation='bilinear')
plt.axis('off')
plt.show()


# <h1 style='text-align:center; font-size:25px'>WordCloud of the most frequent actor names</h1>

# In[ ]:


stopwords = set(STOPWORDS)

genre = wordcloud.WordCloud(stopwords=stopwords,background_color='white')

genre.generate(str(df['actors']))

plt.figure(figsize=(15,15))
plt.imshow(genre,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


df_writer = df.assign(var1 = df.writer.str.split(',')).explode('var1').reset_index(drop = True)

df_writer['splitted'] = df_writer.var1.str.lstrip()


# In[ ]:


writers = pd.DataFrame(df_writer['splitted'].value_counts()).reset_index().head(10)

fig = px.bar(writers,writers['index'],writers['splitted'],labels={'index':'name','splitted':'count'})
fig.update_layout(title='Top 10 Writers',title_x=0.5)


# <h1 style='text-align:center; font-size:25px'>WordCloud of the most frequent Director names</h1>

# In[ ]:


stopwords = ['object','director'] + list(STOPWORDS)

genre = wordcloud.WordCloud(max_words=6000,stopwords=stopwords,background_color='white')

genre.generate(str(df['director']))

plt.figure(figsize=(15,15))
plt.imshow(genre,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


df_director = df.assign(var1 = df.director.str.split(',')).explode('var1').reset_index(drop = True)

df_director['splitted'] = df_director.var1.str.lstrip()

directors = pd.DataFrame(df_director['splitted'].value_counts()).reset_index().head(10)

fig = px.bar(directors,directors['index'],directors['splitted'],labels={'index':'name','splitted':'count'})
fig.update_layout(title='Top 10 Directors',title_x=0.5)


# <h1 style='text-align:center; font-size:30px'>Genres</h1>

# <h1 style='text-align:center; font-size:25px'>WordCloud of the Genres</h1>

# In[ ]:


stop_words = ['dtype','Name','object','Length'] + list(STOPWORDS)

genre = wordcloud.WordCloud(max_words=6000,stopwords=stop_words,background_color='white')

genre.generate(str(df['genre']))

plt.figure(figsize=(15,15))
plt.imshow(genre,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


df_genres = df.groupby('genre')[['imdb_rating']].sum().reset_index().sort_values('imdb_rating',ascending=False)


# In[ ]:


df_genre = df.assign(var1 = df.genre.str.split(',')).explode('var1').reset_index(drop = True)

df_genre['splitted'] = df_genre.var1.str.lstrip()

genres = pd.DataFrame(df_genre['splitted'].value_counts()).reset_index().head(10)

fig = px.bar(genres,genres['index'],directors['splitted'],labels={'index':'name','splitted':'count'})
fig.update_layout(title='Top 10 Genres',title_x=0.5)


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=df_genres['genre'], y=df_genres['imdb_rating'],
            text=df_genres['imdb_rating'],
            textposition='auto',
        )])

fig.update_layout(title='Sum of IMBD Ratings per Genre',title_x=0.5)
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=df_genres['genre'].head(10), y=df_genres['imdb_rating'].head(10),
            text=round(df_genres['imdb_rating'].head(10),2),
            textposition='auto',
        )])

fig.update_layout(title='Top 10 Genres with the highest sum of IMDB ratings',title_x=0.5)
fig.show()


# In[ ]:


df_genres_m = df.groupby('genre')[['metascore']].sum().reset_index().sort_values('metascore',ascending=False)


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=df_genres_m['genre'], y=df_genres_m['metascore'],
            text=df_genres_m['metascore'],
            textposition='auto',
        )])

fig.update_layout(title='Sum of Metascore Ratings per Genre',title_x=0.5)
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=df_genres_m['genre'].head(10), y=df_genres_m['metascore'].head(10),
            text=round(df_genres_m['metascore'].head(10),2),
            textposition='auto',
        )])

fig.update_layout(title='Top 10 Genres with the highest sum of Metascore ratings',title_x=0.5)
fig.show()


# <h1 style='text-align:center; font-size:30px'>Year</h1>

# In[ ]:


fig = px.histogram(df['year'],labels={'value':'year'})
fig.update_layout(title='Distribution of show years',title_x=0.5)


# In[ ]:


df_years = df.groupby('year')[['imdb_rating']].sum().reset_index().sort_values('imdb_rating',ascending=False)


# In[ ]:


fig = px.bar(df_years,x=df_years['year'],y=df_years['imdb_rating'])
fig.update_layout(title='Sum of Imdb ratings per year',title_x=0.5)


# In[ ]:


fig = px.pie(df_years.head(10),names=df_years['year'].head(10),values=df_years['imdb_rating'].head(10),labels=df_years['year'],hole=0.5)
fig.update_traces(textposition='inside',name='year+label')

fig.update_layout(title='Top 10 years with the highest Cumulative IMDB ratings',title_x=0.5)


# In[ ]:


df_years_m = df.groupby('year')[['metascore']].sum().reset_index().sort_values('metascore',ascending=False)


# In[ ]:


fig = px.bar(df_years_m,x=df_years['year'],y=df_years_m['metascore'])
fig.update_layout(title='Sum of Metascore ratings per year',title_x=0.5)


# In[ ]:


fig = px.pie(df_years_m.head(10),names=df_years_m['year'].head(10),values=df_years_m['metascore'].head(10),labels=df_years_m['year'],hole=0.5)
fig.update_traces(textposition='inside',name='year+label')

fig.update_layout(title='Top 10 years with the highest Cumulative Metascore ratings',title_x=0.5)


# <h1 style='text-align:center; font-size:30px'>Runtime</h1>

# In[ ]:


fig = px.box(df,y='runtime')
fig.update_layout(title='Box Plot of Runtimes',title_x=0.5)


# In[ ]:


fig = px.histogram(df['runtime'])
fig.update_layout(title='Distribution of movie/show runtimes',title_x=0.5)


# In[ ]:


df_runtime_top10 = df.sort_values('runtime',ascending=False).head(10)
df_runtime_top10['runtime'] = df_runtime_top10['runtime'].apply(lambda x:x.replace('min','')).astype(int)

fig = px.bar(df_runtime_top10,x=df_runtime_top10['title'],y=df_runtime_top10['runtime'],text=df_runtime_top10['runtime'])
fig.update_layout(title='Top 10 movies/shows with the most runtimes',title_x=0.5)
fig.show()


# In[ ]:


df_runtime = df.groupby('runtime')[['imdb_rating']].sum().reset_index().sort_values('imdb_rating',ascending=False)


# In[ ]:


fig = px.bar(df_runtime,x=df_runtime['runtime'],y=df_runtime['imdb_rating'],text=df_runtime['imdb_rating'])
fig.update_layout(title='Cumulative IMDB Ratings grouped by runtimes',title_x=0.5)

fig.show()


# In[ ]:


fig = px.bar(df_runtime.head(10),x=df_runtime['runtime'].head(10),y=df_runtime['imdb_rating'].head(10),text=round(df_runtime['imdb_rating'],2).head(10))
fig.update_layout(title='Top 10 Runtimes with the highest sum of IMDB Ratings',title_x=0.5)

fig.show()


# In[ ]:


df_runtime_m = df.groupby('runtime')[['metascore']].sum().reset_index().sort_values('metascore',ascending=False)


# In[ ]:


fig = px.bar(df_runtime_m,x=df_runtime_m['runtime'],y=df_runtime_m['metascore'],text=df_runtime_m['metascore'])
fig.update_layout(title='Cumulative Metascore Ratings grouped by runtimes',title_x=0.5)

fig.show()


# In[ ]:


fig = px.bar(df_runtime_m.head(10),x=df_runtime_m['runtime'].head(10),y=df_runtime_m['metascore'].head(10),text=round(df_runtime_m['metascore'],2).head(10))
fig.update_layout(title='Top 10 Runtimes with the highest sum of Metascore Ratings',title_x=0.5)

fig.show()


# <h1 style='text-align:center; font-size:30px'>Metascore</h1>

# In[ ]:


fig = px.histogram(df['metascore'])
fig.update_layout(title='Metascore Distribution',title_x=0.5)


# In[ ]:


fig = px.box(df,y='metascore')
fig.update_layout(title='Box Plot of Metascore Ratings',title_x=0.5)


# In[ ]:


df_top10 = df.sort_values('metascore',ascending=False).head(10)
fig = go.Figure(data=[go.Bar(
            x=df_top10['title'], y=df_top10['metascore'],
            text=df_top10['imdb_rating'],
            textposition='auto',
        )])
fig.update_layout(title='Top 10 movies/shows/espisodes with the Highest Metascore Ratings',title_x=0.5)


# In[ ]:


fig = px.pie(df['metascore'],values=df['metascore'],names=df['genre'])
fig.update_traces(textposition='inside')
fig.update_layout(title='Number of Metascore Ratings by Genre',title_x=0.5)
fig.show()


# <h1 style='text-align:center; font-size:30px'>Languages</h1>

# In[ ]:


df_lang = df.assign(var1 = df.language.str.split(',')).explode('var1').reset_index(drop = True)

df_lang['splitted'] = df_lang.var1.str.lstrip()

lang = pd.DataFrame(df_lang['splitted'].value_counts()).reset_index().head(10)
lang.drop(df.index[5], inplace=True)

fig = px.bar(lang,lang['index'],lang['splitted'])
fig.update_layout(title='Top 10 Languages',title_x=0.5)


# <h1 style='text-align:center; font-size:30px'>Type</h1>

# In[ ]:


fig = px.histogram(df['type'])
fig.update_layout(title='Distribution of show types',title_x=0.5)


# In[ ]:


df_type = df.groupby('type')[['imdb_rating']].sum().reset_index().sort_values('imdb_rating',ascending=False)
fig = px.histogram(df_type,df_type['type'],df_type['imdb_rating'])
fig.update_layout(title='Sum of IMDB Ratings per show types',title_x=0.5)


# In[ ]:


fig = px.pie(df_type,names=df_type['type'],values=df_type['imdb_rating'],labels=df_type['type'],hole=0.5)
fig.update_layout(title='Sum of IMDB Ratings per show types',title_x=0.5)


# <h1 style='text-align:center; font-size:30px'>Age Ratings</h1>

# In[ ]:


fig = px.histogram(df['rated'])
fig.update_layout(title='Distribution of age ratings',title_x=0.5)


# In[ ]:


df_r = df.groupby('rated')[['imdb_rating']].sum().reset_index().sort_values('imdb_rating',ascending=False)


# In[ ]:


fig = px.histogram(df_type,df_r['rated'],df_r['imdb_rating'])
fig.update_layout(title='Sum of IMDB Ratings per Age ratings',title_x=0.5)


# In[ ]:


fig = px.pie(df_r,names=df_r['rated'],values=df_r['imdb_rating'],labels=df_r['rated'],hole=0.5)
fig.update_traces(textposition='inside')
fig.update_layout(title='Sum of IMDB Ratings grouped by Age Ratings',title_x=0.5)


# In[ ]:


fig = px.treemap(df_r, path=['rated'], values=df_r['imdb_rating'], height=700,
                 title='Sum of IMDB Ratings grouped by Age Ratings', color_discrete_sequence = px.colors.qualitative.Dark2)
fig.data[0].textinfo = 'label+text+value'
fig.update_layout(title_x=0.5)
fig.show()


# <h1 style='text-align:center; font-size:30px'>Date added at</h1>

# In[ ]:


fig = px.histogram(df['added_at'])
fig.update_layout(title='Distribution of the added dates of the shows',title_x=0.5)


# In[ ]:


df_aa = df.groupby('added_at')[['imdb_rating']].sum().reset_index().sort_values('imdb_rating',ascending=False)


# In[ ]:


fig = px.bar(df_aa,df_aa['added_at'],df_aa['imdb_rating'])
fig.update_layout(title='Added_at dates with the highest sum of IMDB ratings',title_x=0.5)


# In[ ]:


fig = px.bar(df_aa.head(10),df_aa['added_at'].head(10),df_aa['imdb_rating'].head(10))
fig.update_layout(title='Top 10 added_at dates with the highest sum of IMDB ratings',title_x=0.5)


# <h1 style='text-align:center; font-size:50px; font-weight:bold;'>Recommender Systems</h1>

# <h1 style='text-align:center; font-size:30px'>1. Content based Recommender System based on movie plots</h1>

# In[ ]:


import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[ ]:


df['plot'] = df['plot'].fillna('')


# In[ ]:


tfidf = TfidfVectorizer()
plot_matrix = tfidf.fit_transform(df['plot'])


# In[ ]:


similarity_matrix = linear_kernel(plot_matrix,plot_matrix)


# In[ ]:


mapping = pd.Series(df.index,index=df.title)


# In[ ]:


def plot_content_recommender(m_name):
    m_index = mapping[m_name]
    similarity_score = list(enumerate(similarity_matrix[m_index]))
    similarity_score = sorted(similarity_score,key=lambda x:x[1],reverse=True)
    similarity_score = similarity_score[1:10]
    indices = [i[0] for i in similarity_score]
    return df.title.iloc[indices]


# In[ ]:


plot_content_recommender('Coco')


# In[ ]:


plot_content_recommender('Finding Nemo')


# In[ ]:


plot_content_recommender('Cars 3')


# <h1 style='text-align:center; font-size:30px'>2. Content based Recommender System based on movie plot,actors,directors and genre</h1>

# In[ ]:


data = df[['title','plot','director','actors','imdb_rating','genre','plot']]
data['actors'] = data['actors'].str.split(',').fillna('')
data['genre_splitted'] = data['genre'].str.split(',').fillna('')
data['director'] = data['director'].fillna('')
data['plot'] = data['plot'].fillna('')


# In[ ]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[ ]:


features = ['actors', 'genre', 'director']

for feature in features:
    data[feature] = data[feature].apply(clean_data)


# In[ ]:


data


# In[ ]:


def create_soup(x):
    return ' '.join(x['actors']) + ' ' + x['director'] + ' ' + ' '.join(x['genre'] + ' '.join(x['plot']))


# In[ ]:


data['soup'] = data.apply(create_soup, axis=1)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


count = CountVectorizer()
count_matrix = count.fit_transform(data['soup'])


# In[ ]:


sim_matrix = cosine_similarity(count_matrix,count_matrix)


# In[ ]:


data = data.reset_index()
mapping = pd.Series(data.index, index=data['title'])


# In[ ]:


def extended_recommender(m_name):
    m_index = mapping[m_name]
    similarity_score = list(enumerate(sim_matrix[m_index]))
    similarity_score = sorted(similarity_score,key=lambda x:x[1],reverse=True)
    similarity_score = similarity_score[1:10]
    indices = [i[0] for i in similarity_score]
    return data.title.iloc[indices]


# In[ ]:


extended_recommender('Coco')


# In[ ]:


extended_recommender('Moana')


# In[ ]:


extended_recommender('Toy Story')


# In[ ]:




