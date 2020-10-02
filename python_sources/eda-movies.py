#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Movies as i see****

# In today's world, undoubtedly everyone is familiar with the word movie. Everyone enjoys watching movie. Like many others, even I am a movie buff and love watching movies with different genres with drama being my favorite genre. In the below notebook, I have generated some meaningful insights and respresent the insights using visualizations. I have a dataset which consist around 45,500 movies and various factors associated with the movie. Using this data, we will try to get some useful numbers and thehidden pattern involved in data.

# In[ ]:


Importing necessary libraries and loading dataset


# In[ ]:


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pandas_profiling import ProfileReport
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
            
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


movies = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')
movies.head()


# Let us get familiar with the data

# In[ ]:


movies.columns


# In[ ]:


movies.info()


# There are a total of 45,466 movies with 24 columns. From the above data we come to know that belongs to collection and homepage NaN values in excess. Let us get into analysis to find our burning questions.

# Data Wrangling******

# In[ ]:


movies = movies.drop(columns = ['overview','homepage','id','imdb_id','status','poster_path','video','original_title','adult'])


# Here we are dropping the columns as they are not the dominant factors. The original_title column is dropped as it means that the title of movie will be in native language and we already have a title column which is in translated language.We will be able to deduce if the movie is a foreign language film by looking at the original_language feature so no tangible information is lost in doing so.

# In[ ]:


movies['revenue'] = movies['revenue'].replace(0,np.nan)


# We see that revenue column has many values as 0. Still we will keep this column as revenue will play one of the important feature during our analysis
# 
# The budget feature has some unclean values that makes Pandas assign it as a generic object. We proceed to convert this into a numeric variable and replace all the non-numeric values with NaN. Finally, as with budget, we will convert all the values of 0 with NaN to indicate the absence of information regarding budget

# In[ ]:


movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
movies['budget'] = movies['budget'].replace(0, np.nan)
movies[movies['budget'].isnull()].shape


# In[ ]:


movies['return'] = movies['revenue']/movies['budget']


# In[ ]:


A return value > 1 would indicate profit whereas a return value < 1 would indicate a loss. We have created a return column in advance which would be used later.


# In[ ]:


movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['year'] = pd.DatetimeIndex(movies['release_date']).year
movies['month'] = movies['release_date'].dt.month_name()


# New year and month column are added to dataset to find the number of movies occuring in a particular year and month

# Exploratory Data Analysis******

# Production Countries The Dataset consists majority of movies in the English language (more than 31000). However, it is common for a particular movie to be shot in various locations. Here we will try to find which countries attract filmmakers the most.

# In[ ]:


movies['production_countries'] = movies['production_countries'].fillna('[]').apply(ast.literal_eval)
change = lambda x: [i['name'] for i in x] if isinstance(x, list) else []
movies['production_countries'] = movies['production_countries'].apply(change) 


# In[ ]:


x = movies.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
x.name = 'countries' #if we dont do this than we get error while joining as column wont have any name


# In[ ]:


country_df = movies.drop('production_countries', axis=1).join(x)
country_df = pd.DataFrame(country_df['countries'].value_counts())
country_df['country'] = country_df.index
country_df.columns = ['num_movies', 'country']
country_df = country_df.reset_index().drop('index', axis=1)


# In[ ]:


country_df.head(10)


# In[ ]:


country_df = country_df[country_df['country'] != 'United States of America']


# In[ ]:


data = dict(type = 'choropleth',
           locations = country_df['country'],
           colorscale = 'Portland',
           locationmode = 'country names',
           z = country_df['num_movies'],
           text = country_df['country'],
           marker = dict(line = dict(color = 'rgb(255,255,255)',width=1)),
           colorbar = {'title' : 'No of movies per country'})


# In[ ]:


layout = dict(title = 'Countries where movies is directed',
             geo = dict(showframe= False,
                       projection = {'type' : 'mercator'}))


# In[ ]:


choromap = go.Figure(data=[data], layout=layout)
iplot(choromap)


# Unsurprisingly, the United States is the most popular destination of production for movies given that our dataset largely consists of English movies. Europe is also an extremely popular location with the UK, France, Germany and Italy in the top 5. Japan and India are the most popular Asian countries when it comes to movie production.

# Franchise movies Let us now have a brief look at Franchise movies. I was curious to discover the longest running and the most successful franchises among many other things.

# In[ ]:


fran_df = movies[movies['belongs_to_collection'].notnull()]
fran_df['belongs_to_collection'] = fran_df['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
fran_df = fran_df[fran_df['belongs_to_collection'].notnull()]


# In[ ]:


pivotfran_df = pd.pivot_table(fran_df,
                              index = 'belongs_to_collection',
                              values='revenue',
                              aggfunc={'revenue':['count','sum']}).reset_index()


# In[ ]:


pivotfran_df.sort_values('sum', ascending=False).head(10)


# The Harry Potter Franchise is the most successful movie franchise raking in more than 7.707 billion dollars from 8 movies. The Star Wars Movies occupies second spot with 7.403 billion dollars from 8 movies too. James Bond is third but the franchise has significantly more movies compared to the others in the list

# Longest Running Franchises Here we will try to find which franchise managed to deliver the largest number of movies under a single banner. However, this does not imply that successful movie franchises tend to have more movies. Some franchises, such as Harry Potter, have a predefined storyline and it wouldn't make sense to produce more movies despite its enormous success.

# In[ ]:


pivotfran_df.sort_values('count', ascending=False).head(10)


# In[ ]:


p = pivotfran_df.sort_values('count', ascending=False)


# In[ ]:


fig = px.scatter(p.head(25), x="count", y="sum",size="sum", color="belongs_to_collection",
                 hover_name="count", log_x=True, size_max=60)
fig.show()


# The James Bond Movies is the largest franchise ever with over 26 movies released under the banner. Friday the 13th and Pokemon come in at a distant second and third with 12 and 11 movies respectively. Here we can conclude that revenue is not influenced by number of movies as Harry potter franchise with only 8 movies managed to gather revenue more than james bond franchise which has 26 movies under his belt.

# Production companies

# In[ ]:


movies['production_companies'] = movies['production_companies'].fillna('[]').apply(ast.literal_eval)
prod_name = lambda x: [i['name'] for i in x] if isinstance(x, list) else []
movies['production_companies'] = movies['production_companies'].apply(prod_name)


# In[ ]:


x = movies.apply(lambda x: pd.Series(x['production_companies']),axis=1).stack().reset_index(level=1, drop=True)
x.name = 'p_companies'


# In[ ]:


company_df = movies.drop('production_companies', axis=1).join(x)


# Total no of movies under particular company

# In[ ]:


MovieCount_df = pd.pivot_table(
    company_df,
    values=["title"],
    index=["p_companies"],
    aggfunc="count"
)


# In[ ]:


movieCount_df = MovieCount_df.rename(columns={"title": "total"})


# In[ ]:


movieCount_df.sort_values('total',ascending=False)


# In[ ]:


company_sum = pd.DataFrame(company_df.groupby('p_companies')['revenue'].sum().sort_values(ascending=False))
company_sum.columns = ['Total']
company_count = pd.DataFrame(company_df.groupby('p_companies')['revenue'].count().sort_values(ascending=False))
company_count.columns = ['Number']
company_mean = pd.DataFrame(company_df.groupby('p_companies')['revenue'].mean().sort_values(ascending=False))
company_mean.columns = ['Average']


# In[ ]:


company_data = pd.concat((company_sum, company_count, company_mean), axis=1)


# In[ ]:


company_data.sort_values('Total', ascending=False).head(10)


# Warner Bros is the highest earning production company with earning a staggering 63.5 billion dollars from 491 movies. Universal Pictures comes second and Paramaount Pictures taking the third position with 55 billion dollars and 48 billion dollars in revenue respectively. Here we can say that more movies lead to more generation of revenue.

# Most Succesful Production Companies Which production companies produce the most succesful movies on average? Let us find out. We will only consider those companies that have made at least 10 movies

# In[ ]:


company_data[company_data['Number'] >= 10].sort_values('Average', ascending=False).head(10)


# Pixar Animation Studio take the top spot with marvel studios and heyday films following. Here we come to a conclusion that highest earning production company would not necessarily be the most successful production company.

# Original Language In this section, let us look at the languages of the movies in our dataset. From the production countries, we have already deduced that the majority of the movies in the dataset are English. Let us see what the other major languages represented are.

# In[ ]:


movies['original_language'].drop_duplicates().shape[0]


# In[ ]:


lang_df = pd.DataFrame(movies['original_language'].value_counts())
lang_df['language'] = lang_df.index
lang_df.columns = ['number', 'language']
lang_df.head()


# There are over 93 languages represented in our dataset. As we had expected, English language films form the overwhelmingly majority. French and Italian movies come at a very distant second and third respectively. Let us represent the most popular languages (apart from English) in the form of a bar plot.

# Popularity, Vote Average and Vote Count We will try to gain a deeper understanding of the popularity, vote average and vote count features and try and deduce any relationships between them as well as other numeric features such as budget and revenue.

# In[ ]:


#popularity, vote avg, vote count
def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan


# In[ ]:


movies['popularity'] = movies['popularity'].apply(clean_numeric).astype('float')
movies['vote_count'] = movies['vote_count'].apply(clean_numeric).astype('float')
movies['vote_average'] = movies['vote_average'].apply(clean_numeric).astype('float')


# In[ ]:


movies['popularity'].describe()


# In[ ]:


movies['popularity'].plot(logy=True, kind='hist')


# The Popularity score seems to be an extremely skewed quentity with a mean of only 2.9 but maximum values reaching greather than 500. The popularity score greater than 500 can be considered as an outlier as there is a huge amount of difference between mean and the highest score

# In[ ]:


movies[['title', 'popularity', 'year','revenue']].sort_values('popularity', ascending=False).head(10)


# Minions is the most popular movie. Wonder Woman and Beauty and the Beast come in second and third respectively. we also come to a conclusion that popularity and revenue are not proportional.

# In[ ]:


movies['vote_count'].describe()


# In[ ]:


movies[['title', 'vote_count', 'year','revenue']].sort_values('vote_count', ascending=False).head(10)


# Inception and The Dark Knight, two critically acclaimed and commercially successful Christopher Nolan movies figure at the top of our chart.

# In[ ]:


movies['vote_average'] = movies['vote_average'].replace(0, np.nan)
movies['vote_average'].describe()


# In[ ]:


movies[movies['vote_count'] > 2000][['title', 'vote_average', 'vote_count' ,'year']].sort_values('vote_average', ascending=False).head(10)


# The Shawshank Redemption and The Godfather are the two most critically acclaimed movies. Interestingly, they are the top 2 movies in IMDB's Top 250 Movies list too.

# Release month and year******

# In[ ]:


Count = movies['month'].value_counts()
mon_df = pd.DataFrame(Count)


# In[ ]:


mon_df['month_name'] = mon_df.index
month_df = mon_df.rename(columns={'month':'count'})
month_df.reset_index().drop('index', axis=1)
month_df['month_name'] = pd.Categorical(month_df['month_name'], ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])


# In[ ]:


month_df = month_df.sort_values('month_name')
month_df


# In[ ]:


color_palette_list = ['#d63031', '#ff7675', '#0984e3', '#74b9ff','#81ecec', '#00cec9','#fab1a0','#e17055','#fdcb6e','#ffeaa7','#00b894','#55efc4']
trace = go.Bar(
            x=month_df['month_name'],
            y=month_df['count'],
            marker=dict(
            color=color_palette_list))
data = [trace]

#extra for layout-withou this also graph plots
layout = go.Layout(
    title='Number of movies released in a particular month',
    font=dict(color='#909090'),
    xaxis=dict(
        title='Months',
        titlefont=dict(
            family='Arial, sans-serif',
            size=12,
            color='#909090'
        ),
        showticklabels=True,
        tickangle=-45,
        tickfont=dict(
            family='Arial, sans-serif',
            size=12,
            color='#909090'
        ),
),
    yaxis=dict(
        title="Number of movies",
        titlefont=dict(
            family='Arial, sans-serif',
            size=12,
            color='#909090'
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            family='Arial, sans-serif',
            size=12,
            color='#909090'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# From the bar chart it is clear that January is the most popular month when it comes to movie releases followed by september and october.

# In which months do bockbuster movies tend to release? To answer this question, we will consider all movies that have made in excess of 100 million dollars and calculate the average gross for each month.

# In[ ]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# In[ ]:


month_mean = pd.DataFrame(movies[movies['revenue'] > 1e8].groupby('month')['revenue'].mean())
month_mean['mon'] = month_mean.index
plt.figure(figsize=(12,6))
plt.title("Average Gross by the Month for Blockbuster Movies")
sns.barplot(x='mon', y='revenue', data=month_mean, order=month_order)


# We see that the months of April, May and June have the highest average gross among high grossing movies. This can be attributed to the fact that blockbuster movies are usually released in the summer vacation and therefore, the audience is more likely to have free time to watch movies.

# In[ ]:


year_p = pd.pivot_table(movies,index='month',values='title',columns='year',aggfunc='count')
year_p = year_p.fillna(0)
year_p.drop(year_p.iloc[:,0:115],axis=1,inplace=True)


# In[ ]:


sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(year_p, annot=True, linewidths=.5, ax=ax, fmt='n', yticklabels=month_order)


# From the heatmap can see that a highest of 247 movies were released in the December 2011. For 2018 and 2020 the almost all values are 0 as the data is only till movies released prior to 2018

# Runtime Now let's try and see what we can find from runtime

# In[ ]:


movies['runtime'].describe()


# The average length of a movie is about 1 hour and 30 minutes. The longest movie on record in this dataset is 1256 minutes long

# In[ ]:


movies['runtime'] = movies['runtime'].astype('float')
plt.figure(figsize=(12,6))
sns.distplot(movies[(movies['runtime'] < 300) & (movies['runtime'] > 0)]['runtime'])


# In[ ]:


movies['genres'] = movies['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
s = movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'


# In[ ]:


genre_df = movies.drop('genres', axis=1).join(s)


# In[ ]:


genre_df['genre'].value_counts().shape[0]


# There are a total of 32 different genres. Now let us check movie on which genre is made highly

# In[ ]:


pop_gen = pd.DataFrame(genre_df['genre'].value_counts()).reset_index()
pop_gen.columns = ['genre', 'movies']
pop_gen.head(10)


# In[ ]:


fig = px.bar(pop_gen.head(10), x='genre', y='movies',title='Most popular genre')
fig.show()


# In[ ]:


fig = px.pie(pop_gen, values='movies', names='genre', title='major Genre used in movies')
fig.show()


# Drama is the most popular genre which occupies around 22.2% of the dataset. Next in the list is comedy follwed by thriller

# Next up, we will try to find the demand of genre starting from 2000. We will wrangle data to check whether there is a surge in particular genre or is the trend almost constant.

# In[ ]:


genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Adventure', 'Science Fiction']


# In[ ]:


pop_gen_movies = genre_df[(genre_df['genre'].isin(genres)) & (genre_df['year'] >= 2000) & (genre_df['year'] <= 2017)]
pop_gen_movies['month'] = pd.Categorical(pop_gen_movies['month'], ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
ctab = pd.crosstab([pop_gen_movies['year']], pop_gen_movies['genre']).apply(lambda x: x/x.sum(), axis=1)


# In[ ]:


ctab[genres].plot(kind='bar', stacked=True, colormap='jet', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stacked Bar Chart of Movie Proportions by Genre")
plt.show()


# In[ ]:


fig = go.Figure(go.Scatter(
        x = ctab.index,
        y = ctab.Action,
        mode='lines+markers',
        name = 'Action'
))

fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Adventure,
        mode='lines+markers',
        name = 'Adventure'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Comedy,
        mode='lines+markers',
        name = 'Comedy'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Crime,
        mode='lines+markers',
        name = 'Crime'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Drama,
        mode='lines+markers',
        name = 'Drama'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Horror,
        mode='lines+markers',
        name = 'Horror'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Romance,
        mode='lines+markers',
        name = 'Romance'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab['Science Fiction'],
        mode='lines+markers',
        name = 'Science Fiction'
))
fig.add_trace(go.Scatter(
        x = ctab.index,
        y = ctab.Thriller,
        mode='lines+markers',
        name = 'Thriller'
))

fig.show()


# Here we figure out that the trend is almost constant with not a huge surge in genre.

# In[ ]:


movies[movies['budget'].notnull()][['title', 'budget', 'revenue', 'return', 'year']].sort_values('budget', ascending=False).head(10)


# Most successful movie

# In[ ]:


succ_movies = movies[(movies['return'].notnull()) & (movies['budget'] > 5e6)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return', ascending=False).head(10)


# In[ ]:


fig = px.pie(succ_movies, values='revenue', names='title', title='Revenue generated by most successful movies in %')
fig.show()


# From the pie chart it is clear that the most successful movie is E.T the Extra-Terrestrial followed by Star Wars which is close to E.T. Jaws occupies third position with 11.7% share. This concludes that most critically aclaimed movies and most successful movies are independent.

# In[ ]:


corr = movies[['budget','popularity','revenue','runtime','vote_average','vote_count']].corr()
fig = go.Figure(
                data=go.Heatmap(
                z=corr,
                x=corr.columns,
                y=corr.columns,
                hoverongaps=True,
                colorscale="blues",
            )
        )


# In[ ]:


fig.show()


# From the above figure it is clear that revenue and vote_count are highly correlated which means may imply that more the revenue, more the people viewing the movie hence more vote_count. Even budget and revenue are correlated (i.e) more the budget of movie, more would be the revenue generated
