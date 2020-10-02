#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n                                       \ndiv.h2 {\n    background-color: #3E5AE6;\n    background-image: linear-gradient(120deg, #3E5AE6, #A37CE6);\n    text-align: left;\n    color: white;              \n    padding:9px;\n    padding-right: 100px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 40px; \n}                                  \n                                      \nbody {\n  font-size: 12px;\n}    \n                                                                               \ndiv.h3 {\n    color: #3E5AE6; \n    font-size: 18px; \n    margin-top: 20px; \n    margin-bottom:4px;\n}\n                                     \ndiv.h4 {\n    color: #159957;\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\n                                         \nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n}\n                                        \nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\n  \n                                      \nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}   \n                                         \ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\n                                       \ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n    text-align: center;\n} \n                                         \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n}\n                                          \ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \n                                      \n                                      \ntable.rules tr.best\n{\n    color: green;\n}    \n                                       \n.output { \n    align-items: center; \n}\n                                      \n.output_png {\n    display: table-cell;\n    text-align: center;\n    margin:auto;\n}                                          \n                                                                                                                                                                                                                                      \n</style>  ')


# <center><iframe src='https://gfycat.com/ifr/AngelicAgonizingEidolonhelvum' frameborder='0' scrolling='no' allowfullscreen width='640' height='412'></iframe></center>

# <br>
# <a id='bkground'></a>
# <div class="h2"><center>Introduction</center></div>
# <br>
# 
# Recommendation systems having been much present right now, there are so many approaches to make and improve they.<br>
# We can find they in online shops, music streaming and video streaming services.<br>
# In this Kernel, I make an EDA on [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) and got the data used for movie ratings to build a Recommendation System with Alternating Least Square (ALS), with it, it will be possible make a user based system or a item based system.<br>
# <br>
# For this job, I will work with Pandas Dataframes, visualizations with Altair, printable text with BeautifulText and PySpark for ALS.<br>
# This Kernel is inspired by the great work made by these guys:
# * @tombresee for made this awesome [kernel](https://www.kaggle.com/tombresee/next-gen-eda) with great use of HTML.
# * @artgor with his great kernels using Altair for Data Visualization, [here's an example](https://www.kaggle.com/artgor/dota-eda-fe-and-models)
# * @vchulski for made this [kernel](https://www.kaggle.com/vchulski/tutorial-collaborative-filtering-with-pyspark) that explain so good how to use ALS with PySpark.
# 
# Alright, let's start installing all nedeed modules and imports, you can in the buttons to see the code or output.

# In[ ]:


get_ipython().system('pip install -U vega_datasets notebook vega')


# In[ ]:


get_ipython().system('pip install ujson')


# In[ ]:


get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')
get_ipython().system('pip install pyspark')


# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 8)
import os
import gc
import ujson as json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns

import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs
from plotly.offline import init_notebook_mode
from plotly.offline import plot,iplot

init_notebook_mode(connected=True)

import altair as alt
from altair.vega import v5
from IPython.display import HTML
alt.renderers.enable('notebook')

from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
th_props = [('font-size', '13px'), ('background-color', 'white'), ('color', '#666666')]
td_props = [('font-size', '15px'), ('background-color', 'white')]
styles = [dict(selector="td", props=td_props), dict(selector="th", props=th_props)]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

DATA_PATH = '../input/the-movies-dataset/'


# In[ ]:


# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey
def prepare_altair():
    """
    Helper function to prepare altair for working.
    """

    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION
    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
    noext = "?noext"
    
    paths = {
        'vega': vega_url + noext,
        'vega-lib': vega_lib_url + noext,
        'vega-lite': vega_lite_url + noext,
        'vega-embed': vega_embed_url + noext
    }
    
    workaround = f"""    requirejs.config({{
        baseUrl: 'https://cdn.jsdelivr.net/npm/',
        paths: {paths}
    }});
    """
    
    return workaround
    
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
           
@add_autoincrement
def render(chart, id="vega-chart"):
    """
    Helper function to plot altair visualizations.
    """
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

# setting up altair
workaround = prepare_altair()
HTML("".join((
    "<script>",
    workaround,
    "</script>",
)))


# It's time to take a look in all files provided by the dataset.

# In[ ]:


print('Data Files in Directory')
print(os.listdir(DATA_PATH))


# For now, I will ignore all small dataset versions.<br>
# Time to import relvant (Ratings, Links and Metadata) files and check the data.

# In[ ]:


ratings = pd.read_csv(DATA_PATH+'ratings.csv')
links = pd.read_csv(DATA_PATH+'links.csv')
metadata = pd.read_csv(DATA_PATH+'movies_metadata.csv')


# In[ ]:


# Function that I wrote to print all relevant infos in dataset
import io

def get_df_info(df):
    display(df.head(3))
    buf = io.StringIO()
    df.info(buf=buf)
    info = buf.getvalue().split('\n')[-2]
    display(f'Number of Rows: {df.shape[0]}, Number of Columns: {df.shape[1]}')
    display('Data Types')
    df_types = df.dtypes
    df_types = pd.DataFrame({'Column':df_types.index, 'Type':df_types.values})
    display(df_types) 
    display(info)
    missing = df.isnull().sum().sort_values(ascending=False)
    display('Missing Values')
    if missing.values.sum() == 0:
        display('No Missing Values')
    else:
        missing = missing[missing > 0]
        missing = pd.DataFrame({'Column' : missing.index, 'Missing Values' : missing.values})
        display(missing)


# <a id='bkground'></a>
# <div class="h3"><center>Ratings Content</center></div>

# In[ ]:


get_df_info(ratings)


# <a id='bkground'></a>
# <div class="h3"><center>Links Content</center></div>

# In[ ]:


get_df_info(links)


# <a id='bkground'></a>
# <div class="h3"><center>Metadata Content</center></div>

# In[ ]:


get_df_info(metadata)


# In all the data info displayed, we can see that only ratings have a large memory usage, and I will use this dataset to make the recommendation system based on user ratings, the dataset have the relevant data like userId, movieId and ratings.<br>
# It's important to say that the ratings dataset doesn't have any missing value, therefore, will not needed any treatment like data imputation or drop NA rows.<br>
# The other datasets will be used for Exploratory Data Analysis.
# 

# <br>
# <a id='bkground'></a>
# <div class="h2"><center>Exploratory Data Analysis</center></div>
# <br>

# <div class="h3">Let's start with the following approaches</div><br>
# * Rating Frequency.
# * Analysis of most rated movies.
# * World cloud with most common words.
# 
# Let's start plotting an Histogram to see the rating distribution.

# In[ ]:


plt.rcParams['figure.figsize'] = (7, 6)
plt.hist(ratings['rating'], bins=10);
plt.title('Ratings Count', size=10)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show();


# And for a better visualization, let's represent by a pie chart with the percent representation.

# In[ ]:


values = ratings.rating.value_counts()
labels = values.index
colors = ['red', 'blue', 'green', 'yellow', 'black']
trace = go.Pie(labels=labels, 
               values=values,
                marker=dict(colors=colors) 
              )
layout = go.Layout(title='Ratings by Total Percent')
fig = go.Figure(data=trace, layout=layout)
iplot(fig)


# We can see most movies were rated with 4, on a scale of 1 to 5. A fewer movies (compared to the total dataset) were rated with low grades. <br>
# Let's see which movies were rated most times, taking the 10 most rated.

# In[ ]:


df_aux = ratings['movieId'].value_counts().reset_index().head(10).rename(columns={'index': 'movieId', 'movieId': 'count'})
df_aux['movieId'] = df_aux['movieId'].astype(str)


# In[ ]:


render(alt.Chart(df_aux).mark_bar().encode(
    x=alt.X('movieId:N', axis=alt.Axis(title='Movie ID'), sort=list(df_aux['movieId'].values)),
    y=alt.Y('count:Q', axis=alt.Axis(title='Total Count')),
    tooltip=['movieId', 'count']
).properties(title='Movie Count', height=300, width=800).interactive())


# Time to discover which movies have this IDs.<br> 
# Let's check the IDs on IMDB and get some info.

# In[ ]:


# Get the Movie on metadata
def get_movie_metadata(movieId):
    metadata['imdb_id'] = metadata['imdb_id'].astype('category')
    imdb_id = links[links['movieId'] == movieId]
    imdb_id = imdb_id.imdbId.values[0]
    if len(str(imdb_id)) == 7:
        movie_rated = metadata[metadata['imdb_id'] == 'tt'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    elif len(str(imdb_id)) == 6:
        movie_rated = metadata[metadata['imdb_id'] == 'tt0'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    elif len(str(imdb_id)) == 5:
        movie_rated = metadata[metadata['imdb_id'] == 'tt00'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    elif len(str(imdb_id)) == 4:
        movie_rated = metadata[metadata['imdb_id'] == 'tt000'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    elif len(str(imdb_id)) == 3:
        movie_rated = metadata[metadata['imdb_id'] == 'tt0000'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    elif len(str(imdb_id)) == 2:
        movie_rated = metadata[metadata['imdb_id'] == 'tt00000'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    elif len(str(imdb_id)) == 1:
        movie_rated = metadata[metadata['imdb_id'] == 'tt000000'+imdb_id.astype(str)]
        df = movie_rated.loc[:,['title', 'overview', 'vote_average', 'release_date']]
        return df.reset_index(drop=True)
    else:
        pass
# Get Movie List
def get_movie(df):
    movieIdIdx = df['movieId'].values.astype(int)
    df_aux_b = pd.DataFrame({'title': ['aaa'], 
                           'overview': ['bbb'], 
                           'vote_average': [1.7], 
                           'release_date': ['1999-01-01']
        })
    for i in movieIdIdx:
        df_aux_b = df_aux_b.append(get_movie_metadata(i), ignore_index=True)

    df_aux_b.drop(0, inplace=True)
    df_aux_b['release_date'] = df_aux_b['release_date'].apply(lambda x : x.split('-')[0])
    df_aux_b['release_date'] = df_aux_b['release_date'].astype(int)
    df_aux_b.rename(columns={'release_date' : 'release_year'}, inplace=True)
    return df_aux_b.reset_index(drop=True)


# In[ ]:


df_movies = get_movie(df_aux)
df_movies


# Nice, we have good movies in the list, I'm a great fan of Forrest Gump and The Silence of the Lambs!<br>
# Most of the movies listed were released in 90's, the only exception is Star Wars (1977).<br>
# Maybe most people are always viewing 90's movies? It's a interesting information to take note.<br>
# <br>
# Another curios information is that in the most rated movies, none have a rate above 9.0.<br>
# Let's expand it, let's get all the 1000 most rated movies and examine what words are frequent in they overviews.

# In[ ]:


df_aux = ratings['movieId'].value_counts().reset_index().head(1001).rename(columns={'index': 'movieId', 'movieId': 'count'})
df_aux['movieId'] = df_aux['movieId'].astype(str)
df_aux = get_movie(df_aux)
get_df_info(df_aux)


# Time to use Natural Language Processing (NLP) with NLTK module and transform everything in overview for lower case, word tokens and remove stopwords and make the Word Cloud.

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

df_aux['overview'] = df_aux.overview.apply(lambda x : x.lower())
df_aux['overview'] = df_aux.overview.apply(lambda x : tokenizer.tokenize(x))
df_aux['overview'] = df_aux.overview.apply(lambda x : [w for w in x if w not in stop_words])
df_aux['overview'] = df_aux.overview.apply(lambda x : ' '.join(x))

word_count = df_aux.overview.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).sort_values(ascending=False)
word_count = pd.DataFrame({'word' : word_count.index, 'count': word_count.values})
# Plot the WordCloud
d = {}
for a, x in word_count.values:
    d[a] = x

wordcloud = WordCloud(background_color = 'white',
                      max_words = 50,
                      width = 2000,
                      height = 2000)
wordcloud.generate_from_frequencies(frequencies=d)
plt.rcParams['figure.figsize'] = (10, 10)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('Most Frequent words in TOP 1000 rated movies', fontsize = 25)
plt.show();


# The words "Life" and "World" are commons and connecting with other words like "Love", "Family", "Father" and "Wife", we caan conclude most of the movies are Family Friendly. <br>
# As we can see, the word "Man" have a big presence on the Word Cloud, and looking at the Top 10 Movies Rated most movies have a male protagonist. <br>
# This concludes the EDA and it's time to start to build the model.

# In[ ]:


del ratings, df_aux, df_movies
gc.collect()


# <br>
# <a id='bkground'></a>
# <div class="h2"><center>Model using PySpark</center></div>
# <br>

# Let's start importing all needed models and setting Spark.

# In[ ]:


import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext('local')
spark = SparkSession(sc)


# Create the Schema.

# In[ ]:


data_schema = StructType([
    StructField('userId', IntegerType(), False),
    StructField('movieId', IntegerType(), False),
    StructField('rating', FloatType(), False),
    StructField('timestamp',IntegerType(), False)
])
final_stat = spark.read.csv(
    '/kaggle/input/the-movies-dataset/ratings.csv', header=True, schema=data_schema
).cache()

ratings = (final_stat.select(
    'userId',
    'movieId',
    'rating'
)).cache()


# Split in Train (70%) and Test (30%).

# In[ ]:


(training, test) = ratings.randomSplit([0.7, 0.3], seed=42)


# And train the model, the evaluation will be made on test set using Mean Absolute Error (MAE).

# In[ ]:


als = ALS(
          rank=30,
          maxIter=4, 
          regParam=0.1,
          userCol='userId', 
          itemCol='movieId', 
          ratingCol='rating',
          coldStartStrategy='drop',
          implicitPrefs=False
         )
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName='mae', labelCol='rating',
                                predictionCol='prediction')

mae = evaluator.evaluate(predictions)
print(f'MAE (Test) = {mae}')


# And finally, generate the Best recommendation for each user (User Based Recommendation System).
# The movieId, the first element in recommendations vector, is the same of ratings dataframe.

# In[ ]:


model.recommendForAllUsers(1).show(5)


# Let's see which movie was recommended for a particular userId.

# In[ ]:


get_movie_metadata(156589)


# Show the most recommended user for each movie (Item Based Recommendation System).
# Again, the movieId is the same of ratings dataframe.

# In[ ]:


model.recommendForAllItems(1).show(5)


# And this finish the work!

# <br>
# <a id='bkground'></a>
# <div class="h2"><center>End Notes</center></div>
# <br>

# As the Kernel have memory and storage limitations, I couldn't get better results, but what I got was enough to demonstrate how to work with Recommendation Systems in PySpark.
# 
# Thanks for your reading!
