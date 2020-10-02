#!/usr/bin/env python
# coding: utf-8

# # **Movie Recommender System based on Wiki-plots using sentence embeddings**
# ![recommended movie?](https://alvinalexander.com/sites/default/files/2017-09/netflix-christmas-movie-suggestions.jpg)*You may like these?*
# 
# #### Dataset contains movie plots scraped from wikipedia from 1902-2017 from approximately 22 regions along with important metadata
# <!-- blank line -->
# ----
# ## Content (plot) based Recommender System
# ### We embed all sentences within the plots using Google's [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) and compare the input plot's associated embeddings using [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

# ### **Imports**

# In[ ]:


import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook
import tensorflow as tf
import tensorflow_hub as hub
from nltk import sent_tokenize
from tqdm import tqdm
from scipy import spatial
from operator import itemgetter
tqdm.pandas()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Plotly imports

# In[ ]:


import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# ### Loading Dataset

# In[ ]:


movie = pd.read_csv('../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')
full_mov = pd.read_csv('../input/movie-database/full_mov_db.csv')
full_mov.head()


# In[ ]:


movie.head()


# In[ ]:


len(movie)


# ### Dropping duplicate plots (multi lingual movie releases generally have wiki pages for more than one language versions)

# In[ ]:


## Drop duplicates
movie = movie.drop_duplicates(subset='Plot', keep="first")
len(movie)


# In[ ]:


movie.reset_index(inplace=True)
movie.drop(columns=['index'],inplace=True)
movie.head()


# ### ** bare-bones EDA**

# In[ ]:


movie['word count'] = movie['Plot'].apply(lambda x : len(x.split()))


# In[ ]:


movie['word count'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='no of plots',
    title='Plot Word Count Distribution')


# In[ ]:


movie['Origin/Ethnicity'].value_counts().iplot(kind='bar')


# In[ ]:


movie['Release Year'].value_counts().iplot(kind='bar')


# ### **Plot text preprocessing**

# In[ ]:


import re
def clean_plot(text_list):
    clean_list = []
    for sent in text_list:
        sent = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-.:;<=>?@[\]^`{|}~"""), '',sent)
        sent = sent.replace('[]','')
        sent = re.sub('\d+',' ',sent)
        sent = sent.lower()
        clean_list.append(sent)
    return clean_list


# ### **Embedding using USE** (for more info refer - [USE tutorial](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb))

# In[ ]:


plot_emb_list = []
with tf.Graph().as_default():
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    messages = tf.placeholder(dtype=tf.string, shape=[None])
    output = embed(messages)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        for plot in tqdm_notebook(full_mov['Plot']):
            sent_list = sent_tokenize(plot)
            clean_sent_list = clean_plot(sent_list)
            sent_embed = session.run(output, feed_dict={messages: clean_sent_list})
            plot_emb_list.append(sent_embed.mean(axis=0).reshape(1,512))            
full_mov['embeddings'] = plot_emb_list
full_mov.head()


# ### Pickling the embeddings for future (re)use

# In[ ]:


full_mov.to_pickle('./updated_mov_df_use_2.pkl')


# ## Similar Movie function

# In[ ]:


def similar_movie(movie_name,topn=5):
    plot = full_mov.loc[movie_name,'Plot'][:1][0]
    with tf.Graph().as_default():
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sent_list = sent_tokenize(plot)
            clean_sent_list = clean_plot(sent_list)
            sent_embed2 = (session.run(output, feed_dict={messages: clean_sent_list})).mean(axis=0).reshape(1,512)
            similarities = []
            for tensor,title in zip(full_mov['embeddings'],full_mov.index):
                cos_sim = 1 - spatial.distance.cosine(sent_embed2,tensor)
                similarities.append((title,cos_sim))
            return sorted(similarities,key=itemgetter(1),reverse=True)[1:topn+1]


# ### Testing our model - using Interstellar's plot

# In[ ]:


full_mov.set_index('Title', inplace=True)


# In[ ]:


similar_movie('Interstellar')


# ## **The results seem to make sense**
# ### You can use other embeddings like BERT/ELMo/Flair to get better/different results
# ### An upvote will be appreciated :)
