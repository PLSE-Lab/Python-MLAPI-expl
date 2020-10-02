#!/usr/bin/env python
# coding: utf-8

# In[1]:


from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor 

import os 
import gensim
import pandas as pd

import itertools

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import scikitplot

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF


# ### Data preprocessing

# In[2]:


lemmatizer = WordNetLemmatizer()


def stem(text):
    return lemmatizer.lemmatize(text)


def map_parallel(f, iterable, **kwargs):
    with ProcessPoolExecutor() as pool:
        result = pool.map(f, iterable, **kwargs)
    return result


def retrieve_articles(start, chunksize=1000):
    return arxiv.query(
        search_query=search_query,
        start=start,
        max_results=chunksize
    )


# ### Actual text mining functions

# In[3]:


def vectorize_text(examples_df, vectorized_column='summary', vectorizer=CountVectorizer):

    vectorizer = vectorizer(min_df=2)
    features = vectorizer.fit_transform(examples_df[vectorized_column])

    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(valid_example_categories).reshape(-1, 1)
    labels_ohe = ohe.fit_transform(labels).todense()
    vectorized_data = {
        'features': features,
        'labels': labels,
        'labels_onehot' : labels_ohe
    }
    return vectorized_data, (vectorizer, ohe, le)


def extract_keywords(text):
    """
    Use gensim's textrank-based approach
    """
    return gensim.summarization.keywords(
        text=stem(text),
        lemmatize=True
    )


# In[4]:


def filter_out_small_categories(df, categories, threshold=200):

    class_counts = categories.value_counts()
    too_small_classes = class_counts[class_counts < threshold].index
    too_small_classes

    valid_example_indices = ~categories.isin(too_small_classes)
    valid_examples = df[valid_example_indices]
    valid_example_categories = categories[valid_example_indices]
    
    return valid_examples, valid_example_categories


# ### Print top words for topics from NMF topic model

# In[5]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % (topic_idx + 1)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# ### Plotting utils

# # Load ML articles from arxiv

# In[6]:


data_path = '../input/matrix_factorization_arxiv_query_result.json'
articles_df = pd.read_json(data_path)
articles_df.head()


# ## Display some basic information

# In[7]:


articles_df[['title', 'authors', 'published', 'summary']].head()


# In[8]:


pd.io.json.json_normalize(articles_df['arxiv_primary_category'])


# In[9]:


articles_df.info()


# ## arXiv categories

# In[10]:


categories = articles_df['arxiv_primary_category'].apply(itemgetter('term'))

main_categories = categories.apply(lambda s: s.split('.')[0].split('-')[0])


# Toplevel categories

# In[11]:


main_categories_counts = main_categories.value_counts(ascending=True)
main_categories_counts.plot.barh()
plt.show()


# Toplevel categories with more than 200 papers

# In[12]:


main_categories_counts[main_categories_counts > 200].plot.barh()
plt.show()


# In[13]:


categories.value_counts(ascending=True)[-10:].plot.barh()
plt.show()


# ## Extract keywords from summaries

# In[14]:


get_ipython().run_cell_magic('time', '', "\narticles_df['summary_keywords'] = list(\n    map_parallel(extract_keywords, articles_df['summary'])\n)")


# In[15]:


n_examples = 20 

for __, row in itertools.islice(articles_df.iterrows(), n_examples):
  print(20 * '*')
  print(row['title'])
  print(20 * '*')
  print('keywords:', row['summary_keywords'].split('\n'))
  print()


# In[16]:


article_keyword_lengths = articles_df['summary_keywords'].apply(lambda kws: len(kws.split('\n')))


# In[17]:


article_keyword_lengths.plot.hist(bins=article_keyword_lengths.max(), title='Number of summary keywords')


# ## Topic modeling

# In[18]:


valid_examples, valid_example_categories = filter_out_small_categories(articles_df, main_categories)
valid_examples.shape


# In[19]:


vectorized_data, (vectorizer, ohe, le) = vectorize_text(
    valid_examples,
    vectorized_column='summary_keywords',
    vectorizer=TfidfVectorizer
)


# We'll use Nonnegative Matrix Factorization for topic modeling.

# In[20]:


x_train, x_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    vectorized_data['features'],
    vectorized_data['labels_onehot'],
    vectorized_data['labels'],
    stratify=vectorized_data['labels'],
    test_size=0.2,
    random_state=0
)


# In[21]:


nmf = NMF(n_components=5, solver='mu', beta_loss='kullback-leibler')


# In[22]:


get_ipython().run_cell_magic('time', '', '\ntopics = nmf.fit_transform(x_train)')


# In[23]:


n_top_words = 10

tfidf_feature_names = vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)


# ## Cross tabulate dominant topics and categories

# In[24]:


dominant_topics = topics.argmax(axis=1) + 1
categories = le.inverse_transform(y_train_labels[:,0])
pd.crosstab(dominant_topics, categories)

