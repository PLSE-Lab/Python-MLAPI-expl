#!/usr/bin/env python
# coding: utf-8

# The aim of this notebook is to give an overview of symptoms reported relating to Covid-19 from the given dataset. I re-use code partly from 
# - https://www.kaggle.com/salmanhiro/covids-incubation-transmission-related-articles
# - https://www.kaggle.com/dgunning/browsing-the-papers-with-a-bm25-search-index
# - 

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
        # print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Going through metadata to find articles relevant to Covid Symptoms

# ## Cleaning and Filtering

# As you can see, the metadata file lists all articles in the dataset with some of the most relevant information; the "sha"-value lets us find/load a specific article from the dataset.

# In[ ]:


df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
df.head()


# The "title" and "abstract" columns provide information about the article's content that we'll use to find relevant articles. Of course, without access to the full text, the article will not be of much help. We can get the full text if either the doi is present or has_full_text is True.

# In[ ]:


print("Total number of articles:", len(df))
print("Number of articles without full text:", len(df[df["has_full_text"].isnull() & df["doi"].isnull()]))


# The articles without full text won't be useful, so we'll drop these. We'll also drop columns we won't be using.

# In[ ]:


df = df[["sha", "source_x", "title", "abstract", "doi", "has_full_text"]]
df = df[df["has_full_text"].notna() | df["doi"].notna()]
print(len(df), "articles left")
df.head()


# Of the articles with full text access, we'll now check how many articles are missing abstracts and titles.

# In[ ]:


print("Number of articles with missing abstracts:", len(df[df["abstract"].isnull()]))
print("Number of articles with missing titles:", len(df[df["title"].isnull()]))
print("Number of articles with missing title AND abstract:", len(df[df["title"].isnull() & df["abstract"].isnull()]))


# We can still use articles without abstracts in the metadata file as long as at least the title is present. It seems as though the articles without titles are exactly the articles with missing title AND abstract (these will not be useful at all). Let's drop those.

# In[ ]:


df = df[df["abstract"].notna() | df["title"].notna()]
print(len(df), "articles left")


# Great, we only have articles left that either have an abstract or title in the metadata file and whose full text is available. Let's clean the titles and abstracts!

# In[ ]:


import string
def remove_punc_and_lower(s):
    try:
        if s:
            return s.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower()  # replacing punctuation w/ space (and making lowercase)
    except:
        print(s)
        
def doi_url(d): 
    return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

df['doi'] = df['doi'].apply(lambda s: doi_url(s) if pd.notnull(s) else s)
df['title'] = df['title'].apply(lambda s: remove_punc_and_lower(s) if pd.notnull(s) else s)
df['abstract'] = df['abstract'].apply(lambda s: remove_punc_and_lower(s) if pd.notnull(s) else s)


# In[ ]:


df.head(3)


# //TODO

# ## Selecting Relevant Articles
# 
# We first define a list of keywords relating to "Symptoms" that are of interest.

# In[ ]:


symptoms_keywords = {"symptom", "symptoms", "symptomatology", "semiology", "sign", "signs", "manifestation"}


# The following function will now check whether one of the keywords is present in either the abstract or the title of an article.

# In[ ]:


def filter_keywords(row, keywords=symptoms_keywords):
    text = ""
    if pd.notnull(row["abstract"]):
        text += row["abstract"]
    if pd.notnull(row["title"]):
        text += row["title"]
    for word in text.split():
        if word in keywords:
            return True
    return False


# In[ ]:


m = df.apply(filter_keywords, axis=1)  # mask; true for rows that contain the keywords


# > _df_symptoms_ is now a dataframe of articles that contain keywords relevant to *symptoms*

# In[ ]:


df_symptoms = df[m]
df_symptoms.reset_index(drop=True, inplace=True)


# In[ ]:


print("Number of relevant articles:", len(df_symptoms))


# In[ ]:


df_symptoms.head()


# ## (short) EDA of the Metadata
# 
# We now take a quick look at the metadata of articles relevant to Covid-19 Symptoms.

# In[ ]:


corpus = " ".join(df_symptoms.title.values) + " ".join(df_symptoms.title.values)


# ## Word Cloud

# In[ ]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install mpld3')
import mpld3
mpld3.enable_notebook()


# In[ ]:


wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(corpus)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Most common words

# In[ ]:


from nltk.corpus import stopwords
stopwords_eng = set(stopwords.words("english"))
corpus_without_stopwords = " ".join([word for word in corpus.split() if word not in stopwords_eng])


# In[ ]:


from collections import Counter
c = Counter(corpus_without_stopwords.split())
c.most_common(10)
pd.DataFrame(c.most_common(10), columns=["word", "occurences"])


# # //TODO

# # Using The Articles

# The first objective is to get all the articles with their full text.

# ## Full-Text EDA

# ## SciBERT etc.

# ## Comorbidities
