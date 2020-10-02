#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import ast # Abstract Syntax Trees; handling of JSON content

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# load data into data frame
df = pd.read_csv('../input/covid19-research-preprint-data/COVID-19-Preprint-Data_ver5.csv')


# # Basic explorations

# In[ ]:


df.head()


# In[ ]:


df.describe(include='all')


# In[ ]:


tab_by_date = df['Date of Upload'].value_counts().sort_index()
plt.figure(figsize=(20,6))
tab_by_date.plot(kind='bar')
plt.grid()
plt.title('Date of Upload - Distribution')
plt.show()


# In[ ]:


df['Uploaded Site'].value_counts().plot(kind='bar')
plt.title('Uploaded Site')
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
df['Number of Authors'].hist(bins=50)
plt.title('Number of Authors')
plt.show()


# # Wordcloud of Titles

# In[ ]:


text = " ".join(title for title in df['Title of preprint'])
stopwords = set(STOPWORDS)


# In[ ]:


wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                      width = 600, height = 400,
                      background_color="white").generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# # Wordcloud of Abstracts

# In[ ]:


text = " ".join(abst for abst in df.Abstract)
stopwords = set(STOPWORDS)


# In[ ]:


wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                      width = 600, height = 400,
                      background_color="white").generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# # Search for keywords

# In[ ]:


# define keyword
my_keyword = 'Remdesivir'


# In[ ]:


def word_finder(i_word, i_text):
    found = str(i_text.lower()).find(i_word.lower())
    if found == -1:
        result = 0
    else:
        result = 1
    return result

# partial function for mapping
word_indicator_partial = lambda text: word_finder(my_keyword, text)
# build indicator vector (0/1) of hits
keyword_indicator = np.asarray(list(map(word_indicator_partial, df.Abstract)))


# In[ ]:


# number of hits
print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())


# In[ ]:


# add index vector as additional column
df['selection'] = keyword_indicator

# select only hits from data frame
df_hits = df[df['selection']==1]


# In[ ]:


# show results
df_hits


# In[ ]:


# look at an example: title...,
example_row = 1
df_hits['Title of preprint'].iloc[example_row]


# In[ ]:


# ... abstract
df_hits.Abstract.iloc[example_row]


# In[ ]:


# ... and authors
author_list = ast.literal_eval(df_hits.Authors.iloc[example_row])
author_list


# In[ ]:


# and corresponding institution counts
author_dict = ast.literal_eval(df_hits['Author(s) Institutions'].iloc[example_row])
author_dict


# In[ ]:


# finally a wordcloud of the selected results' abstracts
text = " ".join(abst for abst in df_hits.Abstract)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=200,
                      width = 600, height = 400,
                      background_color="white").generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


# save results in CSV file for further processing
df_hits.to_csv('results.csv')


# # Trying SciSpacy

# In[ ]:


get_ipython().system('pip install scispacy')

# medium model
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz')

# named entity extraction
# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz    ')


# In[ ]:


import scispacy
import spacy

from spacy import displacy

import en_core_sci_md
import en_ner_bc5cdr_md


# In[ ]:


# look at an abstract
text = df_hits.Abstract.iloc[10]
text


# In[ ]:


nlp = en_core_sci_md.load()
doc = nlp(text)


# In[ ]:


# sentence parsing demo
displacy.render(next(doc.sents), style='dep', jupyter=True)


# In[ ]:


# Try basic entity extraction
doc.ents


# In[ ]:


# display entities
displacy.render(doc.sents, style='ent', jupyter=True)


# In[ ]:


# use specific Named Entity Recognition
nlp = en_ner_bc5cdr_md.load()


# In[ ]:


doc = nlp(text)


# In[ ]:


# Try more specific entity extraction
doc.ents


# In[ ]:


# display entities
displacy.render(doc.sents, style='ent', jupyter=True)

