#!/usr/bin/env python
# coding: utf-8

# # Clustering German Articles using Word2Vec
# 
# ## Word2Vec
# 
# Word2Vec is a method to represent words in a numerical - vector format such that words that are closely related to each other are close to each other in numeric vector space. This method was developed by Thomas Mikolov in 2013 at Google.
# 
# Each word in the corpus is modeled against surrounding words, in such a way that the surrounding words get maximum probabilities. The mapping that allows this to happen , becomes the word2vec representation of the word. The number of surrounding words can be chosen through a model parameter called "window size". The length of the vector representation is chosen using the parameter 'size'.
# 
# In this notebook, the library gensim is used to construct the word2vec models
# 
# ## Loading the library and getting the data
# 

# In[ ]:




import pandas as pd
import numpy as np
import os
import re
import gensim
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("german")



data_test = pd.read_csv('../input/10k-german-news-articles/Articles.csv')
data_test.head()


# In[ ]:



body_new = [re.sub('<[^<]+?>|&amp', '', word) for word in data_test['Body'].values]
data_test['Body'] = body_new
#! python -m spacy download de_core_news_sm


# To tackle stopwords in German, the spaCy module is used. The spaCy module consists of information regarding stopwords for different languages. The command `! python -m spacy download de_core_news_sm` enables the download of German- related module

# In[ ]:


get_ipython().system(' python -m spacy download de_core_news_sm')
import spacy
import de_core_news_sm
nlp = de_core_news_sm.load()


#nlp = spacy.load('de_core_news_sm')


# ## Dropping NA values and Duplicate Articles
# 
# From the pulled data, rows that contain NA values are dropped. Duplicate rows are also dropped. After this, the columns "Headline" & "Body" are chosen. Each article is indexed with the row number it belongs in.

# In[ ]:


## Drop Duplicates and NA values

data_test_clean = data_test.dropna()
data_test_clean = data_test_clean.drop_duplicates()


# In[ ]:


article_data = data_test_clean[['ID_Article','Title','Body']].drop_duplicates()
article_data.shape


# After dropping rows that contain NA values and rows that have duplicated values, redundant words such as punctuation marks and stopwords are removed. As the texts are in German, the stopword corpus for German is used from the `spacy` library. After this preprocessing, the words are converted to a list of words to be fed into the model.

# In[ ]:



## Convert the body text into a series of words to be fed into the model

def text_clean_tokenize(article_data):
    
    review_lines = list()

    lines = article_data['Body'].values.astype(str).tolist()

    for line in lines:
        tokens = word_tokenize(line)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('','',string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('german'))
        words = [w for w in words if not w in stop_words]
        words = [stemmer.stem(w) for w in words]

        review_lines.append(words)
    return(review_lines)
    
    
review_lines = text_clean_tokenize(article_data)


# In[ ]:


## Building the word2vec model

model =  gensim.models.Word2Vec(sentences = review_lines,
                               size=100,
                               window=2,
                               workers=4,
                               min_count=2,
                               seed=42,
                               iter= 50)

model.save("word2vec.model")


# The model is built on the existing article data we have. Let's have a look at word similarities for a few words. The function `model.wv.similar_by_word(<word>)`, displays top words that are similar to the `<word>` as per cosine similarility

# In[ ]:





word_list = list(model.wv.vocab)

for words in word_list[1:10]:
    print('Similar Words for :',words)
    
    print(model.wv.similar_by_word(words))
    print('--------------------------\n')


# ## Converting articles into numeric vectors
# 
# The word2vec model converts each word into a numeric vector. Each article is converted to a numeric vector by taking the constituent words' numeric vectors and averaging them.
# 
# To do that the following steps are taken
# * Tokenize each article into its constituent words
# * Map each word to numeric vector representation.
# * Take average for each article.
# 

# In[ ]:






#print(word_list)

# Convert each article lines to word2vec representation
import spacy

def tokenize(sent):
    doc = nlp.tokenizer(sent)
    return [token.lower_ for token in doc if not token.is_punct]

new_df = (article_data['Body'].apply(tokenize).apply(pd.Series))

new_df = new_df.stack()
new_df = (new_df.reset_index(level=0)
                .set_index('level_0')
                .rename(columns={0: 'word'}))

new_df = new_df.join(article_data.drop('Body', 1), how='left')


# In[ ]:


new_df = new_df[['word','ID_Article']]
vectors = model.wv[word_list]
vectors_df = pd.DataFrame(vectors)
vectors_df['word'] = word_list
merged_frame = pd.merge(vectors_df, new_df, on='word')
merged_frame_rolled_up = merged_frame.drop('word',axis=1).groupby('ID_Article').mean().reset_index()
del merged_frame
del new_df
del vectors


# ## Numeric Vectors for each article
# 
# A glimpse of the numeric vectors for each article is displayed below. Each article is represented by a 100 long numeric vector. These numeric vectors are then used to compute pair wise cosine simlilarities amongst articles.

# In[ ]:


merged_frame_rolled_up.head()


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_matrix = pd.DataFrame(cosine_similarity(merged_frame_rolled_up))
cosine_matrix.columns = list(merged_frame_rolled_up['ID_Article'])


# ## Getting Recommendations
# 
# The cosine similarity matrix gives a pair wise relationship between articles. This relationship/similarity ranges from 0 to 1. With 0 being least similar and with 1 being most similar.

# In[ ]:


cosine_matrix.head()


# In[ ]:



reco_articles = {}
i = 0
for col_name in cosine_matrix.columns:
    tmp = cosine_matrix[[col_name]].sort_values(by=col_name,ascending=False)
    tmp = tmp.iloc[1:]
    tmp = tmp.head(20)
    recommended_articles = list(article_data[article_data['ID_Article'].isin(tmp.index)]['Title'].values)
    chosen_article = list(article_data[article_data['ID_Article']==col_name]['Title'].values)
    tmp = {'Chosen-Articles': len(recommended_articles)* chosen_article,'Recommended-Articles':recommended_articles}
    reco_articles[i] = tmp
    i = i+1
    del tmp
print('Ended')
    
    
    
    
    


# In[ ]:


## Convert Dictionary Object to a data frame

df_reco = pd.concat([pd.DataFrame(v) for k, v in reco_articles.items()])
df_reco.head()


# In[ ]:


## Making sure that the same articles do not get recommended

df_reco = df_reco[df_reco['Chosen-Articles']!=df_reco['Recommended-Articles']]


# ## Recommendation for 10 randomly chosen articles

# In[ ]:


import random

list_of_articles = df_reco['Chosen-Articles'].values
random.shuffle(list_of_articles)
list_of_articles = list_of_articles[:9]

for article in list_of_articles:
    tmp = df_reco[df_reco['Chosen-Articles']==article]
    print('--------------------------------------- \n')
    print('Recommendation for ',article,' is :')
    print('Recommended Articles')
    print(tmp['Recommended-Articles'].values)
    


# In[ ]:




