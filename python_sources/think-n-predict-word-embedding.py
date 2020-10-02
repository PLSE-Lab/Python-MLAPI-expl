#!/usr/bin/env python
# coding: utf-8

# # Think'n'Predict: Word Embedding
# This notebook is meant to clarify our approach of word embedding using word2vec for job descriptions

# ## 1. Data Loading

# In[1]:


import pandas as pd
DATA_PATH = '../input/indeed.csv'
df = pd.read_csv(DATA_PATH)
df = df.sample(1000)
df.head()


# ## 2. Data Processing
# A lot of advanced NLP libraries exists in python, here we will implement two of them: NLTK & spiCy
# ### 2.1 Using NLTK
# #### 2.1.1 Loading libraries

# In[ ]:


from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

wordnet_lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))


# #### 2.1.2 Tokenize & Lemmatize
# Here we will do the NLP pipeline of lemmatizing (based on part-of-speech), tokenizing and removing stopwords using NLTK

# In[ ]:


# This function will get the part-of-speech of a certain word
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[ ]:


# Here we will go though our data and tokenize each description then we lemmatize each token (word) based on its part-of-speech tag and also removing the stop-words
nltk_processed_data = []
for description in df.description :
    processed_description = [wordnet_lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w.lower())) for w in word_tokenize(description) if w.lower() not in stopWords]
    nltk_processed_data.append(processed_description)


# ### 2.2 Using spaCy
# #### 2.2.1 Loading libraries

# In[ ]:


import string
import spacy

# !python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')


# #### 2.2.2 Tokenize & Lemmatize
# Here we will do the NLP pipeline of lemmatizing, tokenizing and removing stopwords using spaCy

# In[ ]:


# Here we will go though our data and do the nlp pipeline provided by spaCy which consits of tokenize -> PoS tag -> Parsing -> NER -> ...
# Then we get the lemmatizaion of the tokens that are not stop words so that we create a list of tokenized descriptions
spacy_processed_data = []
for description in df.description :
    doc = nlp(description)
    processed_description = [token.lemma_.lower() for token in doc if not token.is_stop and str(token).lower() not in string.punctuation]
    spacy_processed_data.append(processed_description)


# ## 3. Word Embedding: word2vec
# Here we will implement the word2vec model and train it on our data

# In[ ]:


import multiprocessing
from gensim.models import Word2Vec

def get_model(processed_data):
    model = Word2Vec(
            processed_data,
            size=150,
            window=10,
            min_count=5,
            workers=multiprocessing.cpu_count())
    model.train(processed_data, total_examples=len(processed_data), epochs=10)
    return model


# #### 3.1 Using NLTK

# In[ ]:


nltk_model = get_model(nltk_processed_data)


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

vocab = list(nltk_model.wv.vocab)
X = nltk_model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()


# In[ ]:


fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()


# #### 3.2 Using spaCy

# In[ ]:


spacy_model = get_model(spacy_processed_data)


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

vocab = list(spacy_model.wv.vocab)
X = spacy_model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()


# In[ ]:


fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()


# In[ ]:


keys = ['data', 'devops', 'mobile', 'blockchain', 'cyber', 'security', 'web']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in spacy_model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(spacy_model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


# In[ ]:


import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


# In[ ]:





def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(30, 30))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Top similar words for each track', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

