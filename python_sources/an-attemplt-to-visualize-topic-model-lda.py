#!/usr/bin/env python
# coding: utf-8

# # An Attemplt to Visualize Topic Model (LDA)
# 
# I'm curious how well topic models (latent Dirichlet allocation algorithm, specifically) performs in terms of seperating the corpus into semantically similar groups. This is an attempt of me to visually figure it out.
# 
# The topic model is taken from [CareerVillage.org Recommendation Engine](https://www.kaggle.com/danielbecker/careervillage-org-recommendation-engine) by [Daniel Becker](https://www.kaggle.com/danielbecker) (it's fantastic; go upvote) with a few modifications:
# 
# 1. Slightly different text cleaning
# 2. Hashtags are removed. I want to separate natural language understanding from (implicit) tag grouping in this task, that is, only focus on the questions, not tags.
# 3. The number of topics is reduced from 18 to 10. (This is just to make life easier for me. You can use whatever number you like.)
# 
# You can skip all the preprocessing and model fitting using the links below. For visualization, first I try to plot the tf-idf feature space using [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/). Then I use the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-large/3) from Google and plot the sentence embeddings. (The sentence embeddings are more likely to capture semantic similarities than tf-idf).

# ## Contents
# 
# 1. [Imports](#Imports)
# 2. [Preprocessing](#Preprocessing)
#   * [Checking](#Checking)
#   * [The Real Deal](#The-Real-Deal)
# 3. [Topic Modeling](#Topic-Modeling)
#   * [Show Topics](#Show-Topics)
# 4. [TF-IDF Space](#TF-IDF-Sace)
#   * [Prepare TF-IDF Matrix](#Prepare-TF-IDF-Matrix)
#   * [Dimension Reduction using UMAP](#Dimension-Reduction-using-UMAP)
# 5. [Sentence Embeddings Space](#Sentence-Embeddings-Space)
#   * [Extract the embeddings](#Extract-the-embeddings)
#   * [Visualization](#Visualization)
#   * [Examine the sentence embeddings](#Examine-the-sentence-embeddings)
# 6. [Summary](#Summary)

# ## Imports

# In[ ]:


import os
import re
import html as ihtml
import warnings
import random
warnings.filterwarnings('ignore')

os.environ["TFHUB_CACHE_DIR"] = "/tmp/"

import spacy
nlp = spacy.load('en_core_web_sm')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')
#nlp.remove_pipe('tagger')

import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import gensim
import scipy

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import seaborn as sns
import umap

pd.set_option('display.max_colwidth', -1)

SEED = 13
random.seed(SEED)
np.random.seed(SEED)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Preprocessing

# In[ ]:


input_dir = '../input/'

questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))


# In[ ]:


# Spacy Tokenfilter for part-of-speech tagging
token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']

def clean_text(text, remove_hashtags=True):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    if remove_hashtags:
        text = re.sub(r"#[a-zA-Z\-]+", "", text)
    text = re.sub(r"\s+", " ", text)        
    return text

def nlp_preprocessing(data):
    """ Use NLP to transform the text corpus to cleaned sentences and word tokens

    """    
    def token_filter(token):
        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list

        """    
        return not token.is_stop and token.is_alpha and token.pos_ in token_pos
    
    processed_tokens = []
    data_pipe = nlp.pipe(data, n_threads=4)
    for doc in data_pipe:
        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]
        processed_tokens.append(filtered_tokens)
    return processed_tokens


# In[ ]:


questions['questions_full_text'] = questions['questions_title'] + ' '+ questions['questions_body']


# ### Checking

# In[ ]:


sample_text = questions[questions['questions_full_text'].str.contains("&a")]["questions_full_text"].iloc[0]
sample_text


# In[ ]:


sample_text = clean_text(sample_text)
sample_text


# In[ ]:


sample = nlp_preprocessing([sample_text])
" ".join(sample[0])


# ### The Real Deal

# In[ ]:


get_ipython().run_cell_magic('time', '', "questions['questions_full_text'] = questions['questions_full_text'].apply(clean_text)")


# In[ ]:


questions['questions_full_text'].sample(2)


# In[ ]:


get_ipython().run_cell_magic('time', '', "questions['nlp_tokens'] = nlp_preprocessing(questions['questions_full_text'])")


# In[ ]:


questions['nlp_tokens'].sample(2)


# ## Topic Modeling

# In[ ]:


# Gensim Dictionary
extremes_no_below = 10
extremes_no_above = 0.6
extremes_keep_n = 8000

# LDA
num_topics = 10
passes = 20
chunksize = 1000
alpha = 1/50


# In[ ]:


def get_model_results(ldamodel, corpus, dictionary):
    """ Create doc-topic probabilities table and visualization for the LDA model

    """  
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    transformed = ldamodel.get_document_topics(corpus)
    df = pd.DataFrame.from_records([{v:k for v, k in row} for row in transformed])
    return vis, df  


# In[ ]:


get_ipython().run_cell_magic('time', '', "lda_tokens = questions['nlp_tokens']\n\n# Gensim Dictionary\nlda_dic = gensim.corpora.Dictionary(lda_tokens)\nlda_dic.filter_extremes(no_below=extremes_no_below, no_above=extremes_no_above, keep_n=extremes_keep_n)\nlda_corpus = [lda_dic.doc2bow(doc) for doc in lda_tokens]\n\nlda_tfidf = gensim.models.TfidfModel(lda_corpus)\nlda_corpus = lda_tfidf[lda_corpus]\n\n# Create LDA Model\nlda_model = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=num_topics, \n                                            id2word = lda_dic, passes=passes,\n                                            chunksize=chunksize,update_every=0,\n                                            alpha=alpha, random_state=SEED)")


# In[ ]:


# Create Visualization and Doc-Topic Probapilities
lda_vis, lda_result = get_model_results(lda_model, lda_corpus, lda_dic)
lda_questions = questions[['questions_id', 'questions_title', 'questions_body']]
lda_questions = pd.concat([lda_questions, lda_result.add_prefix('Topic_')], axis=1)


# ### Show Topics

# In[ ]:


# Disabled for compatibility issue
# lda_vis


# In[ ]:


print("\n\n".join(["Topic{}:\n {}".format(i, j) for i, j in lda_model.print_topics()]))


# ## TF-IDF Space
# 
# Here we use cosine similarities. Questions with similar term frequency distribution will have higher similarity scores. It requires the TF-IDF vectors to have the same L2-norms. (I'm not sure if UMAP does the normalization for us or not. The safer way is to do it ourselves.)

# ### Prepare TF-IDF Matrix

# In[ ]:


corpus_csr = gensim.matutils.corpus2csc(lda_corpus).T


# In[ ]:


# There exist some zero rows:
non_zeros = np.where(corpus_csr.sum(1) != 0)[0]
print(corpus_csr.shape[0])
corpus_csr = corpus_csr[non_zeros, :]
print(corpus_csr.shape[0])


# In[ ]:


# Normalize by row
corpus_csr = corpus_csr.multiply(
    scipy.sparse.csr_matrix(1/np.sqrt(corpus_csr.multiply(corpus_csr).sum(1))))


# In[ ]:


# Double check the norms
np.sum(np.abs(corpus_csr.multiply(corpus_csr).sum(1) - 1) > 0.001)


# ### Dimension Reduction using UMAP

# In[ ]:


get_ipython().run_cell_magic('time', '', 'embedding = umap.UMAP(metric="cosine", n_components=2).fit_transform(corpus_csr)')


# In[ ]:


df_emb = pd.DataFrame(embedding, columns=["x", "y"])
df_emb["label"] = np.argmax(lda_result.iloc[non_zeros].fillna(0).values, axis=1)


# In[ ]:


df_emb_sample = df_emb.sample(5000)
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    df_emb_sample["x"].values, df_emb_sample["y"].values, s=2, c=df_emb_sample["label"].values# , cmap="Spectral"
)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
plt.title("TF-IDF matrix embedded into two dimensions by UMAP", fontsize=18)
plt.show()


# Questions from different topics all got mixed together. Not much can be seen in the above plot. Let's try to plot each topic separately:

# In[ ]:


g = sns.FacetGrid(df_emb, col="label", col_wrap=2, height=5, aspect=1)
g.map(plt.scatter, "x", "y", s=0.2).fig.subplots_adjust(wspace=.05, hspace=.5)


# We're able to see some patterns now. Note since LDA is a mixture model, a question can be assigned multiple topics. What if we only plot questions with only one dominant topic?

# In[ ]:


# keep well separated points
df_emb_sample = df_emb[np.amax(lda_result.iloc[non_zeros].fillna(0).values, axis=1) > 0.7]
print("Before:", df_emb.shape[0], "After:", df_emb_sample.shape[0])
g = sns.FacetGrid(df_emb_sample, col="label", col_wrap=2, height=5, aspect=1)
g.map(plt.scatter, "x", "y", s=0.3).fig.subplots_adjust(wspace=.05, hspace=.5)


# We removed ~3,000 questions (the number is relative low, so in this model topics are quite distinguishable). The results are not much different, though.

# ## Sentence Embeddings Space
# 
# The model used is the universal sentence encoder (large/transformer) version 3. The extracted sentence embeddings will have a dimension of 512. Here we also use cosine similarity.

# ### Extract the embeddings

# In[ ]:


embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")


# In[ ]:


import logging
from tqdm import tqdm_notebook
tf.logging.set_verbosity(logging.WARNING)
BATCH_SIZE = 128

sentence_input = tf.placeholder(tf.string, shape=(None))
# For evaluation we use exactly normalized rather than
# approximately normalized.
sentence_emb = tf.nn.l2_normalize(embed(sentence_input), axis=1)

sentence_embeddings = []       
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in tqdm_notebook(range(0, len(questions), BATCH_SIZE)):
        sentence_embeddings.append(
            session.run(
                sentence_emb, 
                feed_dict={
                    sentence_input: questions["questions_full_text"].iloc[i:(i+BATCH_SIZE)].values
                }
            )
        )


# In[ ]:


sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
sentence_embeddings.shape


# ### Visualization

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import umap\nembedding = umap.UMAP(metric="cosine", n_components=2).fit_transform(sentence_embeddings)')


# In[ ]:


df_se_emb = pd.DataFrame(embedding, columns=["x", "y"])
df_se_emb["label"] = np.argmax(lda_result.fillna(0).values, axis=1)
df_se_emb["label"] = df_se_emb["label"].astype("category")


# In[ ]:


df_emb_sample = df_se_emb.sample(5000)
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    df_emb_sample["x"].values, df_emb_sample["y"].values, s=2, c=df_emb_sample["label"].values# , cmap="Spectral"
)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
plt.title("Sentence embeddings embedded into two dimensions by UMAP", fontsize=18)
plt.show()


# Still the topics are not very separable from each other. Let's check the topics one by one:

# In[ ]:


g = sns.FacetGrid(df_se_emb, col="label", col_wrap=2, height=5, aspect=1)
g.map(plt.scatter, "x", "y", s=0.2).fig.subplots_adjust(wspace=.05, hspace=.5)


# Arguably the pattern is slightly stronger here than the TF-IDF one. We can see more obvious small clusters(higher density areas) here. But still a topic can contains some very semantically different questions (large distance).

# ### Examine the sentence embeddings

# Let's check if the sentence embeddings are doing a good job by finding similar questions to some random samples of questions:

# In[ ]:


def find_similar(idx, top_k):
    cosine_similarities = sentence_embeddings @ sentence_embeddings[idx][:, np.newaxis]
    return np.argsort(cosine_similarities[:, 0])[::-1][1:(top_k+1)]


# In[ ]:


IDX = 0
similar_ids = find_similar(IDX, top_k=3).tolist()
for idx in [IDX] + similar_ids:
    print(questions["questions_full_text"].iloc[idx], "\n")


# In[ ]:


IDX = 5
similar_ids = find_similar(IDX, top_k=3).tolist()
for idx in [IDX] + similar_ids:
    print(questions["questions_full_text"].iloc[idx], "\n")


# In[ ]:


IDX = 522
similar_ids = find_similar(IDX, top_k=3).tolist()
for idx in [IDX] + similar_ids:
    print(questions["questions_full_text"].iloc[idx], "\n")


# In[ ]:


IDX = 13331
similar_ids = find_similar(IDX, top_k=3).tolist()
for idx in [IDX] + similar_ids:
    print(questions["questions_full_text"].iloc[idx], "\n")


# Looks reasonable. We can actually build a recommendation upon these embeddings directly.

# ## Summary
# 
# Topic model provides unsupervised clustering/classification that is (somewhat) interpretable. Most of the time we can tell what a topic is about once we see its most distinguishable words. However, as we found out in the visualization, the sensitivity of the model is quite high. A topic can contain a very diverse set of questions. One way to solve this is to increase the number of topics, but it is hard to tell how many is enough.
# 
# We also explores the topic model visualized under sentence embeddings space. The word/term-based model seems to successfully captured some semantic information as we can see some obvious clusters.
# 
# But if we just want to find questions in the close vicinity in the semantic space of a question, directly using the sentence embeddings and cosing similarity might be enough. 
