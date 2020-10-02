#!/usr/bin/env python
# coding: utf-8

# # Semantic Maps for Documents
# 
# This is an approach I take often to better understand the semantic landscape of a corpus. Using Soft Cosine Similarity with word embeddings, this process works particularly well when the corpus is a collection of short texts like reviews, comments, or tweets, where approaches like TFIDF fall short.  For this notebook we'll apply this approach to better understand Kramer's dialogue throughout Seinfeld.
# 
# The steps are as follows:
# 
# 1. Process all the documents so each document is a collection of tokens that are in the word embedding vocabulary. In our case, we're using the `glove-wiki-gigaword-50` corpus, where all tokens are lowercase, so we'll lowercase all the words. Remove documents that don't have any words in the vocabulary.
# 2. Calculate a TFIDF matrix and use this to calculate the cosine distance between each document's TFIDF vector. 
# 3. Using UMAP (or alternatively, t-SNE), take the precomputed distance matrix and project down to 2 dimensions.
# 4. Combine the supporting information for each document into a DataFrame. This includes the original documents, the tokenized documents, and the embeddings (and optionally labels or other attributes that might be interesting to plot).
# 5. Create a 2D scatterplot of the projection. The value here comes from interactive hover tooltips so that you can quickly explore why documents are in a similar location in the projection. We use `plotly` for our visualization, which can create HTML tooltips, so we create those for each point. 
# 6. Repeat steps 3-5 for other similarity measures. In this notebook, we use Soft Cosine Distance, which takes a long time to calculate, but is able to use the word embeddings to capture more semantic information about the documents. Comparatively, TFIDF-cosine distance only captures overlap in tokens.
# 
# Beyond these visualizations, there are a lot of other places you can take this. One possibility would be to add labels as colors to the plot -- this will actually give you a good idea of how easy a classification machine learning task would be by leveraging the topological properties of the data. If your color labels are distinct in the 2D projection, generally models will have an easier time distinguishing classes when those features are used. Another possibility is to perform clustering (using something like HDBSCAN) to find collections of documents that have similar semantic information and use those clusters like an informal topic model.
# 
# Finally, my process is iterative going between the scatterplots and the processing steps, but this isn't indicated in the notebook. Often times the outliers in the scatterplots can indicate additional areas to improve on the text preprocessing.

# In[ ]:


from functools import partial
from itertools import combinations
from textwrap import wrap

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim.downloader as api
import plotly.graph_objs as go
import spacy
import umap
from gensim.corpora import Dictionary
from gensim.matutils import softcossim
from plotly.offline import init_notebook_mode, iplot


# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nnlp = spacy.load("en")\n\nmodel = api.load("glove-wiki-gigaword-50");\nmodel.init_sims(replace=True)')


# In[ ]:


scripts = pd.read_csv("../input/scripts.csv", index_col=0)


# In[ ]:


scripts.head()


# In[ ]:


scripts.tail()


# In[ ]:


character = "KRAMER"
character_script = scripts[(scripts["Character"] == character)]
print(f"original n lines: {len(scripts)}, character n lines {len(character_script)}")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndialogues = character_script["Dialogue"].astype(str).tolist()\n\ntokenized_docs = []\nfor i, doc in enumerate(nlp.pipe(dialogues, n_threads=-1)):\n    tokens = [token.lower_ for token in doc if token.lower_ in model.vocab and token.is_alpha]\n    if len(tokens) > 0:\n        tokenized_docs.append((i, tokens))\n\nd_index, d_tokenized = zip(*tokenized_docs)\nkept_dialogues = [dialogues[i] for i in d_index]\nprint(f"original: {len(dialogues)}, reduced: {len(kept_dialogues)}")')


# ## TFIDF - Cosine Similarity

# In[ ]:


tfidf_vectors = TfidfVectorizer(analyzer=lambda x: x).fit_transform(d_tokenized)
cos_dist = 1 - (tfidf_vectors.toarray() * tfidf_vectors.T)


# In[ ]:


tfidf_embedding = umap.UMAP(metric="precomputed", random_state=666).fit_transform(cos_dist)
embedding_df = pd.DataFrame(tfidf_embedding, columns=["dim0", "dim1"])
sentence_text_series = pd.Series(kept_dialogues, name="text")
sentence_token_series = pd.Series(d_tokenized, name="tokens")
tfidf_df = pd.concat([sentence_text_series, sentence_token_series, embedding_df], axis=1)


# In[ ]:


def build_tooltip(row):
    text = "<br>".join(wrap(row["text"], 40))
    tokens = "<br>".join(wrap(", ".join(row["tokens"]), 40))
    full_string = [
        "<b>Text:</b> ",
        text,
        "<br>",
        "<b>Tokens:</b> ",
        tokens
    ]
    return "".join(full_string)

tfidf_df["tooltip"] = tfidf_df.apply(build_tooltip, axis=1)


# In[ ]:


init_notebook_mode(connected=True)

trace = go.Scatter(
    x = tfidf_df["dim0"],
    y = tfidf_df["dim1"],
    name = "TFIDF Embedding",
    mode = "markers",
    marker = dict(
        color = "rgba(49, 76, 182, .8)",
        size = 5,
        line = dict(width=1)),
    text=tfidf_df["tooltip"])

layout = dict(title="2D Embeddings - TFIDF",
             yaxis = dict(zeroline=False),
             xaxis = dict(zeroline=False),
             hovermode = "closest")

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# ## Soft Cosine Similarity

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndictionary = Dictionary(d_tokenized)\ncorpus = [dictionary.doc2bow(document) for document in d_tokenized]\nsimilarity_matrix = model.similarity_matrix(dictionary)\ncorpus_softcossim = partial(softcossim, similarity_matrix=similarity_matrix)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsentence_pairs = combinations(corpus, 2)\nscs_sims = [corpus_softcossim(d1, d2) for d1, d2 in sentence_pairs]')


# In[ ]:


n_sentences = len(corpus)
scs_empty = np.zeros((n_sentences, n_sentences))
upper_indices = np.triu_indices(n_sentences, 1)
scs_empty[upper_indices] = scs_sims
scs_sim = np.triu(scs_empty, -1).T + scs_empty
np.fill_diagonal(scs_sim, 1)
scs_dist = 1 - scs_sim


# In[ ]:


scs_embedding = umap.UMAP(metric="precomputed", random_state=666).fit_transform(scs_dist)
scs_embedding_df = pd.DataFrame(scs_embedding, columns=["dim0", "dim1"])
scs_df = pd.concat([sentence_text_series, sentence_token_series, scs_embedding_df], axis=1)


# In[ ]:


scs_df["tooltip"] = scs_df.apply(build_tooltip, axis=1)


# In[ ]:


trace = go.Scatter(
    x = scs_df["dim0"],
    y = scs_df["dim1"],
    name = "SCS Embedding",
    mode = "markers",
    marker = dict(
        color = "rgba(49, 76, 182, .8)",
        size = 5,
        line = dict(width=1)),
    text=scs_df["tooltip"])

layout = dict(title="2D Embeddings - SCS",
             yaxis = dict(zeroline=False),
             xaxis = dict(zeroline=False),
             hovermode = "closest")

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[ ]:




