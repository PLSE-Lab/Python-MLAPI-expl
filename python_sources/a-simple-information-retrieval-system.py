#!/usr/bin/env python
# coding: utf-8

# # NSU Distributional Semantics 2019 Course. Seminar 2

# In this seminar, we will learn how to build the simples count-based models for information retrieval tasks. We will ide a TED Talks dataset from Kaggle (https://www.kaggle.com/rounakbanik/ted-talks).

# ## Reading data
# At this step, we will open a dataset and select two text columns from it.

# In[ ]:


import pandas as pd


# In[ ]:


data_path = "../input/ted_main.csv"
data = pd.read_csv(data_path)


# In[ ]:


data.head()


# In[ ]:


data = data[['description', 'main_speaker', 'name']]


# In[ ]:


data.head()


# ## Preparing data for processing
# At this step, we are going to prepare our text data. Preparation includes tokenization and stop-words filtering. Today we will omit the second step. We will also be working only with descriptions.

# To make a tokenization, we can simply use tokenizers from nltk.

# In[ ]:


from nltk import WordPunctTokenizer


# In[ ]:


tokenizer = WordPunctTokenizer()


# In[ ]:


descriptions = [tokenizer.tokenize(description.lower()) for description in data["description"]]


# In[ ]:


print(descriptions[0])


# ## Converting texts to a Bag-of-Words format

# In[ ]:


from gensim import corpora


# Here we're gonna use the default Dictionary function (but you can implement your own converter if you wish).

# In[ ]:


corpora_dict = corpora.Dictionary(descriptions)


# In[ ]:


print(corpora_dict.token2id)


# By default, id2token is empty. Let's fill this dictionary.

# In[ ]:


for token, token_id in corpora_dict.token2id.items():
    corpora_dict.id2token[token_id] = token


# In[ ]:


print(corpora_dict.id2token)


# In[ ]:


len(corpora_dict)


# Now let's look at the BoW representation of an arbitrary sentense.

# In[ ]:


new_doc = "Save trees in sake of ecology!"
new_vec = corpora_dict.doc2bow(tokenizer.tokenize(new_doc.lower()))
print(new_vec)

for word_id, _ in new_vec:
    print(corpora_dict.id2token[word_id], end=' ')


# Now let's do it with all texts.

# In[ ]:


corpus = [corpora_dict.doc2bow(text) for text in descriptions]


# In[ ]:


print(corpus[0])


# Now it's time to make a simple search machine.

# ## BoW search machine

# In[ ]:


from gensim import similarities


# In[ ]:


index_bow = similarities.SparseMatrixSimilarity(corpus, num_features=len(corpora_dict))


# In[ ]:


def search(index, query, top_n=5, prints=False):
    """
    This function searches the most similar texts to the query.
        :param index: gensim.similarities object
        :param query: a string
        :param top_n: how many variants it returns
        :param prints: if True returns the results, otherwise prints the results
        :returns: a list of tuples (matched_document_index, similarity_value)
    """
    # getting a BoW vector
    bow_vec = corpora_dict.doc2bow(query.lower().split())
    similarities = index[bow_vec]  # get similarities between the query and all index documents
    similarities = [(x, i) for i, x in enumerate(similarities)]
    similarities.sort(key=lambda elem: -elem[0])  # sorting by similarity_value in decreasing order
    res = []
    if prints:
        print(f"{query}\n")
    for result in similarities[:top_n]:
        if prints:
            print(f"{data['description'][result[1]]} \t {result[0]}\n")
        else:
            res.append((result[1], result[0]))
    if not prints:
        return res


# In[ ]:


search(index_bow, "education system", prints=True)


# In[ ]:


search(index_bow, "healthy food", prints=True)


# Seems like it works. But can our system search texts by citations?

# In[ ]:


search(index_bow, "In an emotionally charged talk", prints=True)


# Great! But what about searching by an annotation?

# In[ ]:


search(index_bow, "Majora Carter: Greening the ghetto", prints=True)


# Seems like our tagret document is not in top-5 results.
# 
# On the next step, we will make more 'smart' model, TF-IDF model. 

# ## TF-IDF model

# In[ ]:


from gensim.models import TfidfModel


# In[ ]:


model_tfidf = TfidfModel(corpus)


# In[ ]:


vector = model_tfidf[corpus[0]]


# In[ ]:


print(vector)


# In[ ]:


corpus_tfidf = model_tfidf[corpus]


# In[ ]:


index_tfidf = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(corpora_dict))


# In[ ]:


search(index_tfidf, "Majora Carter: Greening the ghetto", prints=True)


# Much better! Now that we have Majora Carter talks in the top of the results. How many talks did she have?

# In[ ]:


data[data["main_speaker"] == "Majora Carter"]


# Now, it's time to use dense vectors instead of sparse ones.

# ## Doing SVD / LSA with your own hands
# 
# This approach has a lot of names but it's main idea is quite simple: we try to approximate out source matrix by matrix of a lower rank. In this task, we will use original BoW matrix.

# In[ ]:


from scipy.sparse import coo_matrix


# In[ ]:


i_inds = []
j_inds = []
data_ij_values = []

for i_ind, sparse_doc in enumerate(corpus):
    for j_ind, data_ij in sparse_doc:
        i_inds.append(i_ind)
        j_inds.append(j_ind)
        data_ij_values.append(data_ij)
sparse_corpus = coo_matrix((data_ij_values, (i_inds, j_inds)))
full_corpus = sparse_corpus.toarray()


# sparse_corpus and full_corpus are matrices with sizes $N_{documents} \times V$ where $V = len(vocabulary)$

# In[ ]:


sparse_corpus


# In[ ]:


full_corpus


# 

# We want to work with words as rows, so we have to transpose the matrix.

# In[ ]:


import numpy as np
import scipy.linalg as la


# In[ ]:


full_corpus = full_corpus.T


# In[ ]:


U, s, Vt = la.svd(full_corpus)


# In[ ]:


print(U.shape, s.shape, Vt.shape)


# Now we can choose how many singular values (s) we will take to approximate an original matrix.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(16,10))
plt.plot(np.arange(1, s.shape[0] + 1), s, label="singular values")


# In[ ]:


rank_svd = 250

U_trunced = U[:, :rank_svd]
s_trunced = s[:rank_svd]
Vt_trunced = Vt[:rank_svd, :]


# In[ ]:


print(U_trunced.shape, s_trunced.shape, Vt_trunced.shape)


# In[ ]:


corpus_lsa = U_trunced.dot(np.diag(s_trunced)).dot(Vt_trunced)


# In[ ]:


corpus_lsa.shape


# In[ ]:


corpus_lsa[0]


# Here you can run experiments on word similarity measurement.

# Back to documents.

# In[ ]:


corpus_lsa = corpus_lsa.T


# In[ ]:


index_lsa_bow = similarities.MatrixSimilarity(corpus_lsa, num_features=len(corpora_dict))


# In[ ]:


search(index_lsa_bow, "healthy food", prints=True)


# In[ ]:





# ## LSI
# It is almost the same that we did in the previous section but this time we will used a built-in function.

# In[ ]:


from gensim.models import LsiModel


# In[ ]:


model_lsi = LsiModel(corpus, id2word=corpora_dict.id2token, num_topics=rank_svd)


# In[ ]:


model_lsi.print_topics(5)


# In[ ]:


for i in range(rank_svd):
    print(i, model_lsi.projection.s[i], s_trunced[i], np.allclose(model_lsi.projection.s[i], s_trunced[i]))


# In[ ]:


corpus_lsi = model_lsi[corpus]


# In[ ]:


len(corpus_lsi), len(corpus_lsi[0])


# In[ ]:


index_lsi_bow = similarities.MatrixSimilarity(corpus_lsi, num_features=len(corpora_dict))


# In[ ]:


search(index_lsi_bow, "education system", prints=True)


# In[ ]:


search(index_lsi_bow, "healthy food", prints=True)


# Can you explain why do we have zeros here?

# # Homework (10 points)

# ## Your own Dictionary (2 points)
# 
# Implement a class analogous to corpora.Dictionary.

# In[ ]:


class MyDictionary():
    def __init__(tokenized_texts):
        self.token2id = dict()
        self.id2token = dict()
        # YOUR CODE HERE    
    def doc2bow(tokenized_text):
        # YOUR CODE HERE
        return # YOUR CODE HERE


# In[ ]:


test_corpus = [['hello', 'world'], ['hello']]
my_dictionary = MyDictionary(test_corpus)
for word in {'hello', 'world'}:
    assert word in my_dictionary.token2id
    assert my_dictionary.token2id[word] = my_dictionary.id2token[my_dictionary.token2id[word]]
my_test_corpus_bow = [my_dictionary.doc2bow(text) for text in test_corpus] 
test_corpus_bow = [[(0, 1), (1, 1)], [(0, 1)]]
assert my_test_corpus_bow == test_corpus_bow


# ## Deleting stopwords (4 points)
# 
# In this task, you will clear our text corpur from stopwords and non-words like ',', '!)' etc. After that, build a new BoW and TF-IDF models. Make several queries to old and new systems and compare tre results. Did deleting stopwords really increased a quality of the search?

# In[ ]:


# You may need regular expressions to check tokens on being real 'words'
import re

# YOUR CODE HERE

clean_corpus = [] # YOUR CODE HERE


# ## Visualizing word embeddings (4 points)
# 
# Given the example of visualizing BoW vectors on a 2D-plain, plot the same graphs for TF-IDF model without stopwords. Does distributional hipothesis work here? Explain your answer.

# In[ ]:


import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook
output_notebook()

def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    if isinstance(color, str): color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig


# In[ ]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

word_vectors_pca = PCA(n_components=2, random_state=4117).fit_transform(full_corpus)  # insert TF-IDF vectors here
word_vectors_pca = preprocessing.scale(word_vectors_pca)


# In[ ]:


period = 50  # you can use 10 or 25 if it's ok for your computer

words = [corpora_dict.id2token[i] for i in range(len(corpora_dict))][::period]
draw_vectors(word_vectors_pca[:, 0][::period], word_vectors_pca[:, 1][::period], token=words)


# The other way of projecting high-dimentional data on a 2D plain is t-SNE.

# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


word_tsne = TSNE(n_components=2, verbose=100).fit_transform(full_corpus[::period])


# In[ ]:


draw_vectors(word_tsne[:, 0], word_tsne[:, 1], color='green', token=words)


# 

# # Conclusion
# 
# Tell what have you learned from this seminar.

# In[ ]:




