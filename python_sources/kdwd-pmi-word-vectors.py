#!/usr/bin/env python
# coding: utf-8

# A few years ago I wrote a [kernel exploring word vectors](https://www.kaggle.com/gabrielaltay/word-vectors-from-pmi-matrix) calculated from Pointwise Mutual Information.  This is a reboot of that kernel using the Kensho Derived Wikimedia Dataset. This new version includes a dynamic context window, context distribution smoothing, and eigenvalue weighting.  

# # Kensho Derived Wikimedia Dataset - Word Vectors from Decomposing a Word-Word Pointwise Mutual Information Matrix
# Lets create some simple [word vectors](https://en.wikipedia.org/wiki/Word_embedding) by applying a [singular value decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition) to a [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) word-word matrix.  There are many other ways to create word vectors, but matrix decomposition is one of the most straightforward.  A well cited description of the technique used in this notebook can be found in Chris Moody's blog post [Stop Using word2vec](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/).    If you are interested in reading further about the history of word embeddings and a discussion of modern approaches check out the following blog post by Sebastian Ruder, [An overview of word embeddings and their connection to distributional semantic models](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/).  Especially interesting to me is the work by Omar Levy, Yoav Goldberg, and Ido Dagan which shows that tuning hyperparameters is as (if not more) important as the algorithm chosen to build word vectors. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://transacl.org/ojs/index.php/tacl/article/view/570) (LGD15). 
# 
# We will be using the [Kensho Derived Wikimedia Dataset](https://www.kaggle.com/kenshoresearch/kensho-derived-wikimedia-data) which contains the text of English Wikipedia from 2019-12-01. In this notebook tutorial we will implement as much as we can without using libraries that obfuscate the algorithm.  We're not going to write our own linear algebra or sparse matrix routines, but we will calculate unigram frequency, skipgram frequency, and the pointwise mutual information matrix "by hand".  We will also use the notation from LGD15 so you can follow along using that paper. Hopefully this will make the method easier to understand!   

# In[ ]:


from collections import Counter
import json
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


# In[ ]:


sns.set()
sns.set_context('talk')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


NUM_KLAT_LINES = 5_343_564
MIN_UNIGRAM_COUNT = 100
WINDOW = 5
MAX_PAGES = 100_000
INTROS_ONLY = True
kdwd_path = os.path.join("/kaggle", "input", "kensho-derived-wikimedia-data")


# # Read Data and Preview
# Lets start with a class to read the Wikipedia text.  We'll give ourselves the option to only use *Introduction* sections and to limit the number of pages.

# In[ ]:


def tokenizer(text):
    return text.strip().lower().split()


# In[ ]:


class KdwdLinkAnnotatedText:
    
    def __init__(self, file_path, intros_only=False, max_pages=1_000_000_000):
        self._file_path = file_path
        self._intros_only = intros_only
        self._max_pages = max_pages
        self._num_lines = NUM_KLAT_LINES
        self.pages_to_parse = min(self._num_lines, self._max_pages)
        
    def __iter__(self):
        with open(self._file_path) as fp:
            for ii, line in enumerate(fp):
                page = json.loads(line)
                for section in page['sections']:
                    yield section['text']
                    if self._intros_only:
                        break
                if ii + 1 >= self.pages_to_parse:
                    break


# In[ ]:


file_path = os.path.join(kdwd_path, "link_annotated_text.jsonl")
klat_intros_2 = KdwdLinkAnnotatedText(file_path, intros_only=True, max_pages=2)
klat_intros = KdwdLinkAnnotatedText(file_path, intros_only=INTROS_ONLY, max_pages=MAX_PAGES)


# In[ ]:


two_intros = [intro for intro in klat_intros_2]


# In[ ]:


two_intros


# # Unigrams
# Now lets calculate a unigram vocabulary. The following code assigns a unique ID to each token, stores that mapping in two dictionaries (`tok2indx` and `indx2tok`), and counts how often each token appears in the corpus.

# In[ ]:


def filter_unigrams(unigrams, min_unigram_count):
    tokens_to_drop = [
        token for token, count in unigrams.items() 
        if count < min_unigram_count]                                                                                 
    for token in tokens_to_drop:                                                             
        del unigrams[token]
    return unigrams


# In[ ]:


def get_unigrams(klat):
    unigrams = Counter()
    for text in tqdm(
        klat, total=klat.pages_to_parse, desc='calculating unigrams'
    ):
        tokens = tokenizer(text)
        unigrams.update(tokens)
    return unigrams


# In[ ]:


unigrams = get_unigrams(klat_intros)
print("token count: {}".format(sum(unigrams.values())))                          
print("vocabulary size: {}".format(len(unigrams)))


# In[ ]:


unigrams = filter_unigrams(unigrams, MIN_UNIGRAM_COUNT)
print("token count: {}".format(sum(unigrams.values())))                          
print("vocabulary size: {}".format(len(unigrams))) 


# In[ ]:


tok2indx = {tok: indx for indx, tok in enumerate(unigrams.keys())}
indx2tok = {indx: tok for tok, indx in tok2indx.items()}


# # Skipgrams
# Now lets calculate word-context pairs (i.e., skipgrams).  We will loop through each token in a section (the "word") and then use a `word2vec` style dynamic window to sample a context token to form skipgrams.

# In[ ]:


def get_skipgrams(klat, max_window, tok2indx, seed=938476):
    rnd = random.Random()
    rnd.seed(a=seed)
    skipgrams = Counter()
    for text in tqdm(
        klat, total=klat.pages_to_parse, desc='calculating skipgrams'
    ):
        
        tokens = tokenizer(text)
        vocab_indices = [tok2indx[tok] for tok in tokens if tok in tok2indx]
        num_tokens = len(vocab_indices)
        if num_tokens == 1:
            continue
        for ii_word, word in enumerate(vocab_indices):
            
            window = rnd.randint(1, max_window)
            ii_context_min = max(0, ii_word - window)
            ii_context_max = min(num_tokens - 1, ii_word + window)
            ii_contexts = [
                ii for ii in range(ii_context_min, ii_context_max + 1) 
                if ii != ii_word]
            for ii_context in ii_contexts:
                context = vocab_indices[ii_context]
                skipgram = (word, context)
                skipgrams[skipgram] += 1 

    return skipgrams


# In[ ]:


skipgrams = get_skipgrams(klat_intros, WINDOW, tok2indx)
print("number of unique skipgrams: {}".format(len(skipgrams)))
print("number of skipgrams: {}".format(sum(skipgrams.values())))
most_common = [
    (indx2tok[sg[0][0]], indx2tok[sg[0][1]], sg[1]) 
    for sg in skipgrams.most_common(25)]
print('most common: {}'.format(most_common))


# # Sparse Matrices
# We will calculate several matrices that store word-word information. These matrices will be $N \times N$  where $N \approx 100,000$  is the size of our vocabulary. We will need to use a sparse format so that it will fit into memory. A nice implementation is available in [scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html). To create these sparse matrices we create three iterables that store row indices, column indices, and data values.

# # Word-Word Count Matrix
# Our very first word vectors will come from a word-word count matrix. This matrix is symmetric so we can (equivalently) take the word vectors to be the rows or columns. However we will try and code as if the rows are word vectors and the columns are context vectors.

# In[ ]:


def get_count_matrix(skipgrams, tok2indx):
    row_indxs = []                                                                       
    col_indxs = []                                                                       
    dat_values = []                                                                      
    for skipgram in tqdm(
        skipgrams.items(), 
        total=len(skipgrams), 
        desc='building count matrix row,col,dat'
    ):
        (tok_word_indx, tok_context_indx), sg_count = skipgram
        row_indxs.append(tok_word_indx)
        col_indxs.append(tok_context_indx)
        dat_values.append(sg_count)
    print('building sparse count matrix')
    return sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))


# In[ ]:


count_matrix = get_count_matrix(skipgrams, tok2indx)


# In[ ]:


# normalize rows
count_matrix_l2 = normalize(count_matrix, norm='l2', axis=1)


# In[ ]:


# demonstrate normalization
irow=10
row = count_matrix_l2.getrow(irow).toarray().flatten()
print(np.sqrt((row*row).sum()))

row = count_matrix.getrow(irow).toarray().flatten()
print(np.sqrt((row*row).sum()))


# In[ ]:


xx1 = count_matrix.data
xx2 = count_matrix_l2.data
nbins = 30

fig, axes = plt.subplots(1, 2, figsize=(18,8))

ax = axes[0]
counts, bins, patches = ax.hist(xx1, bins=nbins, density=True, log=True)
ax.set_xlabel('count_matrix')
ax.set_ylabel('fraction')

ax = axes[1]
counts, bins, patches = ax.hist(xx2, bins=nbins, density=True, log=True)
ax.set_xlabel('count_matrix_l2')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(1e-4, 5e1)

fig.suptitle('Distribution of Embedding Matrix Values');


# # Word Similarity with Sparse Count Matrices

# In[ ]:


def ww_sim(word, mat, tok2indx, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    #dists = cosine_distances(mat, v1).flatten()
    dists = euclidean_distances(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores


# In[ ]:


word = 'city'


# In[ ]:


ww_sim(word, count_matrix, tok2indx)


# In[ ]:


ww_sim(word, count_matrix_l2, tok2indx)


# Note that the normalized vectors will produce the same similarities as the un-normalized vectors as long as we are using cosine similarity. 

# # Pointwise Mutual Information Matrices
# 
# ## Definitions
# 
# $$
# \begin{align}
# PMI(w, c) = 
# \log \frac
# {\hat{P}(w,c)}
# {\hat{P}(w)\hat{P}(c)} =
# \log \frac
# {\#(w,c) \, \cdot \lvert D \rvert}
# {\#(w) \cdot \#(c)}
# \\
# \\
# PPMI(w, c) = {\rm max} \left[ PMI(w, c), 0 \right]
# \\
# \\
# \#(w) = \sum_{c^{\prime}} \#(w, c^{\prime}),
# \quad
# \#(c) = \sum_{w^{\prime}} \#(w^{\prime}, c)
# \end{align}
# $$
# 
# 
# ## Context Distribution Smoothing
# 
# $$
# \begin{align}
# PMI_{\alpha}(w, c) = 
# \log \frac
# {\hat{P}(w,c)}
# {\hat{P}(w)\hat{P}_{\alpha}(c)}
# \\
# \\
# \hat{P}_{\alpha}(c) = 
# \frac
# {\#(c)^{\alpha}}
# {\sum_c \#(c)^{\alpha}}
# \end{align}
# $$
# 

# Lets explain how these equations relate to our variables. LGD15 use $\#(w,c)$ to denote the number of times a word-context pair appears in the corpus. We first calculated these numbers in our `skipgrams` variable and then stored them in `count_matrix`. The rows in `count_matrix` represent words and the columns represent contexts. Given a word token and a context token we can look up their indices in `tok2indx` and access the count via `skipgrams` or `count_matrix`

# In[ ]:


word = "the"
context = "of"
word_indx = tok2indx[word]
context_indx = tok2indx[context]
print('pound_wc for ({},{}) from skipgrams: {}'.format(
    word, context, skipgrams[(word_indx, context_indx)]))
print('pound_wc for ({},{}) from count_matrix: {}'.format(
    word, context, count_matrix[word_indx, context_indx]))


# LGD15 use $\#(w)$ to denote the number of times a word appears anywhere in the corpus and $\#(c)$ to denote the number of times a context appears anywhere in the corpus.  We can calculate $\#(w)$ by summing over the columns of `count_matrix` and $\#(c)$ by summing over the rows in `count_matrix`.

# In[ ]:


sum_over_words = np.array(count_matrix.sum(axis=0)).flatten()    # sum over rows
sum_over_contexts = np.array(count_matrix.sum(axis=1)).flatten() # sum over columns

pound_w_check1 = count_matrix.getrow(word_indx).sum()
pound_w_check2 = sum_over_contexts[word_indx]
print('pound_w for "{}" from getrow then sum: {}'.format(word, pound_w_check1))
print('pound_w for "{}" from sum_over_contexts: {}'.format(word, pound_w_check2))

pound_c_check1 = count_matrix.getcol(context_indx).sum()
pound_c_check2 = sum_over_words[context_indx]
print('pound_c for "{}" from getcol then sum: {}'.format(context, pound_c_check1))
print('pound_c for "{}" from sum_over_words: {}'.format(context, pound_c_check2))


# In[ ]:


def get_ppmi_matrix(skipgrams, count_matrix, tok2indx, alpha=0.75):
    
    # for standard PPMI
    DD = sum(skipgrams.values())
    sum_over_contexts = np.array(count_matrix.sum(axis=1)).flatten()
    sum_over_words = np.array(count_matrix.sum(axis=0)).flatten()
        
    # for context distribution smoothing (cds)
    sum_over_words_alpha = sum_over_words**alpha
    Pc_alpha_denom = np.sum(sum_over_words_alpha)
        
    row_indxs = []
    col_indxs = []
    ppmi_dat_values = []   # positive pointwise mutual information
    
    for skipgram in tqdm(
        skipgrams.items(), 
        total=len(skipgrams), 
        desc='building ppmi matrix row,col,dat'
    ):
        
        (tok_word_indx, tok_context_indx), pound_wc = skipgram
        pound_w = sum_over_contexts[tok_word_indx]
        pound_c = sum_over_words[tok_context_indx]
        pound_c_alpha = sum_over_words_alpha[tok_context_indx]

        Pwc = pound_wc / DD
        Pw = pound_w / DD
        Pc = pound_c / DD
        Pc_alpha = pound_c_alpha / Pc_alpha_denom

        pmi = np.log2(Pwc / (Pw * Pc_alpha))
        ppmi = max(pmi, 0)
        
        row_indxs.append(tok_word_indx)
        col_indxs.append(tok_context_indx)
        ppmi_dat_values.append(ppmi)

    print('building ppmi matrix')    
    return sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))


# In[ ]:


ppmi_matrix = get_ppmi_matrix(skipgrams, count_matrix, tok2indx)


# In[ ]:


word = 'city'
ww_sim(word, ppmi_matrix, tok2indx)


# # Singular Value Decomposition
# With the PPMI matrix in hand, we can apply a singular value decomposition (SVD) to create dense word vectors from the sparse ones we've been using.  SVD factorizes a matrix $M$ into a product of matrices $M = U \cdot \Sigma \cdot V^T$ where $U$ and $V$ are orthonormal and $\Sigma$ is a diagonal matrix of eigenvalues. By keeping the top $d$ elements of $\Sigma$, we obtain $M_d = U_d \cdot \Sigma_d \cdot V_d^T$. 
# 
# Word and context vectors are typically represented by:
# 
# $$
# W = U_d \cdot \Sigma_d, \quad C = V_d
# $$
# 
# It has been shown empirically that weighting the eigenvalue matrix can effect performance.  
# 
# $$
# W = U_d \cdot \Sigma_d^p
# $$
# 
# LGD15 suggest always using this weighting but that $p$ should be tuned to the task.  They investigate values of $p=0.5$ and $p=0$ (with $p=1$ corresponding to the traditional case).  Lets try $p=0.5$. 

# In[ ]:


embedding_size = 200
uu, ss, vv = linalg.svds(ppmi_matrix, embedding_size)


# In[ ]:


print('vocab size: {}'.format(len(unigrams)))
print('ppmi size: {}'.format(ppmi_matrix.shape))
print('embedding size: {}'.format(embedding_size))
print('uu.shape: {}'.format(uu.shape))
print('ss.shape: {}'.format(ss.shape))
print('vv.shape: {}'.format(vv.shape))


# Lets check that dot-products between rows of $M_d$ are equal to dot-products between rows of $W$ where, 
# 
# $$
# M_d = U_d \cdot \Sigma_d \cdot V_d^T, \quad W = U_d \cdot \Sigma_d
# $$

# In[ ]:


# Dont do this for full run or we'll run out of RAM

#x = (uu.dot(np.diag(ss)).dot(vv))[word_indx, :]
#y = (uu.dot(np.diag(ss)).dot(vv))[context_indx, :]
#print((x * y).sum())

#x = (uu.dot(np.diag(ss)))[word_indx, :]
#y = (uu.dot(np.diag(ss)))[context_indx, :]
#print((x * y).sum())


# Now lets create our final embeddings.

# In[ ]:


p = 0.5
svd_word_vecs = uu.dot(np.diag(ss**p))
print(svd_word_vecs.shape)


# In[ ]:


nbins = 20
fig, axes = plt.subplots(2, 2, figsize=(16,14), sharey=False)

ax = axes[0,0]
xx = count_matrix.data
ax.hist(xx, bins=nbins, density=True, log=True)
ax.set_xlabel('word_counts')
ax.set_ylabel('fraction')

ax = axes[0,1]
xx = count_matrix_l2.data
ax.hist(xx, bins=nbins, density=True, log=True)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel('word_counts_l2')

ax = axes[1,0]
xx = ppmi_matrix.data
ax.hist(xx, bins=nbins, density=True, log=True)
ax.set_xlabel('PPMI')
ax.set_ylabel('fraction')

ax = axes[1,1]
xx = svd_word_vecs.flatten()
ax.hist(xx, bins=nbins, density=True, log=True)
ax.set_xlabel('SVD(p=0.5)-PPMI')

fig.suptitle('Distribution of Embedding Matrix Values');


# In[ ]:


word = 'car'
sims = ww_sim(word, svd_word_vecs, tok2indx)
for sim in sims:
    print('  ', sim)


# In[ ]:


word = 'king'
sims = ww_sim(word, svd_word_vecs, tok2indx)
for sim in sims:
    print('  ', sim)


# In[ ]:


word = 'queen'
sims = ww_sim(word, svd_word_vecs, tok2indx)
for sim in sims:
    print('  ', sim)


# In[ ]:


word = 'news'
sims = ww_sim(word, svd_word_vecs, tok2indx)
for sim in sims:
    print('  ', sim)


# In[ ]:


word = 'hot'
sims = ww_sim(word, svd_word_vecs, tok2indx)
for sim in sims:
    print('  ', sim)


# In[ ]:


svd_word_vecs.shape


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


svd_2d = TSNE(n_components=2, random_state=3847).fit_transform(svd_word_vecs)


# In[ ]:


svd_2d


# In[ ]:


word='city'
size = 3
indx = tok2indx[word]
cen_vec = svd_2d[indx,:]
dxdy = np.abs(svd_2d - cen_vec) 
bmask = (dxdy[:,0] < size) & (dxdy[:,1] < size)
sub = svd_2d[bmask]


fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(sub[:,0], sub[:,1])
ax.set_xlim(cen_vec[0] - size, cen_vec[0] + size)
ax.set_ylim(cen_vec[1] - size, cen_vec[1] + size)
for ii in range(len(indx2tok)):
    if not bmask[ii]:
        continue
    plt.annotate(
        indx2tok[ii],
        xy=(svd_2d[ii,0], svd_2d[ii,1]),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')


# In[ ]:




