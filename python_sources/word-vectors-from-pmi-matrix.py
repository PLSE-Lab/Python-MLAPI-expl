#!/usr/bin/env python
# coding: utf-8

# # Word Vectors from Decomposing a Word-Word Pointwise Mutual Information Matrix
# Lets create some simple [word vectors](https://en.wikipedia.org/wiki/Word_embedding) by applying a [singular value decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition) to a [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) word-word matrix.  There are many other ways to create word vectors, but matrix decomposition is one of the most straightforward.  A well cited description of the technique used in this notebook can be found in Chris Moody's blog post [Stop Using word2vec](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/).    If you are interested in reading further about the history of word embeddings and a discussion of modern approaches check out the following blog post by Sebastian Ruder, [An overview of word embeddings and their connection to distributional semantic models](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/).  Especially interesting to me is the work by Omar Levy, Yoav Goldberg, and Ido Dagan which shows that tuning hyperparameters is as (if not more) important as the algorithm chosen to build word vectors. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://transacl.org/ojs/index.php/tacl/article/view/570).
# 
# We will be using the ["A Million News Headlines"](https://www.kaggle.com/therohk/million-headlines) dataset which contains headlines published over a period of 15 years from the Australian Broadcasting Corporation (ABC).
# It is a great clean corpus that is large enough to be interesting and small enough to allow for quick calculations.  In this notebook tutorial we will implement as much as we can without using libraries that obfuscate the algorithm.  We're not going to write our own linear algebra or sparse matrix routines, but we will calculate unigram frequency, skipgram frequency, and the pointwise mutual information matrix "by hand".  Hopefully this will make the method easier to understand!   

# In[ ]:


from collections import Counter
import itertools

import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg 
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# # Read Data and Preview

# In[ ]:


df = pd.read_csv('../input/abcnews-date-text.csv')
df.head()


# # Minimal Preprocessing
# We're going to create a word-word co-occurrence matrix from the text in the headlines.  We will define two words as "co-occurring" if they appear in the same headline.  Using this definition, single word headlines are not interestintg for us.  Lets remove them as well as a common set of english stopwords.  

# In[ ]:


headlines = df['headline_text'].tolist()
# remove stopwords
stopwords_set = set(stopwords.words('english'))
headlines = [
    [tok for tok in headline.split() if tok not in stopwords_set] for headline in headlines
]
# remove single word headlines
headlines = [hl for hl in headlines if len(hl) > 1]
# show results
headlines[0:20]


# # Unigrams
# Now lets calculate a unigram vocabulary.  The following code assigns a unique ID to each token, stores that mapping in two dictionaries (`tok2indx` and `indx2tok`), and counts how often each token appears in the corpus. 

# In[ ]:


tok2indx = dict()
unigram_counts = Counter()
for ii, headline in enumerate(headlines):
    if ii % 200000 == 0:
        print(f'finished {ii/len(headlines):.2%} of headlines')
    for token in headline:
        unigram_counts[token] += 1
        if token not in tok2indx:
            tok2indx[token] = len(tok2indx)
indx2tok = {indx:tok for tok,indx in tok2indx.items()}
print('done')
print('vocabulary size: {}'.format(len(unigram_counts)))
print('most common: {}'.format(unigram_counts.most_common(10)))


# # Skipgrams
# Now lets calculate a skipgram vocabulary.  We will loop through each word in a headline (the focus word) and then form skipgrams by examing `back_window` words behind and `front_window` words in front of the focus word (the context words).  As an example, the first sentence (after preprocessing removes the stopword `against`) ,
# ```
# aba decides community broadcasting licence
# ```
# would produce the following skipgrams with `back_window`=`front_window`=`2`, 
# ```
# ('aba', 'decides')
# ('aba', 'community')
# ('decides', 'aba')
# ('decides', 'community')
# ('decides', 'broadcasting')
# ('community', 'aba')
# ('community', 'decides')
# ('community', 'broadcasting')
# ('community', 'licence')
# ('broadcasting', 'decides')
# ('broadcasting', 'community')
# ('broadcasting', 'licence')
# ('licence', 'community')
# ('licence', 'broadcasting')
# ```

# In[ ]:


# note add dynammic window hyperparameter
back_window = 2
front_window = 2
skipgram_counts = Counter()
for iheadline, headline in enumerate(headlines):
    for ifw, fw in enumerate(headline):
        icw_min = max(0, ifw - back_window)
        icw_max = min(len(headline) - 1, ifw + front_window)
        icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
        for icw in icws:
            skipgram = (headline[ifw], headline[icw])
            skipgram_counts[skipgram] += 1    
    if iheadline % 200000 == 0:
        print(f'finished {iheadline/len(headlines):.2%} of headlines')
        
print('done')
print('number of skipgrams: {}'.format(len(skipgram_counts)))
print('most common: {}'.format(skipgram_counts.most_common(10)))


# # Sparse Matrices
# 
# We will calculate several matrices that store word-word information.  These matrices will be $N \times N$ where $N \approx 100,000$ is the size of our vocabulary.  We will need to use a sparse format so that it will fit into memory.  A nice implementation is available in [scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).  To create these sparse matrices we create three iterables that store row indices, column indices, and data values. 

# # Word-Word Count Matrix
# Our very first word vectors will come from a word-word count matrix.  We can (equivalently) take the word vectors to be the rows or columns.  

# In[ ]:


row_indxs = []
col_indxs = []
dat_values = []
ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    if ii % 1000000 == 0:
        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
    tok1_indx = tok2indx[tok1]
    tok2_indx = tok2indx[tok2]
        
    row_indxs.append(tok1_indx)
    col_indxs.append(tok2_indx)
    dat_values.append(sg_count)
    
wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
print('done')


# # Word Similarity with Sparse Count Matrices

# In[ ]:


def ww_sim(word, mat, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores

print('done')


# In[ ]:


ww_sim('strike', wwcnt_mat)


# In[ ]:


wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)
print('done')


# In[ ]:


ww_sim('strike', wwcnt_norm_mat)


# # Pointwise Mutual Information Matrices
# The pointwise mutual information (PMI) for a (word, context) pair in our corpus is defined as the probability of their co-occurrence divided by the probabilities of them appearing individually, 
# <br/><br/>
# $$
# \large {\rm pmi}(w, c) = \log \frac{p(w, c)}{p(w) p(c)}
# $$
# <br/>
# $$  \large p(w_i,c_j) = \frac{\#(w_i,c_j)}{ \sum\limits_{k}^{N}\sum\limits_{l}^{N} \#(w_k,c_l) }$$
# <br/>
# $$ \large p(w_i) = \frac{\sum\limits_{i}^{N}\#(w_i)}{ \sum\limits_{k}^{N}\sum\limits_{l}^{N} \#(w_k,c_l) }$$
# <br/>
# $$ \large p(c_j) = \frac{\sum\limits_{j}^{N}\#(c_j)}{ \sum\limits_{k}^{N}\sum\limits_{l}^{N} \#(w_k,c_l) }$$
# <br/>
# where $\large \#(w_k,c_l)$ is the word-word count matrix we defined above.
# In addition we can define the positive pointwise mutual information as, 
# <br/><br/>
# $$
# {\large \rm ppmi}(w, c) = {\rm max}\left[{\rm pmi(w,c)}, 0 \right]
# $$
# 
# Note that the definition of PMI above implies that $\large {\rm pmi}(w, c) = {\rm pmi}(w, c)$ and so this matrix will be symmetric.  

# In[ ]:


num_skipgrams = wwcnt_mat.sum()
assert(sum(skipgram_counts.values())==num_skipgrams)

# for creating sparse matrices
row_indxs = []
col_indxs = []

# pmi: pointwise mutual information
pmi_dat_values = []
# ppmi: positive pointwise mutual information
ppmi_dat_values = []
# spmi: smoothed pointwise mutual information
spmi_dat_values = []
# sppmi: smoothed positive pointwise mutual information
sppmi_dat_values = []

# Sum over words and contexts
sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

# Smoothing
# According to [Levy, Goldberg & Dagan, 2015], the smoothing operation 
# should be done on the context 
alpha = 0.75
nca_denom = np.sum(sum_over_contexts**alpha)
# sum_over_words_alpha = sum_over_words**alpha
sum_over_contexts_alpha = sum_over_contexts**alpha

ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    if ii % 1000000 == 0:
        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
    tok1_indx = tok2indx[tok1]
    tok2_indx = tok2indx[tok2]
    
    nwc = sg_count
    Pwc = nwc / num_skipgrams

    nw = sum_over_contexts[tok1_indx]
    Pw = nw / num_skipgrams
    
    nc = sum_over_words[tok2_indx]
    Pc = nc / num_skipgrams
    
    pmi = np.log2(Pwc/(Pw*Pc))
    ppmi = max(pmi, 0)
    
#   nca = sum_over_words_alpha[tok2_indx]
    nca = sum_over_contexts_alpha[tok2_indx]
    Pca = nca / nca_denom

    spmi = np.log2(Pwc/(Pw*Pca))
    sppmi = max(spmi, 0)
    
    row_indxs.append(tok1_indx)
    col_indxs.append(tok2_indx)
    pmi_dat_values.append(pmi)
    ppmi_dat_values.append(ppmi)
    spmi_dat_values.append(spmi)
    sppmi_dat_values.append(sppmi)
        
pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))

print('done')


# # Word Similarity with Sparse PMI Matrices

# In[ ]:


ww_sim('strike', pmi_mat)


# In[ ]:


ww_sim('strike', ppmi_mat)


# In[ ]:


ww_sim('strike', spmi_mat)


# In[ ]:


ww_sim('strike', sppmi_mat)


# # Singular Value Decomposition
# With the PMI and PPMI matrices in hand, we can apply a singular value decomposition to create dense word vectors from the sparse ones we've been using. 

# In[ ]:


pmi_use = ppmi_mat
embedding_size = 50
uu, ss, vv = linalg.svds(pmi_use, embedding_size) 

print('done')


# In[ ]:


print('vocab size: {}'.format(len(unigram_counts)))
print('embedding size: {}'.format(embedding_size))
print('uu.shape: {}'.format(uu.shape))
print('ss.shape: {}'.format(ss.shape))
print('vv.shape: {}'.format(vv.shape))


# In[ ]:


unorm = uu / np.sqrt(np.sum(uu*uu, axis=1, keepdims=True))
vnorm = vv / np.sqrt(np.sum(vv*vv, axis=0, keepdims=True))
#word_vecs = unorm
#word_vecs = vnorm.T
word_vecs = uu + vv.T
word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs, axis=1, keepdims=True))

print('done')


# In[ ]:


def word_sim_report(word, sim_mat):
    sim_word_scores = ww_sim(word, word_vecs)
    for sim_word, sim_score in sim_word_scores:
        print(sim_word, sim_score)
        word_headlines = [hl for hl in headlines if sim_word in hl and word in hl][0:5]
        for headline in word_headlines:
            print(f'    {headline}')
            
print('done')


# In[ ]:


word = 'strike'
word_sim_report(word, word_vecs)


# In[ ]:


word = 'war'
word_sim_report(word, word_vecs)


# In[ ]:


word = 'bank'
word_sim_report(word, word_vecs)


# In[ ]:


word = 'car'
word_sim_report(word, word_vecs)


# In[ ]:


word = 'football'
word_sim_report(word, word_vecs)


# In[ ]:


word = 'tech'
word_sim_report(word, word_vecs)


# # Scratch Pad

# In[ ]:


# check a few things
alpha = 0.75
dsum = wwcnt_mat.sum()
nwc = skipgram_counts[(tok1, tok2)]
Pwc = nwc / dsum

indx1 = tok2indx[tok1]
indx2 = tok2indx[tok2]

nw = wwcnt_mat[indx1, :].sum()
Pw = nw / dsum
nc = wwcnt_mat[:, indx2].sum()
Pc = nc / dsum

nca = nc**alpha
nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)
Pca = nca / nca_denom

print('dsum=', dsum)
print('Pwc=', Pwc)
print('Pw=', Pw)
print('Pc=', Pc)
print('Pca=', Pca)
pmi1 = Pwc / (Pw * Pc)
pmi1a = Pwc / (Pw * Pca)
pmi2 = (nwc * dsum) / (nw * nc)

print('pmi1=', pmi1)
print('pmi1a=', pmi1a)
print('pmi2=', pmi2)


# In[ ]:




