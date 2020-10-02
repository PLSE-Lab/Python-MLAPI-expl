#!/usr/bin/env python
# coding: utf-8

# The CORD-19 Research Challenge presents a hard task for those in the Data Sciences.  Using this Dataset is may be difficult to brainstorm how best to extract value from this Dataset for those in Biomedical Research.  The three main objectives I see many taking on in this competition may be document summarization, indexing, question-answer and possible in topic modelling to better navigate the state of research and where possible gaps exist in the literature.  
#   
# Its been a while since last I worked on Natural Language Processing (NLP) and it has been amazing to see how far the tooling and methodologies have developed.  The main aim of this notebook is to create an easy to follow structure exploring some techniques for those unfamiliar with NLP and to provide easy to use and integrate pipeline for people to extend on in other work.  
# 
# Kaggle and their partners must really be commended for compiling a data of such high quality in as short a time-frame as they have.  I have no doubt this will not just keep Data Hobbyist occupied during this time, but also provide valuable insights to researchers hoping to extract value from this corpus. 
# 
# The two main methodologies I aimed to focus on where Latent Dirichlet Allocation and Document Embedding.  Using these approaches, I aimed to explore the structure of ongoing research to get a sense of how spread research interests appear at the current moment. For visualization, I opted to explore the use of UMAP to learn a 2D embedding from these Topic Probabilities and Document Vectors to map from their original space of varying density to a lower-dimensional space of uniform density so exploratory clustering could be performed using DBSCAN.  The pipeline of using TSNE/UMAP with DBSCAN is a common pattern in Data Science pipeline as DBSCAN relies on the assumption that clusters have a similar density in order to threshold on eps.  It is also a common pattern for exploratory analysis as it a method by which to easily discover an appropriate number of cluster in the data, provided your hyperparameters are set sensible and visualize complex high dimensional data which may lie on some manifold. 

# 1. [Data](#Data)
# 2. [Topic Modelling](#Topic-Space)
# 3. [Document Vectors](#Document-Vector-Space)
# 4. [Word Vector-lebel Document Vectors](#Word-vector-level-Document-Embeddings)

# In[ ]:


get_ipython().system(' conda install -y -c conda-forge hvplot=0.5.2 bokeh=1.4.0 gensim umap-learn')


# In[ ]:


from pathlib import Path
import os
import json
from string import punctuation
import re
import warnings
import string

import numpy as np
import pandas as pd
import dask.dataframe as dd

import hvplot.pandas
import holoviews as hv
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN

from umap import UMAP
from gensim.sklearn_api import D2VTransformer, W2VTransformer

hv.extension('bokeh')


# # Data

# I opted to extract from the data paragraphs from the text. This is a common approach in handling large documents, not just for compute but also as we assume that each paragraph contains some idea which addresses a research question which a researcher may be looking for.  Splitting the articles into paragraph also prevents documents mapping to too many topics, as an article may address many research ideas and as a result may make estimation of its topic challenging.  This is also true for Word-vector-level Document Embeddings where averageing over too many words can render embeddings uninformative as they tend to the corpus mean. 

# In[ ]:


get_ipython().system(' find /kaggle/working -type f -delete')


# In[ ]:


# load texts
target_dir = Path('/kaggle/working/articles')
target_dir.mkdir()
r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
table = str.maketrans(dict.fromkeys(string.punctuation.replace('-','')))

# look though the sources
for dirname, _, filenames in os.walk('/kaggle/input'):
    # look through the journals and preprint servers
    for filename in filenames:
        try:
            file_path = Path(dirname, filename)

            # extract the json for articles
            if str(file_path).endswith('.json'):
                with open(file_path, 'r') as f:
                    loaded_json = json.loads(f.read())

                # save articles names as folder titles
                title = loaded_json['metadata']['title']
                if title == '':
                    continue
                else:
                    name = '-'.join(re.split('(?<=[\.\,\?\!\(\)\:\/])\s*', title)[0].replace(" ", "-").split()[:5])
                    name = name.split('/')[0]

                    if len(name) > 100:
                        name = name[:100]

                    name = name.translate(table)

                    article_folder = target_dir / name

                    if article_folder.exists():
                        continue
                    else:
                        article_folder.mkdir()

                        # write out the paragraphs as documents
                        for i, paragraph in enumerate(loaded_json['body_text']):
                            if paragraph['text'] == '':
                                continue
                            else:
                                paragraph_file = article_folder / f'{i}.txt'
                                with open(paragraph_file, 'w') as f:
                                    f.write(paragraph['text'])
        except OSError as e:
            break


# In[ ]:


data = load_files('/kaggle/working/articles', encoding='utf-8')


# In[ ]:


get_ipython().system(' find /kaggle/working/articles -type f -delete')


# After preprocessing we end medium-size corpus of over 230000 paragraphs for analysis, which for the sake of our kernel's 16GB memory limit we are going to subsample. 

# In[ ]:


print(f'{len(data.data)} paragraphs in corpus')


# # Topic Space

# We first model we are looking is Latent Dirichlet Allocation (LDA).  LDA is a probabilistic model which tried to model the probability a document belongs to a topic given that it contains a particular mixture of words.  While this model avoids modelling the structure of sentences, it does provide an efficient mechanism by which to model the probabilities these documents below to particular topics and has been used to great effect at internet giants, such as Yahoo.  

# In[ ]:


class Pipeline_(Pipeline):
    def topics(self, X: np.ndarray, with_final=False) -> np.ndarray:
        """
        :param X: Feature space array
        :param with_final: This will run transform on all the steps
                +           of the pipeline but the last, defaults to False
        :type with_final: bool, optional
        :return: transformed pipeline (self.regressor)
        """
        Xt = X
        for _, _, transform in self._iter(with_final=with_final):
            Xt = transform.transform(Xt)
        return Xt


# In[ ]:


subsampling = 100
X = data.data[::subsampling]


# In[ ]:


pipeline = Pipeline_([('tfidf', CountVectorizer()),
                      ('lda', LatentDirichletAllocation(50)),
                      ('umap', UMAP(n_components=2, n_neighbors=10, metric='cosine'))])
pipeline.fit(X)
Z = pipeline.transform(X)
T = pipeline.topics(X)

L = DBSCAN().fit_predict(X=Z)


# We are going to look at two plots to analyze these documents. These should overlay onto our Topic Embedding Space, learned by applying UMAP to the cosine distances between topic probablities, our DBSCAN clusters and our articles id's.  This should give some interesting insights for those running this notebook who may click through of directory and explore these articles. 

# In[ ]:


(pd.DataFrame(Z, columns=['Topic Component 1', 'Topic Component 2'])
 .assign(cluster = L)
 .sample(1000)
 .assign(cluster = lambda df: df.cluster.astype(str))
 .hvplot.scatter(x='Topic Component 1', y='Topic Component 2', c='cluster', title='CORD-19 Research Topic Clusters in Topic Embedding Space', legend=False))


# In[ ]:


(pd.DataFrame(Z, columns=['Topic Component 1', 'Topic Component 2'])
 .assign(article = np.array(data.target_names)[data.target].tolist()[::subsampling])
 .sample(1000)
 .assign(article = lambda df: df.article.astype(str))
 .hvplot.scatter(x='Topic Component 1', y='Topic Component 2', c='article', title='CORD-19 Research Articles in Topic Embedding Space', legend=False))


# # Document Vector Space

# Document Vectors are learning using negative sampling by comparing a documents vector to proposed learned word vectors which may or may not appear in its text.  This negative sampling trick is the magic behind word2vec and any2vec algorithms and makes for fast and scalable non-contextual word-vector embeddings. Using this approach, we aim to learn embeddings for our paragraph, which we may again embed into a 2D space using UMAP using the cosine angles between these vectors and cluster using DBSCAN. 

# In[ ]:


class Tokenizer(BaseEstimator, MetaEstimatorMixin):
    """Tokenize input strings based on a simple word-boundary pattern."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        ## split on word-boundary. A simple technique, yes, but mirrors what sklearn does to preprocess:
        ## https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/feature_extraction/text.py#L261-L266
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        parser = lambda doc: token_pattern.findall(doc)
        func = lambda df: df.apply(parser)
        return (pd.Series(X)
                .pipe(dd.from_pandas, npartitions=8)
                .map_partitions(func, meta=(None, 'object'))
                .compute(scheduler='processes')
                .values)


# In[ ]:


pipeline_d2v = Pipeline([
    ('tokenize', Tokenizer()),
    ('d2v', D2VTransformer(size=50, iter=100)),
    ('umap', UMAP(n_components=2, n_neighbors=15, metric='cosine', min_dist=0))
])

Z_d2v = pipeline_d2v.fit_transform(X)

L_d2v = DBSCAN().fit_predict(X=Z_d2v)


# In[ ]:


(pd.DataFrame(Z_d2v, columns=['Component 1', 'Component 2'])
 .assign(cluster = L_d2v)
 .sample(1000)
 .assign(cluster = lambda df: df.cluster.astype(str))
 .hvplot.scatter(x='Component 1', y='Component 2', c='cluster', title='CORD-19 Research Document Clusters in Document Embedding Space', legend=False))


# In[ ]:


(pd.DataFrame(Z_d2v, columns=['Component 1', 'Component 2'])
 .assign(article = np.array(data.target_names)[data.target].tolist()[::subsampling])
 .sample(1000)
 .assign(article = lambda df: df.article.astype(str))
 .hvplot.scatter(x='Component 1', y='Component 2', c='article', title='CORD-19 Research Articles in Document Embedding Space', legend=False))


# # Word-vector-level Document Embeddings

# While Document Embeddings provide a valuable extension to the word2vec algorithm, many practitioners opt to aggregate word vectors in order to represent documents.  This approach can often efficiently exploit pre-trained word-embeddings and can be highly flexible to particular weightings to these word-vectors the practitioner looks to utilize based on knowledge of the corpus. 

# In[ ]:


class W2VTransformerDocLevel(W2VTransformer):
    """Extend Gensim's Word2Vec sklearn-wrapper class to further transform word-vectors into doc-vectors by
    averaging the words in each document."""
    
    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000):
        super().__init__(size, alpha, window, min_count, max_vocab_size, sample, seed, workers, min_alpha, sg, hs, negative, cbow_mean, hashfxn, iter, null_word, trim_rule, sorted_vocab, batch_words)
    
    def transform(self, docs):      
        doc_vecs = []
        for doc in docs:
            ## for each document generate a word matrix
            word_vectors_per_doc = []
            for word in doc:
                ## handle out-of vocabulary words
                if word in self.gensim_model.wv:
                    word_vectors_per_doc.append(self.gensim_model.wv[word])

            word_vectors_per_doc = np.array(word_vectors_per_doc)
            ## take the column-wise mean of this matrix and store
            doc_vec = word_vectors_per_doc.mean(axis=0)
            
            if doc_vec.shape != (50,):
                warnings.warn('Empty vector')
                doc_vec = np.zeros((50,))
                
                
            doc_vecs.append(doc_vec)
        return np.stack(doc_vecs)


# In[ ]:


pipeline_w2v = Pipeline([
    ('tokenize', Tokenizer()),
    ('w2v', W2VTransformerDocLevel(size=50, iter=100)),
    ('umap', UMAP(n_components=2, n_neighbors=15, metric='cosine', min_dist=0))
])

Z_w2v = pipeline_w2v.fit_transform(X)

L_w2v = DBSCAN().fit_predict(X=Z_w2v)


# Again, we overlay on a UMAP latent space our Article id's and learned DBSCAN clusters. What appears interesting is the emergence of two cluster consistent between our any2vec approaches. This may be an artefact of the method or some true structure in the data. What may be interesting to investigate is whether our smaller cluster represents a true cluster in the data or whether it represent some artefact such as a disclaimer or copyright used in these articles. 

# In[ ]:


(pd.DataFrame(Z_w2v, columns=['Component 1', 'Component 2'])
 .assign(cluster = L_w2v)
 .sample(1000)
 .assign(cluster = lambda df: df.cluster.astype(str))
 .hvplot.scatter(x='Component 1', y='Component 2', c='cluster', title='CORD-19 Research Clusters in Word-vector-level Document Embedding Space', legend=False))


# In[ ]:


(pd.DataFrame(Z_w2v, columns=['Component 1', 'Component 2'])
 .assign(article = np.array(data.target_names)[data.target].tolist()[::subsampling])
 .sample(1000)
 .assign(article = lambda df: df.article.astype(str))
 .hvplot.scatter(x='Component 1', y='Component 2', c='article', title='CORD-19 Research Articles in Word-vector-level Document Embedding Space', legend=False))


# In[ ]:




