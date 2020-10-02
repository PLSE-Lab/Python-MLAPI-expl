#!/usr/bin/env python
# coding: utf-8

# **Problem statement:**
# 
# I need a way to more quickly understand what's happening in the Kaggle forums and act on it. I want a faster way to summerize trends on forums posts, figure out what questions are good for me to answer and alert my teammates if the community is reporting something en masse. 
# 
# This is actually three seperate problems:
# 
# * Summerization (level of activicty, new/emerging topics, topics that are newly popular)
# * Flag questions I'm likely to know the answer to
#     * Identify possible answerers for a given question
# * Alerts based on anomaly detection (lots of community discussion around a specific topic)
# 
# **Measuring success:**
# 
# Summerization:
# 
# * User feedback on bot output (online learning)
# * Usupervised NLU
#     * Manual verification of topics
#     * Manual verification of keywords
# 
# Flagging questions: 
# 
# * Accuracy of predicting questions I replied to using my forum post history. 
# 
# Alerts:
# 
# * Accuracy at identifying past events/bugs. 
# 
# **Possible approches to summerization:**
# 
# * **Level of activicty** 
#     * time series modelling of # of posts over time
#     * X posts this week (+- from last week), most popular (upvotes), most replied to 
#     
# 
# * Keywords
#     * https://repositorio.inesctec.pt/bitstream/123456789/7623/1/P-00N-NF5.pdf
#     * Faster
# * Topics
#     * More flexible to differences in vocabulary  
# * Hybrid Approach
#     * Keywords + embeddings to group similar keywords
# * Clustering based on embeddings
#     * Do we want to train our own embeddings?
#     * Look into current approaches 

# 

# ## Week 2, comparing unsupervised text clustering/class. methods: 
# 
# Our goal: find and group forum posts on similar topics.
# 
# Challenges: 
# * We don't have labels or tags
# * Topics/clusters will change over time
# * We don't know how many clusters/topics we have and we expect that to change over time
# * We don't have a lot of forum posts per day (~500)
# 
# Solutions: 
# * Run model weekly & apply daily (to address sparsity of data)
# * Use unsupervised methods
# * Run model continuouly/batch run (this means training time/cost is more important to consider)
# 
# ____
# 
# Unsupervised text clustering:
# 
# Words -> inputs:
# * Embeddings (might be a problem if we don't train new embeddings for each time we run the model)
#     * Fasttext can handle out of vocabulary words
#     * Subword embeddings
#     * Biggest factor: how long do they take to train?
#     * Universal Sentence Encoder Embeddings??
# * Td-idf
# * LDA
# * "Take the term-frequency matrix, remove the "expected" frequency (by subtracting, or using the column marginal as a noise model)" -Leland McInnes
# * Embeddings weighted with tf-idf
# * Embeddings -> PCA, remove first principle component -  Arora (2018) 'A simple but tough to beat baseline for sentence embeddings'
# * pLSA ("cheaper" version of LDA)
# 
# Topic modeling:
# * LDA 
#     * Too slow, not great for us
#     * Hard to interpret
#     
# Clustering (with embeddings):
# * Hierach. clustering 
# * Brown clusters
#     * hierarhical 
#     * work on the word level
#     * can be updated actively
#     * would need to find Python code
# * DBSCAN
#     * needs embeddings for input
#     * should reduce dimensionality
#     * note: clusters should be of similar densities
#     * HDBSCAN is hierarchical
# 
# Keywords:
# * Unsupervised keyword extraction (YAKE)
# 
# ____
# 
# Whole pipeline: 
# * https://topsbm.github.io/
# * https://github.com/bigartm/bigartm 
# 
# 1. Words to numbers
#     * tfidf
#     * LDA
#     * pLSA
#     * Embeddings
#         * fasttext
#         * USE embeddings
#         * Glove?
#         * word2vec
#         * elmo 
# 2. Dimensionality Reduction
#     * UMAP
#     * PCA
# 3. Clustering
#     * DBSCAN
#     * HDBSCAN
#     * Spectral clustering

# In[ ]:




