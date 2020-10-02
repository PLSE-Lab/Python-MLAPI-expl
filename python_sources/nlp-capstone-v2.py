#!/usr/bin/env python
# coding: utf-8

# # Grouping Similar Web Novels
# In this unsupervised learning project, I will create a model that groups novels that are similar together.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import sklearn
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import os
print(os.listdir("../input"))


# Stopwords are a list of the most common words that may not be useful in determining whether it belongs to one group or another. Words such as *and, else, her, or, their*.
# Stemmer is used to reduce words to their basic word. An example would be *fighting* would be reduced to *fight*

# In[ ]:


stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")


# In[ ]:


# Utility functions

# takes raw data, converts to sentences, and removes tokens that aren't words or digits
def clean_sentences(text):
    tokens = [sent for sent in nltk.sent_tokenize(text)]

    sent_list = []
    for sent in tokens:
        sent_str = ''
        for i, word in enumerate(nltk.word_tokenize(sent)):
            # nltk doesn't handle apostrophes correctly
            if word[0] == "'":
                sent_str = sent_str[:-1]
            
            # only adds words and digits
            if re.search('[a-zA-Z0-9]', word):
                sent_str += word.lower() + ' '
        sent_list.append(sent_str.strip())

    return sent_list

# takes list of clean sentences and converts to list of tokens
def tokens_only(text):
    tokens = []
    
    for sentence in text:
        tokens.extend(sentence.split(" "))
    
    return tokens

# takes in text, cleans it, and returns lemma only
def lemma_tokens(text):
    tokens = tokens_only(clean_sentences(text))
    
    return [stemmer.stem(token) for token in tokens]


# In[ ]:


# loading files
filenames = ['against_the_gods', 'battle_through_the_heavens', 'desolate_era', 'emperors_domination', 'martial_god_asura', 'martial_world', 'overgeared', 'praise_the_orc', 'sovereign_of_the_three_realms', 'wu_dong_qian_kun']
raw_files = []

for filename in filenames:
    with open('../input/' + filename + '.txt', encoding='utf-8') as myfile:
        raw_files.append(myfile.read())


# In[ ]:


# use extend so it's a big flat list of vocab
all_lemma = []
all_tokens = []
all_sentences = []
all_sentences_label = []

for i, doc in enumerate(raw_files):
    
    # clean sentences    
    tmp_list= clean_sentences(doc)
    all_sentences.extend(tmp_list)
    for j in range(len(tmp_list)):
        all_sentences_label.append(filenames[i])
    
    # convert list of clean sentences to tokens
    tmp_list = tokens_only(tmp_list)
    all_tokens.extend(tmp_list)
    
    # gets root word for tokens in document
    all_lemma.extend(lemma_tokens(doc))


# After creating a list of tokens and lemma, I put them into a dataframe to easily reference them again later 

# In[ ]:


vocab_df = pd.DataFrame({'words': all_tokens}, index = all_lemma)
print ('there are', vocab_df.shape[0], 'words and', len(all_sentences), 'sentences')


# In[ ]:


vocab_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorizer = TfidfVectorizer(max_df=0.75, # drop words that occur in more than 3/4 of the sentence
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False,
                             use_idf=True, # use inverse document frequencies in our weighting
                             norm=u'l2', # Apply a correction factor so that longer sentences and shorter sentences get treated equally
                             smooth_idf=True, # Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                             tokenizer=lemma_tokens,
                             ngram_range=(1,3)                             
                            )

# Apply the vectorizer
tfidf_matrix = vectorizer.fit_transform(raw_files)
print(tfidf_matrix.shape)


# In[ ]:


terms = vectorizer.get_feature_names()


# # Clustering
# Using the data available, it will be run through four different clustering methods to see which method best represents the data. Those methods are KMeans, Mean Shift, Spectral Clustering, and Affinity Propagation

# In[ ]:


from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, SpectralClustering, AffinityPropagation

num_clusters = 5


# ## K Means
# This clustering method splits the tokens into *k* clusters, with each token belonging to the cluster with the nearest mean.

# In[ ]:


km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()


# In[ ]:


novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }
df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])


# In[ ]:


print("Clusters:")
print('-'*40)

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    if i != 0:
        print('\n')
    print("Cluster %d words: " % i, end='')
    
    for j, ind in enumerate(order_centroids[i, :10]):
        if (j == 0):
            print('%s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
        else:
            print(', %s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
    print()
    
    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')


# ## Mean Shift
# Mean Shift clustering finds vectors that are most similar to the data points. It then shifts the data points onto those vectors, which determines how many clusters are available. The number of clusters is not a parameter. Instead, Mean Shift will let us know how many clusters it has found.

# In[ ]:


ms = MeanShift()
ms.fit(tfidf_matrix.toarray())

n_clusters_ = len(np.unique(ms.labels_))
print("Number of estimated clusters: {}".format(n_clusters_))


# As the number of clusters that Mean Shift clustering has found is only 1 (all novels are alike), it is not a good clustering method for this scenario.

# ## Spectral Clustering
# Spectral Clustering groups tokens that are similar based on the eigenvalues of the similarity matrix. Put another way, tokens that show up in similar contexts will determine which novels are grouped together.

# In[ ]:


sc = SpectralClustering(n_clusters=num_clusters)
sc.fit(tfidf_matrix)
clusters = sc.labels_.tolist()


# In[ ]:


sc_df = pd.DataFrame(sc.affinity_matrix_, index=filenames, columns=filenames)
print(sns.heatmap(sc_df.corr()))
sc_df


# In the above affinity matrix, the higher the number is, represents how similar the novel is compared to another novel. When a novel is compared against itself, it is 1 because the two works are identical. Looking at this, we can see that against_the_gods and battle_through_the_heavens are most similar together. praise_the_orc, is most similar to overgeared and desolate_era is also slightly less similar.  However, depending on the number of clusters set, it may or may not be in a category by itself. Emperors_domination isn't similar to any other work so it is likely that it will be in it's own cluster. Martial_god_asura is most similar to sovereign_of_the_three_realms, and martial_world is most similar to wu_dong_qian_kun. Let's see how accurate the predictions are.

# In[ ]:


novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])


# In[ ]:


print("Clusters:")
print('-'*40)

for i in range(num_clusters):
    if i != 0:
        print('\n')

    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')


# Looks like the predictions aren't too far off. The only difference was that desolate_era and emperors_domination were grouped together. If there were six clusters however, would they both have their own individual cluster?

# In[ ]:


num_clusters = 6
sc = SpectralClustering(n_clusters=num_clusters)
sc.fit(tfidf_matrix)
clusters = sc.labels_.tolist()


# In[ ]:


novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])


# In[ ]:


print("Clusters:")
print('-'*40)

for i in range(num_clusters):
    if i != 0:
        print('\n')

    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')


# ## Affinity Propagation
# Like Mean Shift, the number of clusters does not have to be specified for Affinity Propagation. Instead, it determines the number of clusters by finding tokens that are so similar between novels that it is possible that the words can be used in the other novels in the clusters.

# In[ ]:


af = AffinityPropagation()
af.fit(tfidf_matrix)

af_clusters = len(af.cluster_centers_indices_)
print("Number of estimated clusters: {}".format(af_clusters))


# In[ ]:


clusters = af.labels_


# In[ ]:


novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])


# In[ ]:


print("Clusters:")
print('-'*40)

for i in range(af_clusters):
    if i != 0:
        print('\n')

    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')


# Interestingly enough, Affinity Propagation figured that 4 clusters is enough to classify these novels. If I ran KMeans again with 4 clusters, would it group these works similarily?

# In[ ]:


num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()


# In[ ]:


novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])


# In[ ]:


print("Clusters:")
print('-'*40)

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    if i != 0:
        print('\n')
    print("Cluster %d words: " % i, end='')
    
    for j, ind in enumerate(order_centroids[i, :10]):
        if (j == 0):
            print('%s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
        else:
            print(', %s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
    print()
    
    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')


# Out of all the clustering methods applied, the clustering method that appears to best represent the data is KMeans. While KMeans and Affinity Propagation were able to group the clusters identically when the same number of clusters were specified, KMeans can take it a step further and figure out which exact words were the most similar that caused the novels to be grouped together.

# # Classifying Texts
# Let's keep the clustering method in the backburner for a bit and see if we can correctly identify which sentence originated from which novel. To do this, we will push all the sentences into a vectorizer which breaks each sentence down to words and gives it a tf-idf score, which shows how frequent the word appears in that sentence compared to total sentences. The tf-idf scores for all sentences will then be stored in a matrix, which will be used by supervised learning models. In addition, because the number of tokens generated is high, the tokens will also be checked and attempt to combine tokens that are most similar together, like Spectral Clustering.

# In[ ]:


vectorizer = TfidfVectorizer(max_df=0.75, # drop words that occur in more than 3/4 of the sentence
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False,
                             use_idf=True, # use inverse document frequencies in our weighting
                             norm=u'l2', # Apply a correction factor so that longer sentences and shorter sentences get treated equally
                             smooth_idf=True, # Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                             tokenizer=lemma_tokens                        
                            )

# Apply the vectorizer
tfidf_matrix = vectorizer.fit_transform(all_sentences)
print("Number of features: %d" % tfidf_matrix.get_shape()[1])


# In[ ]:


terms = vectorizer.get_feature_names()


# In[ ]:


# splitting into training and test sets
# keeps sentence structure
X_train, X_test, y_train, y_test = train_test_split(all_sentences, all_sentences_label, test_size=0.25, random_state=0)
# scores of tfidf
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_matrix, all_sentences_label, test_size=0.25, random_state=0)

# force output into compressed sparse row if it isn't already; readable format
X_train_tfidf_csr = X_train_tfidf.tocsr()


# In[ ]:


n = X_train_tfidf_csr.shape[0]
terms = vectorizer.get_feature_names()

# create empty list of dictionary, per sentence
sents_tfidf = [{} for _ in range(0,n)]

# for each sentence, list feature words and tf-idf score
for i, j in zip(*X_train_tfidf_csr.nonzero()):
    sents_tfidf[i][terms[j]] = X_train_tfidf_csr[i, j]


# In[ ]:


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# SVD data reducer.  Reduces the number of features from 5252 to 950.
svd= TruncatedSVD(950)
lsa = make_pipeline(svd, Normalizer(copy=False))
# Run SVD on the training data, then project the training data.
lsa_all_sents = lsa.fit_transform(tfidf_matrix)

variance_explained=svd.explained_variance_ratio_
total_variance = variance_explained.sum()
print("Percent variance captured by all components:",total_variance*100)

#Looking at what sorts of sentences solution considers similar, for the first three identified topics
sents_by_component=pd.DataFrame(lsa_all_sents,index=all_sentences)
for i in range(3):
    print('Component {}:'.format(i))
    print(sents_by_component.loc[:,i].sort_values(ascending=False)[0:10])


# In[ ]:


X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(sents_by_component, all_sentences_label, test_size=0.25, random_state=0)


# # Supervised Modeling

# In[ ]:


from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[ ]:


# logistic regression with tf idf
lr = LogisticRegression()

lr.fit(X_train_tfidf, y_train_tfidf)
print('Training set score:', lr.score(X_train_tfidf, y_train_tfidf))
print('Test set score:', lr.score(X_test_tfidf, y_test_tfidf))


# In[ ]:


# logistic regression with SVD
lr.fit(X_train_svd, y_train_svd)
print('Training set score:', lr.score(X_train_svd, y_train_svd))
print('Test set score:', lr.score(X_test_svd, y_test_svd))


# In[ ]:


# random forest with tf idf
rfc = ensemble.RandomForestClassifier()

rfc.fit(X_train_tfidf, y_train_tfidf)
print('Training set score:', rfc.score(X_train_tfidf, y_train_tfidf))
print('Test set score:', rfc.score(X_test_tfidf, y_test_tfidf))


# In[ ]:


# random forest with SVD
rfc.fit(X_train_svd, y_train_svd)
print('Training set score:', rfc.score(X_train_svd, y_train_svd))
print('Test set score:', rfc.score(X_test_svd, y_test_svd))


# In[ ]:


# support vector classification with tf idf
svc = SVC(kernel='linear', gamma='auto')

svc.fit(X_train_tfidf, y_train_tfidf)
print('Training set score:', svc.score(X_train_tfidf, y_train_tfidf))
print('Test set score:', svc.score(X_test_tfidf, y_test_tfidf))


# In[ ]:


# support vector classification with svd
svc.fit(X_train_svd, y_train_svd)
print('Training set score:', svc.score(X_train_svd, y_train_svd))
print('Test set score:', svc.score(X_test_svd, y_test_svd))


# Unfortunately, it appears that these supervised learning models aren't very accurate.  These models appear to highly overfit on the training data sets. However, out of all of these, it appears that logistic regression with SVD, while not the most accurate, does not overfit as much as the other models.

# Next, let's see how well the texts get clustered again with only 25% of the texts.

# In[ ]:


test_list = []
for title in filenames:
    tmp_list = ""
    for i, sentence_label in enumerate(y_test):
        if sentence_label == title:
            tmp_list += X_test[i] + '. '
    test_list.append(tmp_list)        


# In[ ]:


vectorizer = TfidfVectorizer(max_df=0.75, # drop words that occur in more than 3/4 of the sentence
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False,
                             use_idf=True, # use inverse document frequencies in our weighting
                             norm=u'l2', # Apply a correction factor so that longer sentences and shorter sentences get treated equally
                             smooth_idf=True, # Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                             tokenizer=lemma_tokens,
                             ngram_range=(1,3)                             
                            )

# Apply the vectorizer
tfidf_matrix = vectorizer.fit_transform(test_list)
print(tfidf_matrix.shape)


# In[ ]:


terms = vectorizer.get_feature_names()


# In[ ]:


km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()


# In[ ]:


novels = { 'title': filenames, 'text': test_list, 'cluster': clusters }
df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])


# In[ ]:


print("Clusters:")
print('-'*40)

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    if i != 0:
        print('\n')
    print("Cluster %d words: " % i, end='')
    
    for j, ind in enumerate(order_centroids[i, :10]):
        if (j == 0):
            print('%s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
        else:
            print(', %s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
    print()
    
    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')


# Interestingly enough, it appears that the clusters still group the novels together in the same way!
