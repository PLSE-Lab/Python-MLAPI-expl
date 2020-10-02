#!/usr/bin/env python
# coding: utf-8

# # How to boostrap a Covid-19 semantic search engine in 30 minutes
# ## Using Python, Pandas, Gensim/Doc2Vec, UMAP & HDBSCAN...
# 
# Thanks to the [Allen Institute for AI](https://allenai.org/) who released [the CORD-19 dataset](https://pages.semanticscholar.org/coronavirus-research), we will see how we can setup a Semantic Search engine based on state of the art NLP Embedding technologies ([Gensim](https://radimrehurek.com/gensim/)/[Doc2Vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)) and innovative Dimensional Reduction ([UMAP](https://github.com/lmcinnes/umap)) and Clustering algorithms ([HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan)) to query the data and extract topics in a fully unsupervised way.

# ### Setup
# 
# [This Jupyter Notebook](https://colab.research.google.com/drive/14ASPnAt_Mq2rLHRixOgxiwNoeKjZlYgo) is designed to run on [Google Colaboratory](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/) which offer a free hosted python environment to run Data Science & Machine Learning experiments. The Python stack is already feature-packed, most well known Python packages are already installed and ready to import. And for lesser known packages, we can install them as needed with Pip.
# 

# In[ ]:


get_ipython().system('pip install hdbscan')
get_ipython().system('pip install langdetect')


# In this notebook, we will use following packages:
# 
# *   [Pandas](https://pandas.pydata.org): to fetch and manipulate the data.
# *   [langdetect](https://github.com/Mimino666/langdetect): to detect the language of a document.
# *   [Gensim/Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py): for text embedding into vectors of numbers.
# *   [UMAP](https://github.com/lmcinnes/umap): a fantastic new algorithm for Dimensional Reduction.
# *   [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan): the natural clustering algorithm to use on UMAP results.
# *   [Plotly](https://plotly.com/python/): for visualization of numerical data.
# *   [Wordcloud](https://github.com/amueller/word_cloud): to visualize textual results as cloud of words.

# In[ ]:


import gensim
import hdbscan
import langdetect
import matplotlib.pyplot as plt
import numpy
import pandas
import plotly
import umap
import wordcloud


# ### Fetch, prepare and clean the data
# 
# The CORD-19 dataset includes full text for more than 50000 scientific publications related to Coronaviruses. It also includes a metadata file which is well suited for our today's experiment. This is a single JSON file with the metadata for each paper but also the text of all abstracts that we will use to train our NLP model. This metadata file is refreshed every week, so do not hesitate to relaunch this notebook regularly to run on the last available data from [Semantic Scholar](https://pages.semanticscholar.org/coronavirus-research).

# In[ ]:


DATA_URL = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv'

datadf = pandas.read_csv(DATA_URL)
print(datadf.info())
datadf.head()


# After a quick overview on the data, we can spot a couple useful columns like:
# 
# *   **cord_uid:** unique ID for each document.
# *   **publish_time:** the date of the publication.
# *   **title:** title of the publication.
# *   **authors:** the list of author's name separated by semi-colons (;).
# *   **abstract:** the text of the abstract of the publication.
# *   **licence:** type of licencing applying to each paper (be careful to respect those licences if you make some bucks out of this dataset !).
# 
# Unfortunately, this dataset does not include a "language" column so we could filter on English-speaking publications in order to focus the NLP learning process (it is better to focus on one language at a time). 
# 
# Let's use the package "langdetect" to detect the language based on the publication title...
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef detect_language(text):\n  try:\n    return langdetect.detect(text)\n  except:\n    return None\n\ndatadf['language'] = datadf['title'].apply(detect_language)")


# Then we can compute some metrics like:
# 
# *   the number of publications per language.
# *   the number of publication per type of licence.
# 

# In[ ]:


print('Languages: %s' % datadf['language'].unique())
print(datadf[['language', 'cord_uid']].groupby('language')
                                      .count()
                                      .sort_values('cord_uid', ascending=False))
print('Licences: %s...' % datadf['license'].unique())
print(datadf[['license', 'cord_uid']].groupby('license')
                                     .count()
                                     .sort_values('cord_uid', ascending=False))


# Now, we can filter on English only publications (+40000 documents).

# In[ ]:


LANGUAGE = 'en'

docdf = datadf.loc[datadf['language'] == LANGUAGE, ['cord_uid', 'publish_time', 'title', 'authors', 'abstract']]               .rename(columns=dict(cord_uid='id', publish_time='date', abstract='text'))
docdf = docdf.groupby('id').first().reset_index()
docdf.sort_values('date', ascending=False, inplace=True)
docdf.dropna(inplace=True)

print(docdf.info())
docdf.head()


# ### Word Embedding with Gensim Doc2Vec
# 
# The [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py) model takes a list of tagged document as input (the publications text tagged with their ID) and produces a set of numerical vectors as output (one vector per document). It also give us a set of vectors describing the vocabulary of the corpus of publications (one vector per word in the corpus). Those vectors can be compared with cosine distance in order to estimate similarities between words, or between documents. Here, I choose to embed the textual data in a space of 300 dimensions. For each word, the resulting numerical vector will consist in 300 numbers. The training of the Doc2Vec model should take around 20 minutes to complete...

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nEPOCHS = 100\n\ndocs = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text,\n                                                                            deacc=True,\n                                                                            min_len=3,\n                                                                            max_len=100),\n                                             [id])\n        for id, text in docdf[['id', 'text']].values]\n\ndoc2vec = gensim.models.Doc2Vec(docs,\n                                vector_size=300,\n                                window=5,\n                                min_count=5,\n                                epochs=EPOCHS)")


# Let's have a look at the resulting vocabulary and query words similar to "covid".

# In[ ]:


KEYWORD = 'covid'

vocab = list(doc2vec.wv.vocab)
vocab.sort()
print('%d words in vocabulary...' % len(vocab))
print('Sample: %s   ...   %s...\n' % (vocab[:100], vocab[-100:]))

print('\nWords similar with "%s":' % KEYWORD)
pandas.DataFrame(doc2vec.wv.most_similar(KEYWORD))


# Interesting! Let's double check the first 5 words most similar to "covid":
# 
# *   **ncov:** is part of the formal name of the virus (2019-nCoV) which have been officialised as [SARS-CoV-2](https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome_coronavirus_2).
# *   **ncp:** stands for Novel Coronavirus Pneumonia, a temporary name of the disease before Covid-19 was officialised.
# *   **pedv:** stands for [Porcine Epidemic Diarrhea Virus](https://en.wikipedia.org/wiki/Porcine_epidemic_diarrhea_virus).
# *   **ibv:** stands for [Infectious Bronchitis Virus](https://en.wikipedia.org/wiki/Avian_coronavirus), a known coronavirus affecting poultries.
# *   **rsv:** stands for [Respiratory Syncytial Virus](https://en.wikipedia.org/wiki/Human_orthopneumovirus), another virus affecting the respiratory tract.
# 
# The Doc2Vec model seems to have grabbed the concept of coronavirus already !
# 

# ### Preparing the numerical data for the next step
# 
# Now, we can extract the word vectors, which are a numerical representation of the vocabulary learned on the CORD-19 textual dataset, and put them in a table. Then, the idea will be to reduce the dimensionality of that table and look for clusters of words.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nvecdf = pandas.DataFrame(doc2vec.wv.vectors, index=doc2vec.wv.index2word)\nprint(vecdf.info())')


# ### Similarity-friendly Dimensionality Reduction with UMAP
# 
# Time to introduce [UMAP](https://github.com/lmcinnes/umap)... It is a new algorithm which follow the same spirit as [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding). Both algorithms aim to reduce the dimensionality of the data while maintaining similarities between data points. The similarities are estimated based on neighboring points and some distance metric. Thanks to different mathematical and implementation choices, UMAP offers a couple advantages over t-SNE:
# *   It runs faster (than the scikit-learn implementation of t-SNE)
# *   It scales better on large datasets with many dimensions
# *   It supports well dimensionality reduction into more than 2 dimensions (ex: 3, 5 or 15 etc.) which is good for embedding or clustering use cases.
# 
# You can learn more about the mathematical fundamentals of UMAP ([Topological Data Analysis](https://en.wikipedia.org/wiki/Topological_data_analysis)) with [this PyData talk](https://youtu.be/7pAVPjwBppo) of the author, Leland McInnes, and you can also have a very good [introduction on how to apply UMAP on textual data](https://youtu.be/OtVR_ZnXLu4) by John Healy. Then for people used to t-SNE, there is a quite deep [comparison of both algorithms](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668) by Nikolay Oskolkov.
# 
# As I am embedding word vectors, I tell UMAP to use Cosine distance as the metric to estimate similarity between pairs of words. And I choose to reduce from 300 to 5 dimensions (rather than 2) which will improve the quality of the clustering coming in the next step.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n_umap = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, metric='cosine')\numapdf = pandas.DataFrame(_umap.fit_transform(vecdf), index=vecdf.index)\nprint(umapdf.info())")


# 5 dimensions is cool for clustering, but not ideal for visualization. So I run UMAP again with only 2 output dimensions for plotting purpose.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n_umap = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine')\numap2ddf = pandas.DataFrame(_umap.fit_transform(vecdf), index=vecdf.index)\nprint(umap2ddf.info())")


# ### Visualization of the UMAP embedding
# 
# Let's have a first look at the result! For that, we can do a scatter plot of the 2D UMAP result and color the points based on their cosine distance with the word "covid"... I also plot the distribution of the distances side by side.

# In[ ]:


COLORS = doc2vec.wv.distances(KEYWORD)
ALPHA = 0.1

figure = plotly.subplots.make_subplots(rows=1, cols=2)
figure.add_trace(dict(type='histogram', x=COLORS, nbinsx=100),
                 1, 1)
figure.add_trace(dict(type='scatter', mode='markers', x=umap2ddf[0], y=umap2ddf[1], text=umap2ddf.index,
                      marker=dict(color=COLORS, colorscale='Plasma', showscale=True, opacity=ALPHA)),
                 1, 2)
figure.update_layout(width=1000, height=400, showlegend=False, margin=dict(l=0, t=0, r=0, b=0))
figure.show()


# As we can see on this visualization, it is not easy to see clusters in the 2D output of UMAP.

# ### Density based clustering with HDBSCAN
# 
# [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) is a new clustering algorithm and the typical choice to use on UMAP results. It improves the [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) algorithm from scikit-learn by supporting clusters with different densities and is easier to use as we just need to tune the min_cluster_size parameter to increase or decrease the number of detected clusters. It also report points which it failed to assign to a cluster as outliers (then cluster ID is -1).

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n_hdbscan = hdbscan.HDBSCAN(min_cluster_size=15)\nclusters = _hdbscan.fit_predict(umapdf)\nunique_clusters = numpy.unique(clusters)\n\nprint('%d clusters...' % len(unique_clusters))\nprint('Clusters: %s...' % unique_clusters[:100])")


# Let's have a look at the distribution of points between clusters.

# In[ ]:


data = [dict(type='histogram', x=clusters)]
layout=dict(width=1000, height=300, margin=dict(l=0, t=0, r=0, b=0))
figure = plotly.graph_objs.Figure(data=data, layout=layout)
figure.show()


# Here, we can see that about 2/3 of the vocabulary is not well clusterized and corresponding words are reported as outliers. Still, we have some word clusters that we can look at too.

# ### Visualization of the clusters or topics
# 
# We can visualize detected clusters with scatter plots. On the left, I plot the full results. On the right, I plot only the detected clusters. We can move the mouse over the plot to see the word behind each data point.

# In[ ]:


ALPHA = 0.1

figure = plotly.subplots.make_subplots(rows=1, cols=2)
for cluster in unique_clusters:
  plotdf = umap2ddf[clusters == cluster]
  figure.add_trace(dict(type='scattergl', mode='markers', x=plotdf[0], y=plotdf[1],
                        marker=dict(opacity=ALPHA), name='cluster#%d' % cluster, text=plotdf.index),
                   1, 1)
  if cluster != -1:
    figure.add_trace(dict(type='scattergl', mode='markers', x=plotdf[0], y=plotdf[1],
                          marker=dict(opacity=ALPHA), name='cluster#%d' % cluster, text=plotdf.index),
                     1, 2)
figure.update_layout(width=1000, height=400, showlegend=False, margin=dict(l=0, t=0, r=0, b=0))
figure.show()


# ### Topic extraction based on keyword search
# 
# Finally, we can use the clusters to extract topics out of the vocabulary and we can query the Doc2Vec model for suggestions of topics related to the keyword "covid".
# 
# For that, I take the average word vectors for each cluster and query the Doc2Vec model for the most similar words. Then we can use [WordCloud](https://github.com/amueller/word_cloud) to visualize the most similar words to some topic (cluster).

# In[ ]:


KEYWORD = 'covid'
NRESULTS = 10
NWORDS = 50

search_vector = doc2vec[KEYWORD]

rezdf = pandas.DataFrame(dict(context=[c for c in unique_clusters if c != -1]))
rezdf['score'] = doc2vec.wv.cosine_similarities(search_vector, [doc2vec.wv.vectors[clusters == c].mean(axis=0)
                                                                for c in unique_clusters if c != -1])
rezdf.sort_values('score', ascending=False, inplace=True)
rezdf = rezdf.head(NRESULTS)
rezdf['sample'] = rezdf['context'].apply(
    lambda context: ', '.join([w for w, score in doc2vec.wv.most_similar([doc2vec.wv.vectors[clusters == context].mean(axis=0)],
                                                                         topn=NWORDS)]))

for row in rezdf.to_dict(orient='record'):
    print('Score: %f, Topic ID: %d, sample: %s' % (row['score'], row['context'], row['sample']))
    cluster_mean_vector = doc2vec.wv.vectors[clusters == row['context']].mean(axis=0)
    similarities = dict(doc2vec.wv.most_similar([cluster_mean_vector], topn=NWORDS))
    plt.figure(figsize=(16, 4))
    plt.imshow(wordcloud.WordCloud(width=1600, height=400)
                        .generate_from_frequencies(similarities))
    plt.title('Cluster#%d' % row['context'], loc='left', fontsize=25, pad=20)
    plt.tight_layout()
    plt.show()
    print()

rezdf


# Looking at the suggestions:
# 
# *    The first result (cluster#5) represents the topic of "known coronaviruses".
# *    The second result (cluster#6) is the topic of "symptoms".
# *    The third one (cluster#9) is the topic of "virus carrier" (from Human to animals).
# 
# I like the 5th results (cluster#15) concerning the topic of "mollecules that play some role in or around the virus". It mentions the ACE receptor which is said to be the entry door for the virus into our cells. For example, as a non-specialist of medecine or bio-sciences, this result suggested me to look at IFN (stands for [Interferon](https://en.wikipedia.org/wiki/Interferon), also comes under/with other names like Cytokine and Interleukin) which seems to attract a lot of interest in recent publications. It could maybe help against virus replication, though some recent publications may show [correlations between those proteins and severe cases of Covid-19](https://cord-19.apps.allenai.org/?q=covid%20cytokine).
# 
# 
# 

# ### Conclusion
# 
# Beside to keep me busy at home, this experiment is an attempt to share about some valuable ML algorithms (Gensim/Word2Vec/Doc2Vec, UMAP & HDBSCAN) available in the Python Open Source world and to show one way of putting them to work. I took some inspiration from [Top2Vec](https://github.com/ddangelov/Top2Vec), a similar experiment available on [Github](https://github.com/ddangelov/Top2Vec). I am not a specialist in bio-sciences and have no guarantee to offer.
# 
# Why I found the result surprisingly cool, I can see a few ways for improvement upon this notebook:
# 
# *   Training the Doc2Vec model on the full text of the publications and not only the abstract.
# *   Improving the text preprocessing (tokenisation): in the current setup, a term like SARS-CoV-2 is broken into 3 parts SARS, CoV and 2 and only "sars" and "cov" will appear in the vocabulary. Hence the model does not understant "SARS-CoV-2".
# *   Pre-training the vocabulary model (Word2Vec) with some large corpus of general text to "teach it English" before to specialize it on CORD-19 dataset.
# 
