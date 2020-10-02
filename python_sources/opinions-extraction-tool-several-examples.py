#!/usr/bin/env python
# coding: utf-8

# # Opinions extraction tool: several examples

# ************************
# 
# ## Results example
# 
# > *What do we know about Chloroquine to treat covid-19?*
# 
# #### Opinions found in the article corpus:
# 
# - (47 sentences) claiming that it is under study and we still need to investigate.
# - (91 sentences) saying that indeed, Chloroquine has a proven effect.
# - (118 sentences) providing another treatment or advise against taking chloroquine at the moment.
# 
# ************************

# 
# 
# ## Intro
# 
# This notebook is the result of the collaborative work of a group of engineers at Atos/Bull.
# 
# Our goal was to **overcomes the problem of quickly finding different opinions** about a given subjet. In fact, it can be very difficult to quickly get reliable information: many different points of view are represented in the medias as well as in the scientific litterature.
# 
# Instead of simply returning the most closest sentences to the query, we chose to **extract the diferent opinions**, which can be shared by the different groups of people working on a subject.
# 
# We wanted to provide a tool easily reusable, reproducible and easy to understand as a Python library installable from Github.
# 
# ![Overview](https://raw.githubusercontent.com/MrMimic/covid-19-kaggle/master/images/kaggle_covid.png)
# 
# ## How it works 
# 
# ### Database creation
# 
# All titles, abstracts and body texts of the dataset are [inserted into an SQLite DB](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/database_utilities.py#L186) (only english articles for the moment).
# 
# They are preprocessed by using the [method we developed](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/text_preprocessing.py#L21). It will lower and stem the text, remove stopwords, remove numeric values, split texts into sentences and sentences into words.
# 
# A [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) embedding and a [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) models have been trained on this pre-processed corpus. Briefly, these models allow to get a fixed-length vector of numeric values to represent each word of the corpus (word2vec) and to weight each word regarding it's frequency among all the corpus and in each document (TF-IDF). The result is a parquet table, [stored on Github](https://github.com/MrMimic/covid-19-kaggle/blob/master/resources/global_df_w2v_tfidf.parquet), containing for each word a float vector and a TF-IDF score.
# 
# ![Table header](https://raw.githubusercontent.com/MrMimic/covid-19-kaggle/master/images/header_w2v_tfidf.jpg "Table header")
# 
# This embedding can be re-generated in more or less 30 minutes on a 8 vCPU machine by using [this script](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/scripts/train_w2v.py).
# 
# Each sentence from the corpus is pre-processed and [vectorized](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/embedding.py#L65). To do so, each pre-processed word from a sentence is represented by its vector and weithed by the TF-IDF score. Then, all vectors from the different words composing the sentence are averaged ([Mean of Word Embeddings](https://books.google.fr/books?id=tBxrDwAAQBAJ&pg=PA95&lpg=PA95&dq=mean+of+word+embedding+MOWE&source=bl&ots=7laX_HWKS0&sig=ACfU3U2DvGwGI6Bs4HTkX0_oP7Nf3UTP2A&hl=en&sa=X&ved=2ahUKEwiXguOJ9tjoAhX3D2MBHS6mAzoQ6AEwCnoECA0QKA#v=onepage&q=mean%20of%20word%20embedding%20MOWE&f=false)). All these pre-processed sentences are [stored in base](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/text_preprocessing.py#L151).
# 
# ### Query matching
# 
# The query is first [vectorised](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/query_matching.py#L57) by using the same strategy and tool as explained above. The cosine similarity of this sentence [versus all stored sentences](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/query_matching.py#L127) vectors is then computed. Briefly, it allows to check how each sentence of the dataset is close from the query. Only the top-k sentences are returned (filtered by minimal distance).
# 
# All these top-k closest sentences are then clusterised by a Kmean algorithm. These clusters will represent the different *opinions* found about the query.
# 
# All the closest sentence from each centroid is highlited (*ie*, the sentence reflecting the most the opinion on this subject) and an HTML (or markdown) report about closest sentences and most relevant papers is written.
# 
# ## What's cool
# 
# - Code is documented, cleaned, PEP8 complient and installable as a Python library. It is hosted on a public Github repository, allowing people to collaborate to the devlopment.
# - The trained embedding is not generic. Even if pre-trained models found on the Internet work well, the context of covid-19 and the kind of sentences to be processed make a locally trained embedding better.
# - Code is optimisez for RAM and rapid processing. Only the resulting DB built on all articles weights gigabytes.
# - The solution is highly portable (even on mobile with less sentences for example) due to the usage of SQLite.
# - Scripts to reproduce experiments (create the DB, perform a query, retrain W2V, generate pagerank score or perform multi queries) are published on the Github and usable out-of-the-box.
# 
# ## What's not
# 
# - The database containing all sentences weights more than 20Go. It is thus unusable on Kaggle. To overcome this issue, we first randomly selected 10 sentences from the body of all articles. Then, we add two parameters instead *only_newest* and *only_covid* instead which select respectively only articles published since 2019 or with an abstract containing a synonym of "covid".
# - Clusters quelity is highly dependent of the number of cluster. Once automatically estimated, it sometimes "split an opinion" into two distinct clusters which should have been merged.
# - The highly-specific embedding also has cons. Many words from queries have not been found enough in the corpus to be kept by the embedding model. Maybe the solution is to use a generic pre-trained embedding and to update it on our data.
# 
# ## What's next
# 
# Version 2.0 of this work will be released before the April, 15th. To come:
# 
# - ~~Auto-estimate K for the number of opinions.~~
# - ~~Maybe some interactive figures.~~
# - ~~Ranking best papers from opinions clusters regarding the authors and their background.~~
# - ~~Uploading the citation network as a directed graphe in Kaggle datasets.~~
# 
# **And during the round #2, we would to develop:**
# 
# - Create an interactive tool, hosted online for instant query.
# - Auto-test the code on Github with unitary tests on the methods to ensure quelity of the code.
# - A multi-lingual search (maybe with trained embedding on different languages instead of just translating the query).
# - Use a larger pre-trained embedding (on the same corpus but maybe with some data augmentation from PubMed on the given subjects).
# - Auto update of the newly published scientific litterature with a link to the Pubmed API.
# 
# ## List of tricks
# 
# - Only newest or covid-related articles are used in Kaggle, the resulting database is way too large.
# - Two regexs are applied on the texts: one transforming every possible way of writing "Coronavirus-2019" into "COVID-19", the other one transforming "Corona Virus" into "Coronavirus".
# - We filter non-english articles for the moment, so they are not in the dabatase.
# 
# ## Usage
# 
# Queries from the different tasks have been reformulated (or raw task if not) and [stored in Github](https://github.com/MrMimic/covid-19-kaggle/blob/master/resources/queries.json). All of them have been sent to the pipeline and the result are stored in markdown format [here](https://github.com/MrMimic/covid-19-kaggle/blob/master/resources/executed_queries.md) (queries have been matched versus the corpus released on April 3th which was not updated since on our devlopment workstations).
# 
# But we think a manual analyse of the refults reflects the most how this tool can be used.
# 
# For this notebook, we will focus on several questions and try to answer them by using our tool.

# ### Setup
# 
# The library can be easily [installed from github](https://github.com/MrMimic/covid-19-kaggle/blob/master/setup.py) by using [pip](https://pypi.org/project/pip/).

# In[ ]:


# Install custom library from Github
get_ipython().system('pip install -q --no-warn-conflicts git+https://github.com/MrMimic/covid-19-kaggle')

from c19 import parameters, database_utilities, text_preprocessing, embedding, query_matching, clusterise_sentences, plot_clusters, display_output

# Ugly dependencies warnings
import warnings
warnings.filterwarnings("ignore")


# Then, the parameters are loaded ([full explaination of the parameters](https://github.com/MrMimic/covid-19-kaggle/blob/master/src/main/python/c19/parameters.py)). 
# 
# *Parameters* class returns default configuration which can be customised as follow.
# 
# The *first_launch* parameter allows to create the database instead of just loading it.

# In[ ]:


import os

params = parameters.Parameters(
    first_launch=True,
    database=parameters.Database(
        local_path="local_database.sqlite",
        kaggle_data_path=os.path.join(os.sep, "kaggle", "input", "CORD-19-research-challenge"),
        only_newest=True,
        only_covid=True
    ),
    preprocessing=parameters.PreProcessing(
        max_body_sentences=0,
        stem_words=False
    ),
    query=parameters.Query(
        cosine_similarity_threshold=0.8,
        minimum_sentences_kept=500,
        number_of_clusters="auto",
        k_min=3,
        k_max=10,
        min_feature_per_cluster=100
    )
)


# We construct the database by loading all titles, abstracts and bodies (please remember that articles are filtered on Kaggle and they are not all loaded into the database).
# 
# We also build up a knowledge graph using each article and it's citations, to build up a citation network of the dataset. Then we applied the [pagerank](https://en.wikipedia.org/wiki/PageRank) algorithm on the network to get a score for each article. The higher the pagerank score is, the more important the article is in the network (regular citations by others, etc). This is used later to return the most imporant papers of an opinion cluster. **The ones that might be the most useful for a scientist to read**.

# In[ ]:


database_utilities.create_db_and_load_articles(
    db_path=params.database.local_path,
    kaggle_data_path=params.database.kaggle_data_path,
    first_launch=params.first_launch,
    only_newest=params.database.only_newest,
    only_covid=params.database.only_covid,
    enable_data_cleaner=params.database.enable_data_cleaner)


# The pre-trained embeddings are loaded from Github (scipt file to re-train it available [here](https://github.com/MrMimic/covid-19-kaggle/blob/83747070e1f63777c542f25df3a46b9a248ff68f/src/main/scripts/train_w2v.py)).
# 
# The embedding model can now return words vectors (which can be weighted by TF-IDF scores).

# In[ ]:


embedding_model = embedding.Embedding(
    parquet_embedding_path=params.embedding.local_path,
    embeddings_dimension=params.embedding.dimension,
    sentence_embedding_method=params.embedding.word_aggregation_method,
    weight_vectors=params.embedding.weight_with_tfidf)


# The sentences are pre-processed, vectorised and inserted into the SQLite database.

# In[ ]:


text_preprocessing.pre_process_and_vectorize_texts(
    embedding_model=embedding_model,
    db_path=params.database.local_path,
    first_launch=params.first_launch,
    stem_words=params.preprocessing.stem_words,
    remove_num=params.preprocessing.remove_numeric,
    batch_size=params.preprocessing.batch_size,
    max_body_sentences=params.preprocessing.max_body_sentences)


# All sentences are finally loaded in RAM to be matched versus the user query.

# In[ ]:


full_sentences_db = query_matching.get_sentences_data(
    db_path=params.database.local_path)


# The database is ready to be used.
# 
# ### Analyse: opinions about Chloroquine
# 
# For now, we will just illustrate how our tool can be use for the chloroquine case study.
# 
# If you forked the notebook, you can just relaunch from this cell when you change the query.

# In[ ]:


query = "What do we know about hydroxychloroquine to treat covid-19 disease?"


# The cosine similarity between each sentence and the query will be computed.
# 
# The minimal number of sentences to keep is defined by the **minimal_number_of_sentences** parameters.
# 
# This number will influence the **similarity_threshold** if this limit to not return enough sentences. It will be incidcated by this kind of log:
# 
# > Similarity threshold lowered from 0.8 to 0.79 due to minimal number of sentence constraint.

# In[ ]:


closest_sentences_df = query_matching.get_k_closest_sentences(
    query=query,
    all_sentences=full_sentences_db,
    embedding_model=embedding_model,
    minimal_number_of_sentences=params.query.minimum_sentences_kept,
    similarity_threshold=params.query.cosine_similarity_threshold)


# After filtering the closest sentences, we retrieve from our database the pagerank score of their corresponding papers. This will add a column pagerank in the dataframe.

# In[ ]:


closest_sentences_df = database_utilities.get_df_pagerank_by_doi(
    db_path=params.database.local_path, df=closest_sentences_df)


# Results are clustered, with the number of clusters *K* being auto-computed by [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). Then, sentences are ranked to extract the most representative sentence for each cluster.
# 
# > Please note that for this example, the number of cluster is set to "auto". If the query you want to run has an expected number of "opinions", you can replace the "auto" by the desired number of cluster. For example, in a yes/no question, you can set the number of cluster to two.
# 
# The "auto" argument of the number of clusters for the clustering algorithm can add computing time to the query. If too long, tweak the following parameters to get less close sentences or define a number of clusters manually.
# 
# - params.query.minimum_sentences_kept
# - query.cosine_similarity_threshold
# 
# This algorithm can sometimes design clusters of one or two sentences. To prevent this to happen, use the **min_feature_per_cluster** argument. The number of clusters will decrease gradually until all clusters contain this minimal amount of sentences (until it reach *min_cluster* argument value).

# In[ ]:


closest_sentences_df = clusterise_sentences.perform_kmean(
    k_closest_sentences_df=closest_sentences_df,
    number_of_clusters=params.query.number_of_clusters,
    k_min=params.query.k_min,
    k_max=params.query.k_max,
    min_feature_per_cluster=params.query.min_feature_per_cluster
)


# Now, let's plot these clusters as an interactive scatter plot.

# In[ ]:


plot_clusters.scatter_plot(
    closest_sentences_df=closest_sentences_df,
    query=query)


#  **Bigger symbols correspond to the nearest sentence for each cluster's centroid**.
# 
# It can be seen as the most representative sentence for this given opinion.
# 
# > Note that the legend is interactive and you can show/hide clusters or sections
# 
# And finally, let's output a pretty HTML report about these clusters.

# In[ ]:


display_output.create_html_report(
    query=query,
    closest_sentences_df=closest_sentences_df,
    top_x=2,
    db_path=params.database.local_path)


# #### Analyses results
# 
# Here, we could say that:
# 
# - A cluster <span style="color:red"><b>represents sentences claiming that it is under study and we still need to investigate</b></span>.
# - Another cluster would <span style="color:red"><b>represent sentences saying that indeed, Chloroquine has a proven effect</b></span>. 
# - A last cluster <span style="color:red"><b>represents sentences providing another treatment or advise against taking chloroquine at the moment</b></span>.
# 
# Assesing clusters in advance (pre-commit) is quite hard on Kaggle due to the ID of the cluster which can change for each launch. However, seeds have been fixed internally for reproducible KMeans.
# 
# ### Task neonates / pregnancy risk factor
# 
# Let's analyse another question.

# In[ ]:


query = "How neonates and pregnant women are susceptible of developing covid-19?"

closest_sentences_df = query_matching.get_k_closest_sentences(
    query=query,
    all_sentences=full_sentences_db,
    embedding_model=embedding_model,
    minimal_number_of_sentences=params.query.minimum_sentences_kept,
    similarity_threshold=params.query.cosine_similarity_threshold)

closest_sentences_df = database_utilities.get_df_pagerank_by_doi(
    db_path=params.database.local_path, df=closest_sentences_df)

closest_sentences_df = clusterise_sentences.perform_kmean(
    k_closest_sentences_df=closest_sentences_df,
    number_of_clusters=params.query.number_of_clusters,
    k_min=params.query.k_min,
    k_max=params.query.k_max,
    min_feature_per_cluster=params.query.min_feature_per_cluster
)

plot_clusters.scatter_plot(
    closest_sentences_df=closest_sentences_df,
    query=query)


# In[ ]:


display_output.create_html_report(
    query=query,
    closest_sentences_df=closest_sentences_df,
    top_x=2,
    db_path=params.database.local_path)


# <span style="color:red"><b>Here, we see two clusters, one not categorical about risk, other saying that there is no risk</b></span>.
# 
# ### Task tobacco as a risk factor

# In[ ]:


query = "Are smoking or pre-existing pulmonary disease (lung) risk factors for developing covid-19?"

closest_sentences_df = query_matching.get_k_closest_sentences(
    query=query,
    all_sentences=full_sentences_db,
    embedding_model=embedding_model,
    minimal_number_of_sentences=params.query.minimum_sentences_kept,
    similarity_threshold=params.query.cosine_similarity_threshold)

closest_sentences_df = database_utilities.get_df_pagerank_by_doi(
    db_path=params.database.local_path, df=closest_sentences_df)

closest_sentences_df = clusterise_sentences.perform_kmean(
    k_closest_sentences_df=closest_sentences_df,
    number_of_clusters=params.query.number_of_clusters,
    k_min=params.query.k_min,
    k_max=params.query.k_max,
    min_feature_per_cluster=params.query.min_feature_per_cluster
)

plot_clusters.scatter_plot(
    closest_sentences_df=closest_sentences_df,
    query=query)


# In[ ]:


display_output.create_html_report(
    query=query,
    closest_sentences_df=closest_sentences_df,
    top_x=2,
    db_path=params.database.local_path)


# ### Simple question
# 
# This tool can also be used to answer simple question, wich do not need any opinion.
# 
# Just set the number of cluster to one and get the first line of the sorted resulting dataframe.

# In[ ]:


query = "Which is the cell entry receptor for SARS-cov-2?"

params.query.number_of_clusters = 1

closest_sentences_df = query_matching.get_k_closest_sentences(
    query=query,
    all_sentences=full_sentences_db,
    embedding_model=embedding_model,
    minimal_number_of_sentences=params.query.minimum_sentences_kept,
    similarity_threshold=params.query.cosine_similarity_threshold)

closest_sentences_df = database_utilities.get_df_pagerank_by_doi(
    db_path=params.database.local_path, df=closest_sentences_df)

closest_sentences_df = clusterise_sentences.perform_kmean(
    k_closest_sentences_df=closest_sentences_df,
    number_of_clusters=params.query.number_of_clusters,
    k_min=params.query.k_min,
    k_max=params.query.k_max,
    min_feature_per_cluster=params.query.min_feature_per_cluster
)

print(f"\nAnswer to the query: {query}")
print(closest_sentences_df.sort_values(by="distance", ascending=False).head(1)["raw_sentence"][0])


# Keep in mind that these results are coming from a very limited number of papers (published since 01/01/2019) and the entire set will run on the web app.
# 
# More analyses to come before the end of round 1, with nice output and a ranking based on the citation network of the papers from each cluster, and other cool features! Check the git!
# 
# Don't hesitate to try the tool !
# 
# Stay safe !

# In[ ]:




