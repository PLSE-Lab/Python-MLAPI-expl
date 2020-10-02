#!/usr/bin/env python
# coding: utf-8

# # Following approach is designed to generate a Generic solution for NLP reserch task 
# ![download.png](attachment:download.png)
# ## Steps involved :- 
# 
# ## Graph Clustering data creation
# * ### Creation of Citation network graph:- [Base Files_extraction](https://www.kaggle.com/ashimak01/get-file-details)
# * ### Extension of Graph network using cosine Similarity between abstract of reserch paper. ( Code will be available in GIT repo) .
#     We used [BioSentVec](https://github.com/ncbi-nlp/BioSentVec) for this. Due to vec size , we ran this in Colab
#     
# * ### Combine Similarity and Citation Graph [Kernal for Join](https://www.kaggle.com/yatinece/combine-embedding-data-and-citations-article/data) 
# * ### Running Ego split clustering to generate category/Clusters of reserch present [Kernal Code](https://www.kaggle.com/debasmitadas/unsupervised-clustering-covid-research-papers)
# Once clusters were created we give name based on top keywords of reserch parpers , topic and search models
# 
# 
# ## Creating search to select best reserch papers to run NLP analysis on
# * ### For every task and sub-task we generated multiple queries. Task-SubTask Query Mapping [Subtask-Query-Data](https://www.kaggle.com/yatinece/covid19-task-query)
# 
# * ### A search based system is generated to find mose relevant reserch for each Sub-Task [Link](https://www.kaggle.com/sourojit/cord-biobert-search/)
# 
# ## Extractive summarization for reserch paper text
# * ### All search results papers text is summarized to run QA bots. Code avaliable in GitHub repo 
# 
# * ### Question answer bot was used to generate best answers for Each query . Code available at Github. [Data set](https://www.kaggle.com/yatinece/covid-task-qa-answer) 
# 
# ## Submission notebook
# ## [What is known about transmission, incubation, and environmental stability?](https://www.kaggle.com/yatinece/covid-19-task-1-transmission-incubation-graph-algo)
# 
# ## [What do we know about COVID-19 risk factors?](https://www.kaggle.com/yatinece/covid-19-task-2-risk-factors-graph-algo/)
#  
# ## [What do we know about virus genetics, origin, and evolution?](https://www.kaggle.com/debasmitadas/covid-19-task-3-virus-genetics-origin-graph-algo)
# 
# ## [What do we know about vaccines and therapeutics?](https://www.kaggle.com/debasmitadas/covid-19-task-4-vaccines-therapeutics)
# 
# ## [What has been published about medical care?](https://www.kaggle.com/rajesh60/covid-19-task-5-medical-care-using-bio-bert-graph)
# 
# ## [What do we know about non-pharmaceutical interventions?](https://www.kaggle.com/sourojit/covid-19-task-npi-extraction-bio-bert-graph)
# 
# ## [Sample task with sample submission(Help us understand how geography affects virality)](https://www.kaggle.com/aakashdeep/citation-graph-filter-similarity-qa-geography)
# 
# ## [What do we know about diagnostics and surveillance?](https://www.kaggle.com/aakashdeep/citation-graph-filter-similarity-qa-diagnostic)
# 
# ## [What has been published about ethical and social science considerations?](https://www.kaggle.com/kushagra607/cord-19-citation-graph-similarity-qa-ethical)
# 
# ## [What has been published about information sharing and inter-sectoral collaboration?](https://www.kaggle.com/kushagra607/cord-19-citation-graph-similarity-qa-collaboration)
# 
# ## Git repo
# [GitHub Repo](https://github.com/Aakash5/cord19m)
# ### Other tools
# #### [Search system based on Category of keywords mentioned in reserch paper](https://www.kaggle.com/yatinece/search-system-for-top-article-using-wikipedia-db)
# 
# ## Installing and testing sentence embedding extraction modules. Testing For GPU and setting up device. 
# ### Using Pytorch based Bio-Bert download via biobert-embedding
# 
# ### **Testing below package to generate embedding** 
# [Biobert Reference](https://github.com/Overfitter/biobert_embedding)
# **This package main code is modified to run it on GPU**
# 
# [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
# 
# 
# 
# #### *This will select best sentence via Network X graph. Currently using Pagerank method* Other methods can also me used like degree_centrality betweenness_centrality eigenvector_centrality
# 
# 
