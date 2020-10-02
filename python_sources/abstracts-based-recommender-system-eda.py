#!/usr/bin/env python
# coding: utf-8

# # EDA for abstracts based recommender system
# ## The aim of this project is to make an easy way to priopitize and recommend papers to researchers in a fast way to save the researchers' time and make them able to direct all their efforts to innovate a new way to fight COVID-19.
# 
# ### We will start first by data exploration then make a hypothesis based on our data exploration and domain knowledge, after that we will be implementingour solution using NLP and ML libraries like NLTK, TensorFlow and Scikit-Learn

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os


# There are a lot of files in the directory, json files for each paper, a description file and a metadata file for each paper, we will eplore the metadata file to see whether we can find some useful features that surves our purpose or not.

# In[ ]:


meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
print("Cols names: {}".format(meta.columns))
meta.head(7)


# In[ ]:


plt.figure(figsize=(20,10))
meta.isna().sum().plot(kind='bar', stacked=True)


# It seems that the two features 'Microsoft Academic Paper ID', 'WHO #Covidence' have a very huge number of missing values as obvious from the histogram so we will remove them to regulate the scale of the histogram's frequencies

# In[ ]:


meta_dropped = meta.drop(['Microsoft Academic Paper ID', 'WHO #Covidence'], axis = 1)


# In[ ]:


plt.figure(figsize=(20,10))

meta_dropped.isna().sum().plot(kind='bar', stacked=True)


# From the above histogram we can see there are small number of papres with missing urls and a considerable number with missing doi and abstracts. We are interested only in papers that have abstracts and either doi or url to be able to recommend them to the researcher. So lets explore some statistics about the papers with missing abstracts and remove them if possible.

# In[ ]:


miss = meta['abstract'].isna().sum()
print("The number of papers without abstracts is {:0.0f} which represents {:.2f}% of the total number of papers".format(miss, 100* (miss/meta.shape[0])))


# As we see from thee previous results the perscentage of missing abstracts is consedrable but we can ignore it as it is one of the most important features in our approach. Now lets see the number of missing doi and the number of missing urls from the papers without missing abstracts.

# In[ ]:


abstracts_papers = meta[meta['abstract'].notna()]
print("The total number of papers is {:0.0f}".format(abstracts_papers.shape[0]))
missing_doi = abstracts_papers['doi'].isna().sum()
print("The number of papers without doi is {:0.0f}".format(missing_doi))
missing_url = abstracts_papers['url'].isna().sum()
print("The number of papers without url is {:0.0f}".format(missing_url))


# Now lets see if the papers with missing urls have doi or not

# In[ ]:


missing_url_data = abstracts_papers[abstracts_papers["url"].notna()]
print("The total number of papers with abstracts but missing url and missing doi = {:.0f}".format( missing_url_data.doi.isna().sum()))


# Based on the above data exploration, we will need to remove the data that doesn't have both url and doi so that research scientists can find the paper recomended to them.
