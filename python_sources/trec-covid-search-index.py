#!/usr/bin/env python
# coding: utf-8

# # TREC-COVID Search Index
# 
# This notebook builds a search index over the TREC-COVID dataset. Background on how this search index works can be found in the [CORD-19 Analysis with Sentence Embeddings](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings) notebook.
# 
# ## Install environment
# Install the cord19q library and scispacy.

# In[ ]:


# Install cord19q project
get_ipython().system('pip install git+https://github.com/neuml/cord19q')

# Install scispacy model
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz')


# # Build articles database and embeddings index
# 
# The following section loads the necessary backing datasets, builds a SQLite database and an embeddings index to support searching. When complete, the output directory will have an cord19q directory with the search index.

# In[ ]:


import os
import shutil

from cord19q.etl.execute import Execute as Etl
from cord19q.index import Index

# Copy study design models locally
os.mkdir("cord19q")
shutil.copy("../input/cord19-study-design/attribute", "cord19q")
shutil.copy("../input/cord19-study-design/design", "cord19q")

# Build SQLite database for metadata.csv and json full text files
Etl.run("../input/trec-covid-information-retrieval/CORD-19/CORD-19", "cord19q", "../input/cord-19-article-entry-dates/entry-dates.csv", False)

# Copy vectors locally for predictable performance
shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")

# Build the embeddings index
Index.run("cord19q", "/tmp/cord19-300d.magnitude")

