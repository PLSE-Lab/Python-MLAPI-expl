#!/usr/bin/env python
# coding: utf-8

# CORD-19 Forecasting Articles
# ======
# 
# The following is a list of forecasting and modeling articles that could be good background reading for those working on the forecasting effort. 
# 
# Data generated using the [CORD-19 Analysis with Sentence Embeddings Notebook](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings).
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from cord19reports import install\n\n# Install report dependencies\ninstall()')


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import report, render\n\ntask = """\nname: forecasting\n\nForecasting and Modeling:\n    query: forecasting and modeling\n    columns:\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n        - name: Sample Size\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n"""\n\n# Build and render report\nreport(task)\nrender("forecasting")')

