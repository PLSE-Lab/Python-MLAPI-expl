#!/usr/bin/env python
# coding: utf-8

# CORD-19 Papers
# ======
# 
# Query of the CORD-19 dataset to find articles mentioning the CORD-19 dataset and methods used to analyze the data
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from cord19reports import install\n\n# Install report dependencies\ninstall()')


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import report, render\n\ntask = """\nname: query\n\ncord-19:\n    query: +cord-19\n    columns:\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n        - {name: Analysis, query: model ai learning nlp, question: What methods used to analyze data}\n        - name: Sample Size\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n"""\n\n# Build and render report\nreport(task)\nrender("query")')

