#!/usr/bin/env python
# coding: utf-8

# CORD-19 BCG Vaccine
# ======
# 
# Query of the CORD-19 dataset to find articles discussing the BCG vaccine and it's potential effectiveness against COVID-19. 

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from cord19reports import install\n\n# Install report dependencies\ninstall()')


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import report, render\n\ntask = """\nname: query\n\nbcg vaccine:\n    query: +bcg vaccine trial\n    columns:\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n        - {name: BGC Vaccine Effective, query: bcg vaccine covid-19, question: Does BCG vaccine protect against COVID-19, snippet: True}\n        - name: Sample Size\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n"""\n\n# Build and render report\nreport(task)\nrender("query")')

