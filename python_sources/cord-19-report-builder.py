#!/usr/bin/env python
# coding: utf-8

# CORD-19 Report Builder
# ======
# 
# This notebook allows searching a prebuilt model from the [CORD-19 Analysis with Sentence Embeddings Notebook](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings). 
# 
# Queries can be generated to build customized reports. Simply copy and edit this kernel and modify the query below. *Click Run All the first time* which will install all required software and models for the session. After that, you can simply edit and run the query. 
# 
# Results are shown inline with Markdown and exported to CSV.
# 

# ## Query Guide
# 
# The input format for queries in [YML](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html). Every query is required to have a field named "query", which has the search query for finding/retrieving documents. The query should be a series of keywords. The default method does not require any of the words to be present in results as it uses a similarity query. But if a term must be present, it can be prefixed with a plus (+).
# 
# Example queries:
# 
# **antiviral treatment**<br/>
# Returns all documents with the concept of antiviral treatment
#  
# **+antiviral treatment**<br/>
# Returns all documents with the concept of antiviral treatment. Requires matches to have the term antiviral.
# 
# Columns:
# 
# Queries can have any of the following default columns:
# 
# - Date
# - Study
# - Study Link
# - Journal
# - Study Type
# - Sample Size
# - Study Population
# - Matches
# - Entry
# 
# Additional generated columns can also be added. Currently, two types of columns are supported, constant columns and query columns.
# 
# - **constant column**: Generates a column with a constant value<br/>
# {name: Name, constant: Value}
# 
# - **query column**: Runs a subquery against each matching document to pull out additional information<br/>
# {name: Country, query: country city, question: What location}<br/>
# 
# The column above is named country and has a context query to find text within each search result matching country or city. This is important as the question/answer model work best on small snippets of text vs a whole document. The question field is the actual question to ask of the matching content.
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from cord19reports import install\n\n# Install report dependencies\ninstall()')


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import report, render\n\ntask = """\nname: query\n\nantiviral covid-19 success treatment:\n    query: antiviral covid-19 success treatment\n    columns:\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n        - {name: Country, query: what country}\n        - {name: Drugs, query: what drugs tested}\n        - name: Sample Size\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n"""\n\n# Build and render report\nreport(task)\nrender("query")')

