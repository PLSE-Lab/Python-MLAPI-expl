#!/usr/bin/env python
# coding: utf-8

# CORD-19 Relevant Factors
# ======
# 
# This notebook shows the query results for a single task. CSV summary tables can be found in the output section.
# 
# The report data is linked from the [CORD-19 Analysis with Sentence Embeddings Notebook](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings).

# In[ ]:


from cord19reports import install

# Install report dependencies
install()


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import run\n\ntask = """\nid: 2\nname: relevant_factors\n\n# Field definitions\nfields:\n    common: &common\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n\n    containment: &containment\n        - {name: Factors, query: $QUERY, question: what containment method}\n        - {name: Influential, constant: "-"}\n        - {name: Excerpt, query: $QUERY, question: what containment method, snippet: true}\n        - {name: Measure of Evidence, query: countries cities, question: What locations}\n\n    weather: &weather\n        - {name: Factors, query: temperature humidity, question: What weather factor}\n        - {name: Influential, constant: "-"}\n        - {name: Excerpt, query: temperature humidity, question: How weather effects virus}\n        - {name: Measure of Evidence, query: countries cities, question: What locations}\n\n    appendix: &appendix\n        - name: Sample Size\n        - name: Sample Text\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n\n    columns: &columns\n        - *common\n        - *containment\n        - *appendix\n\n# Define query tasks\nEffectiveness of a multifactorial strategy to prevent secondary transmission: \n    query: Multifactorial strategy prevent transmission effect\n    columns: *columns\n\nEffectiveness of case isolation_isolation of exposed individuals to prevent secondary transmission:\n    query: Case isolation exposed individuals, quarantine effect\n    columns: *columns\n\nEffectiveness of community contact reduction:\n    query: Community contact reduction effect\n    columns: *columns\n\nEffectiveness of inter_inner travel restriction:\n    query: Travel restrictions effect\n    columns: *columns\n\nEffectiveness of school distancing:\n    query: School distancing effect\n    columns: *columns\n\nEffectiveness of workplace distancing to prevent secondary transmission:\n    query: Workplace distancing effect\n    columns: *columns\n\nEvidence that domesticated_farm animals can be infected and maintain transmissibility of the disease:\n    query: Evidence that domesticated, farm animals can be infected and maintain transmissibility of the disease\n    columns:\n        - *common\n        - {name: Factors, query: animals studied, question: what animals}\n        - {name: Influential, constant: "-"}\n        - {name: Excerpt, query: animals studied, question: "Can animals transmit SARS-COV-2"}\n        - {name: Measure of Evidence, query: confirmation method, question: What rna confirmation method used}\n        - *appendix\n\nHow does temperature and humidity affect the transmission of 2019-nCoV_:\n    query: Temperature, humidity environment affect on transmission\n    columns:\n        - *common\n        - *weather\n        - *appendix\n\nMethods to understand and regulate the spread in communities:\n    query: Methods to regulate the spread in communities\n    columns: *columns\n\nSeasonality of transmission:\n    query: Seasonality of transmission significant factors and effect\n    columns:\n        - *common\n        - *weather\n        - *appendix\n\nWhat is the likelihood of significant changes in transmissibility in changing seasons_:\n    query: transmission changes with seasonal change\n    columns:\n        - *common\n        - *weather\n        - *appendix\n"""\n\n# Build and display the report\nrun(task)')
