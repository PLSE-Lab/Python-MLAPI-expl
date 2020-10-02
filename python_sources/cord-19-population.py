#!/usr/bin/env python
# coding: utf-8

# CORD-19 Population
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


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import run\n\ntask = """\nid: 1\nname: population\n\n# Field definitions\nfields:\n    common: &common\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n\n    population: &population\n        - {name: Addressed Population, query: $QUERY, question: What group studied}\n        - {name: Challenge, query: $QUERY, question: What challenge discussed}\n        - {name: Solution, query: solutions recommendations interventions, question: What is solution} \n        - {name: Measure of Evidence, constant: "-"} \n\n    appendix: &appendix\n        - name: Sample Size\n        - name: Sample Text\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n\n    columns: &columns\n        - *common\n        - *population\n        - *appendix\n\n# Define query tasks\nManagement of patients who are underhoused or otherwise lower social economic status:\n    query: patients poor, homeless, lower social economic status\n    columns: *columns\n\nMeasures to reach marginalized and disadvantaged populations:\n    query: Communicating with marginalized disadvantaged populations\n    columns: *columns\n\nMethods to control the spread in communities:\n    query: prevent spread in communities\n    columns: *columns\n\nModes of communicating with target high-risk populations:\n    query: Communicating with target high-risk populations - elderly, health care workers\n    columns: *columns\n\nWhat are recommendations for combating_overcoming resource failures_:\n    query: mitigate resource shortages\n    columns: *columns\n\nWhat are ways to create hospital infrastructure to prevent nosocomial outbreaks_:\n    query: prevent nosocomial outbreaks in hospitals\n    columns: *columns\n"""\n\n# Build and display the report\nrun(task)')

