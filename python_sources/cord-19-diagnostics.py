#!/usr/bin/env python
# coding: utf-8

# CORD-19 Diagnostics
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


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import run\n\ntask = """\nid: 6\nname: diagnostics\n\n# Field definitions\nfields:\n    common: &common\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n\n    diagnostics: &diagnostics\n        - {name: Detection Method, query: confirmation method, question: What rna confirmation method used}\n        - {name: Measure of Testing Accuracy, query: sensitivity specificity accuracy, question: What is assay detection accuracy}\n        - {name: Speed of Assay, query: turnaround time minute hour, question: What is assay detection speed}\n\n    appendix: &appendix\n        - name: Sample Size\n        - name: Sample Text\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n\n    columns: &columns\n        - *common\n        - *diagnostics\n        - *appendix\n\n# Define queries\nDevelopment of a point-of-care test and rapid bed-side tests:\n    query: Development of a point-of-care test and rapid bed-side detection methods\n    columns: *columns\n\nDiagnosing SARS-COV-2 with antibodies:\n    query: diagnose sars-cov-2 antibodies\n    columns: *columns\n\nDiagnosing SARS-COV-2 with Nucleic-acid based tech:\n    query: diagnose sars-cov-2 nucleic-acid\n    columns: *columns\n"""\n\n# Build and display the report\nrun(task)')

