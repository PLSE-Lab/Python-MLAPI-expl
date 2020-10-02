#!/usr/bin/env python
# coding: utf-8

# # Which compounds from which compound classes are in which clinical Trials ? 
# 
# A structure - activity relationship (SAR) notebook...
# 
# Using the analyzed version of the ClinicalTrials.gov AACT database, we are annotating chemical compounds in the AACT intervention section. 
# The notebook allows you to search for clinical trials that use the compounds in question as a treatment.
# Even more, using a chemical ontology you may retrieve all compounds that belong to a certain chemical class (e.g. "sesquiterpene derivatives") that are in clinical trials. This way you may see which structure types are most used as treatments in specific diseases or even disease classes (e.g. inflammatory diseases).

# ## 1. Connect to SciWalker-Open-Data via Google BigQuery

# In[ ]:


PROJECT_ID = 'sciwalker-open-data'
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID)


# ## 2. Create the query and fetch the results

# In[ ]:


query = (
"select distinct ct.NI_COMPOUND_NAME as compound, ct.NI_CONDITION_NAME as disease, ct.SOURCE as ct_id "+
"from ontologies.chemistry chem "+
"inner join ( "+
" select distinct ocid "+ 
" from ontologies.compound_classes "+
" where name='sesquiterpene derivatives' ) cls on cls.ocid=chem.ancestorid "+
"inner join clinical_trials_aact.aact_relations ct on ct.OCID_SUBJECT_COMPOUND=chem.ocid" )

query_job = client.query(query, location="US")  
df = query_job.to_dataframe()


# ## 3. Create the frequency table

# In[ ]:


from pandas import crosstab
tab = crosstab(index=df['disease'], columns=df['compound'])


# ## 4. Plot a heat-map of the frequency table[](http://)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5) 
fig, (ax) = plt.subplots(1, 1, figsize=(20,15))
hm = sns.heatmap(tab, ax=ax, cmap="YlOrRd", annot=True, vmin=0, vmax=10, linewidths=.05)


# ## 5. Result
