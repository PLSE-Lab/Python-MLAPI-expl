#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Clinical Trials API
# 
# You can import this into your own Notebooks using this utility script: https://www.kaggle.com/ajrwhite/cdc-clinical-trials-api
# 
# **Please _upvote_ if you find useful, and _acknowledge_ if you re-use it in your own pipeline!**
# 
# Automating the extraction of clinical trial interventions from the CDC's Clinical Trials database.
# 
# See: https://clinicaltrials.gov/api/gui/ref/api_urls

# In[ ]:


import requests
import json
import pandas as pd
import datetime


# In[ ]:


CDC_BASE_URL = 'https://clinicaltrials.gov/api/query/study_fields?expr=COVID-19&max_rnk=1000&fmt=json'


# To use the `study_fields` endpoint we need to pick some fields to extract from this list: https://clinicaltrials.gov/api/info/study_fields_list

# In[ ]:


cdc_extract_fields = [
    'BriefTitle',
    'DesignAllocation',
    'DesignMasking',
    'DesignMaskingDescription',
    'InterventionName',
    'InterventionType',
    'LastKnownStatus',
    'OfficialTitle',
    'OutcomeAnalysisStatisticalMethod',
    'OutcomeMeasureTimeFrame',
    'SecondaryOutcomeMeasure',
    'StartDate',
    'StudyFirstPostDate',
    'StudyFirstPostDateType',
    'StudyFirstSubmitDate',
    'StudyFirstSubmitQCDate',
    'StudyPopulation',
    'StudyType',
    'WhyStopped'
]


# In[ ]:


query_url = f'{CDC_BASE_URL}&fields={",".join(cdc_extract_fields)}'
print(query_url)


# In[ ]:


r = requests.get(query_url)


# In[ ]:


# Check we have a successful extract with code 200
r.status_code


# In[ ]:


# Load the JSON data to a dictionary
j = json.loads(r.content)


# In[ ]:


# This is quite a flat JSON structure, so can be loaded into a DataFrame
df = pd.DataFrame(j['StudyFieldsResponse']['StudyFields'])


# In[ ]:


# Some of the fields are single-item lists which can be cleaned
def de_list(input_field):
    if isinstance(input_field, list):
        if len(input_field) == 0:
            return None
        elif len(input_field) == 1:
            return input_field[0]
        else:
            return '; '.join(input_field)
    else:
        return input_field


# In[ ]:


for c in df.columns:
    df[c] = df[c].apply(de_list)


# In[ ]:


df['StudyFirstPostDate'] = pd.to_datetime(df.StudyFirstPostDate)
df = df.sort_values(by='StudyFirstPostDate', ascending=False)


# Print out all the interventions being tested

# In[ ]:


df[df.StudyType == 'Interventional'].head(100)


# In[ ]:


timestamp = datetime.datetime.now().date().isoformat()

# Write all results
df.to_csv(f'covid19_clinical_trials_{timestamp}.csv', index=False)

# Write just interventional trials
df[df.StudyType ==
   'Interventional'].to_csv(f'covid19_interventional_clinical_trials_{timestamp}.csv',
                                            index=False)

