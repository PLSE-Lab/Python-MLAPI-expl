#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tika')
from tika import parser
import pandas as pd
import json
from tqdm import tqdm_notebook as tqdm


# In[ ]:


def write_json(tika_json, doi):
    clean_doi =  doi.replace("/", "_")
    out_file = "COVID-19/ctakes-json/"+clean_doi+".json"
    with open(out_file, 'w') as jw:
        json_obj = json.dumps(tika_json, indent = 4)
        print("Writing JSON to:  ["+out_file+"]")
        jw.write(json_obj)


# In[ ]:


get_ipython().system('mkdir COVID-19')
get_ipython().system('mkdir COVID-19/ctakes-json')
get_ipython().system('mkdir COVID-19/2020-03-13/')


# In[ ]:


get_ipython().system('curl -k -o COVID-19/2020-03-13/metadata.csv https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-20/metadata.csv')


# In[ ]:


covid_all_sources =  "COVID-19/2020-03-13/metadata.csv"


# In[ ]:


covid_df = pd.read_csv(covid_all_sources)


# In[ ]:


covid_df


# In[ ]:


len(covid_df[covid_df.abstract.isnull() != True])


# In[ ]:


len(covid_df[covid_df['doi'].isnull() != True])


# In[ ]:


valid_abs_df = covid_df[covid_df.abstract.isnull() != True]


# In[ ]:


len(valid_abs_df[valid_abs_df.abstract.isnull()])


# In[ ]:


just_abstracts = valid_abs_df['abstract'].values


# In[ ]:


for i in tqdm(range(0, len(just_abstracts))):
    try:
        tika_json = parser.from_buffer(just_abstracts[i])
        write_json(tika_json, valid_abs_df.iloc[i]['doi'])
    except Exception as e:
        print("Error parsing: "+str(just_abstracts[i])+": Msg: "+str(e))


# In[ ]:




