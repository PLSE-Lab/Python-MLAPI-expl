#!/usr/bin/env python
# coding: utf-8

# # Using EuropePMC API to download academic papers
# In this notebook I will show how one can use [EuropePMC](https://europepmc.org/) to download full text of open access academic papers. [EuropePMC](https://europepmc.org/) is a large repository of academic papers that includes metadata of many publications and full text of some.

# In[ ]:


import pandas as pd


# We have downloaded metadata for papers authored by a famous neuroscientist in [this notebook](https://www.kaggle.com/chrisfilo/using-europepmc-api-to-access-academic-papers) - now we are loading into a dataframe.

# In[ ]:


df = pd.read_csv("../input/karl-fristons-papers-metadata/frisons_papers_metadata.csv")
df.head()


# We don't want all the papers on the list - only those where Friston is the first author.

# In[ ]:


df_first_author = df[df['authorString'].str.startswith('Friston')]
len(df_first_author)


# Only open access papers will have full text available - lets limit our list to those.

# In[ ]:


df_first_author_oa = df_first_author[df_first_author['isOpenAccess']]
len(df_first_author_oa)


# We will iterate over the list of papers and use their `id` to calle the `/fullTextXML` endpoint that will return XML version of papers. We will then download each of them and save to individual XML files.

# In[ ]:


import requests
url_tmpl = "https://www.ebi.ac.uk/europepmc/webservices/rest/{id}/fullTextXML"
for id in df_first_author_oa['id'].to_list():
    r = requests.get(url=url_tmpl.format(id=id))
    assert r.status_code == 200, "Downloading {id} failed.".format(id=id)
    with open(id + '.xml','w') as f:
        f.write(r.text)
        


# In[ ]:


get_ipython().system('ls')


# That's it! Different types of modelling will require different preprocessing of these papers. [Here's an example showing how to remove all XML tags](https://www.kaggle.com/chrisfilo/turn-xml-papers-into-plain-text).
