#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Converting pubmed_id, Microsoft academic Paper ID, doi to str as by default it is coming as float.
# converting has_pdf_parse, has_pmc_xml_parse to bool type
meta = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv',dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str,
    'has_pdf_parse': bool,
    'has_pmc_xml_parse': bool
})
meta.head()


# In[ ]:


# No of columns
len(meta.columns)


# In[ ]:


# Checking the dataset and their reccords against the cord_uid
meta.count()


# In[ ]:


# checking different values for license and values associated with it.
meta.groupby("license")["cord_uid"].count()


# In[ ]:


# checking different values for "has_pdf_parse" and values associated with it.
print(meta.groupby("has_pdf_parse")["cord_uid"].count())
print(meta.groupby("has_pmc_xml_parse")["cord_uid"].count())

# checking the condition when both of them are true.
print(meta[(meta["has_pdf_parse"] == False) & (meta["has_pmc_xml_parse"] == True)].count())


# In[ ]:


# current directory
import os
os.getcwdb()


# We can see that total number sha is equal to total number of true values of has_pdf_parse.

# ### Compiled all understanding in to the below table about the metadata.csv
# 
# Total number of columns in metadata.csv : 18
# 
# 
# | Fields | Reccord count for each column | Description |
# |:--:|:--:|--|
# | cord_uid | 57366 | Unique id of scholarly article present in the dataset. This persistent identifier is a 8-char alphanumeric string unique to each entry. |
# | sha |43540 | There is only sha available for 43540  (Question why does the all records not have the sha? Sha is generated from the text or documents and it is unique for two different documents.)|
# | title | 57203  | This means there are entries which doesn't have the title available in metadata.csv |
# | doi  |54020  | Digital Object Identifier (Digital Object Identifier, is a string of numbers, letters and symbols used to permanently identify an article or document and link to it on the web.) |
# | pmcid |46804| PubMed Central (PMC) is a free digital repository that archives open access full-text scholarly articles|
# | pubmed_id | 40905 | PubMed is a free search engine accessing primarily the MEDLINE database of references and abstracts on life sciences and biomedical topics|
# | license | 57366 | There are 10661 with values "unk" which I think means unknown.|
# | publish_time | 57358 | Publish time of the article. |
# | authors | 54840 | Author of the papers. |
# | journal | 51576 | An academic journal is a peer-reviewed article |
# | Microsoft Academic Paper ID | 964 | Microsoft Academic Graph (MAG) Identifier |
# | WHO #Covidence | 1768 | World Health Organization (WHO) Covidence Identifier, populated for all CZI records and none of the other records |
# | has_pdf_parse | 43540 | refers to all JSON derived from PDF parses |
# | has_pmc_xml_parse | 22263 | bool type, refers to all JSON derived from PMC XML parses; filenames are PMC_ID.xml.json|
# | full_text_file | 48921 | str, "full_text_file" to signal the tar.gz file in which the full text json resides |
# | url | 57060 | URL to paper landing page where available. |
# 
# 
# 
# 
# Others terms to know:
# - CZI: Chan Zuckerberg Initiative
# - PMC: PubMed Central (PMC) is a free digital repository that archives open access full-text scholarly articles
# - bioRxiv: The preprint server for biology, operated by Cold Spring Harbor Laboratory, a research and educational institution. 
# - medRxiv: The preprint server for Health Sciences, operated by Cold Spring Harbor Laboratory, a research and educational institution.
# - pubmed: PubMed is a free search engine accessing primarily the MEDLINE database of references and abstracts on life sciences and biomedical topics.
# - Elsevier: Elsevier is a Dutch publishing and analytics company specializing in scientific, technical, and medical content. It is a part of the RELX Group, known until 2015 as Reed Elsevier. 
# 
# 
# ~Note for any changes in the structure of the data and fields, follow this [link](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/137474)

# In[ ]:


# checking number of the json files by checking files inside
import glob
kaggle_cord_data_root_path = "../input/CORD-19-research-challenge"
articles_path = glob.glob('{}/**/*.json'.format(kaggle_cord_data_root_path), recursive=True)
len(set((articles_path)))


# In[ ]:




