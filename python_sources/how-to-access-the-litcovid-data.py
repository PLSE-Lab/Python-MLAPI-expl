#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import json

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # litcovid2BioCJSON: 
# 
# This file contains articles from the LitCovid corpus [1], which focuses on COVID-19, prepared in the BioC-JSON format [2]. 
# 

# In[ ]:


file_path = '/kaggle/input/litcovid/litcovid2BioCJSON.json'
with open(file_path) as json_file:
     json_file = json.load(json_file)
json_file


# # litcovid2pubtator.json: 
# 
# This file contains articles from the LitCovid corpus [1], which focuses on COVID-19. Each article is automatically annotated with six different entity types: gene/protein, drug/chemical, disease, cell type, species and genomic variants. The data is provided in BioC-JSON format [2], which is described at https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/. The full article text is included for articles in the PMC Open Access subset. The annotated articles can be visualized in PubTator [3] at https://www.ncbi.nlm.nih.gov/research/pubtator/.
# 

# In[ ]:


file_path = '/kaggle/input/litcovid/litcovid2pubtator.json'
with open(file_path) as json_file:
     json_file = json.load(json_file)
json_file

