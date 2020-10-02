#!/usr/bin/env python
# coding: utf-8

# # Intro
# This notebook shows how to use CORD-19_ExpertSystem_MeSH.
# First, let's show the files in the input folder. They refer to [latest CORD-19 release (2020-05-01)]
# (http://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

# In[ ]:


# This Python 3 environment is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Input data files are available in the "../input/" directory.
import os
import json
import os
from collections import defaultdict as dd
import pandas as pd

counter = dd(list)
counter_files = 0
counter_empty_docs = 0

# run on last version
path = '/kaggle/input/cord19-expertsystem-mesh/cord19_expertsystem_mesh_060320'
for dirname, _, filenames in os.walk(path):
    if 'json' in dirname:
        print(f"{'/'.join(dirname.split('/')[-2:])} has {len(filenames)} files")
    for filename in filenames:
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(dirname, filename), 'r') as si:
            json_data = json.loads(si.read())
            counter_empty_keys = []
            counter_files += 1
            for key in json_data:
                if json_data[key]:
                    # we are only interested in knowing if the information is present
                    if key in ('language', 'cord_uid', 'paper_id'):
                        counter[key].append(1)
                    # for other fields, we want to know how many extractions for the current paper
                    else:
                        counter[key].append(len(json_data[key]))
                else:
                    counter_empty_keys.append(key)
            if len(counter_empty_keys) >= 4:
                counter_empty_docs += 1
                
print(f"Total files: {counter_files}")


# Let's open one of the files to see what the JSON looks like:

# In[ ]:


from pprint import pprint
pprint(json_data)


# For the sake of clarity, the following are the attributes accessible for this JSON file:

# In[ ]:


pprint(json_data.keys())


# # How to use this dataset
# The dataset is released as JSON files, mirroring the same folder structure of official CORD-19. 
# The JSON files are named as the original JSONs, making the pairing with CORD-19 JSON files straightforward.
# 
# In particular, the following fields are to be used to match the two datasets and refer to the column fields in CORD-19 metadata.csv:
# 
# * **cord_uid**: 8-char alphanumeric string unique to each entry of CORD-19 dataset, persistent across versions;
# * **paper_id**: paper identifier. If the original source is PDF, this corresponds to "sha" in metadata.csv, otherwise to "pmcid" if PMC xml.

# For each file of CORD-19, the dataset we just loaded contains additional metadata. 
# 
# * MeSH_tax: follows the standard MeSH taxonomy;
# * extractions: relevant concepts found in text with respect to MeSH taxonomy;
# * relations: relations among the entities (diseases, biological processes, drugs, genes, proteins, organizations);
# * organizations, i.e. named entities that identify organizations of any kind somehow relevant for the medical domain;
# * geographical areas, i.e. countries and other locations with MeSH and/or GeoNames ID;
# * dates, list of all dates found in full text;
# * language, stating the main language found looking at title, abstract and body of the json full texts.
# 
# As one can learn from the following table, MeSH-driven tags and extractions have been found for the vast majority of CORD-19 papers. For each paper, as many as 78 MeSH information and 51 extractions are found, on average. 
# 
# As for other fields, each paper has an average of 6 extraction for both health-related organizations and geographic places.
# 
# Finally, 3 out of 4 papers show relations between the MeSH entities recognized. 

# In[ ]:


data = dd(list)
headers = ['field', 'presence in files', 'extractions (sum)', 'extractions (mean)']
for field, extractions in counter.items():
    sum_total_extractions = sum(extractions)
    total_extractions = len(extractions)
    mean_extractions = f"{sum_total_extractions / total_extractions:.2f}"
    availability = f"{len(extractions) / counter_files * 100:.2f}"

    contents = [field, availability, sum_total_extractions, mean_extractions]
    for header, value in zip(headers, contents):
        data[header].append(value)

df = pd.DataFrame(dict(data))
df['extractions (sum)'] = df.apply(lambda x: "{:,.0f}".format(x['extractions (sum)']), axis=1)
print(df.head(10))
        


# Stay tuned! More notebooks will follow on how to make better use of the data we've just got accustomed to.
