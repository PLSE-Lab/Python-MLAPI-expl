#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Human Genes Insights
# 
# Some code is originated from `CORD-19 Solution Toolbox`.
# 
# 
# The steps are:
# 
# * switch JSON files to text files (title, abstract, body_text);  
# * select parts of text including gene mutation words (enriched with Word2Vec): deleted, deleting, deletion, truncation, duplication, recombination, crossovers, transposition, insertion, inserted, inserting,  mutation, mutant, mutated, polymorphism, snp, variant, variation;  
# * find in those parts of text gene approved symbols (source: HGNC, HUGO Gene Nomenclature Committee);  
# * visualize gene insights.
# 
# Unrelevant findings may occur and would require disambiguation techniques.
# 
# ## Load packages

# In[ ]:


import json
import os
import re
import sys

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# ## HGNC Data

# In[ ]:


# read HGNC.txt downloaded from https://www.genenames.org/
hgnc_df = pd.read_csv('../input/hgnc-hugo-gene-nomenclature-committee/HGNC.txt', sep='\t')
hgnc_df.head()


# In[ ]:


# regex to find gene symbols and gene mutation words
r_symbols = re.compile('(?:^|\W)([A-Z][A-Z0-9orf\-_]*[@#]?)(?:$|\W)')
vocab = ['deleted', 'deleting', 'deletion', 'truncation',
         'duplication', 'recombination', 'crossovers', 'transposition',
         'insertion', 'inserted', 'inserting',
         'mutation', 'mutant', 'mutated',
         'polymorphism', 'snp',
         'variant', 'variation']
r_vocab = '(?:' + '|'.join(vocab) + ')'
r_vocab = re.compile(r_vocab)
symbols = set(hgnc_df['Approved symbol'].values)

# check that the regex finds all gene symbols
found_symbols = hgnc_df['Approved symbol'].str.extract('(?:^|\W)([A-Z][A-Z0-9orf\-_]*[@#]?)(?:$|\W)', expand=True)
(found_symbols[0]==hgnc_df['Approved symbol']).all()


# ## Data to be processed

# In[ ]:


# folders
DIR = '/kaggle/input/CORD-19-research-challenge'

l = [d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR,d))]
l.sort()
print(*l, sep='\n')


# In[ ]:


# perform everything:
# 1. switch all JSON files to TEXT files if needed
# 2. process TEXT files to collect all mentionned gene symbols
# 3. simple visualization

def run(dataset):
    # init variables
    json_folder_path = os.path.join(DIR, dataset, dataset)
    print(f'Files in folder "{dataset}": {len(os.listdir(json_folder_path))}')

    text_folder_path = os.path.join('../working', dataset, 'text')
    if not os.path.exists(text_folder_path):
        os.mkdir(os.path.join('../working', dataset))
        os.mkdir(text_folder_path)
    
    # switch all JSON files to TEXT files
    list_of_files = list(os.listdir(json_folder_path))
    
    if len(os.listdir(text_folder_path)) == 0:

        for file in tqdm(list_of_files):
            json_path = os.path.join(json_folder_path, file)
            text_path = os.path.join(text_folder_path, os.path.splitext(file)[0]+'.txt')
            with open(json_path) as json_file, open(text_path, 'w') as text_file:
                json_data = json.load(json_file)
                text_file.write(json_data['metadata']['title'])
                text_file.write('\n\n')
                text_file.write('\n\n'.join(x['text'] for x in json_data['abstract']))
                text_file.write('\n\n')
                text_file.write('\n\n'.join(x['text'] for x in json_data['body_text']))
            
    # process TEXT files to collect all mentionned gene symbols
    list_of_files = list(os.listdir(text_folder_path))
    c = Counter()

    for file in tqdm(list_of_files):

        text_path = os.path.join(text_folder_path, file)

        with open(text_path) as text_file:
            text_data = text_file.readlines()

        for text in text_data:

            if not r_vocab.search(text):
                continue

            found_symbols = [s for s in r_symbols.findall(text) if s in symbols]
            c.update(found_symbols)

    s = pd.Series(list(c.values()), index=c.keys())
    s = s.sort_values(ascending=False)
    
    # visualization
    
    print('Found {} gene symbols in "{}"'.format(len(s), dataset))
    
    # gene symbols
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Top 30 genes mentionned in "{}"'.format(dataset))
    sns.barplot(x=s[:30], y = s[:30].index);
    
    # gene names
    mapping = hgnc_df.set_index('Approved symbol')['Approved name']
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Top 30 genes mentionned in "{}"'.format(dataset))
    sns.barplot(x=s[:30], y = [mapping[i] for i in s[:30].index]);


# ## 1. biorxiv_medrxiv

# In[ ]:


# biorxiv_medrxiv
run('biorxiv_medrxiv')


# ## 2. comm_use_subset

# In[ ]:


# comm_use_subset
run('comm_use_subset')


# ## 3. custom_license

# In[ ]:


# custom_license
run('custom_license')


# ## 4. noncomm_use_subset

# In[ ]:


# noncomm_use_subset
run('noncomm_use_subset')

