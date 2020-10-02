#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Open Research Dataset Challenge (CORD-19)
# 
# Challenge repository at GitHub: https://github.com/chopeen/CORD-19/
# 
# ## Team
# 
# | Name               | Profile                                 |
# |--------------------|-----------------------------------------|
# | Adrianna Safaryn   | https://www.kaggle.com/adriannasafaryn  |
# | Anna Haratym-Rojek | https://www.kaggle.com/annaharatymrojek |
# | Cezary Szulc       | https://www.kaggle.com/cezaryszulc      |
# | Marek Grzenkowicz  | https://www.kaggle.com/chopeen          |
# 
# ## Goal
# 
# We wanted to use named entity recognition (NER) to highlight names of risk factors (RF). Our goal was
# training a **custom NER model for spaCy**, that could later be use to recognize risk factors in medical
# publications.
# 
# ![RF tags in Prodigy](https://raw.githubusercontent.com/chopeen/CORD-19/master/images/ner.png)
# 
# ## Pipeline
# 
# 1. Data preprocessing to extract 'risk factor(s)' sentences
# 2. Manual data annotation in Prodigy
# 3. Pretraining different models - we experimented with different base models and trained a number of
#    *tok2vec* layers to maximize the F-score
#     - base models: `en_vectors_web_lg`, `en_core_web_lg`, `en_core_sci_lg`
#     - *tok2vec* layers were trained for: RF sentences, subset of abstracts, all abstracts
# 4. Labelling more data by correcting the predictions of the top model trained in the previous step
# 5. Go back to step #3 to pretrain a new model using more data and then label even more data
# 6. Training the final model with all gathered annotations
# 
# ## Model performance
# 
# Each iteration uses all datasets from the previous one and adds more annotations. For detailed information about every
# trained model, see the notebook [train_experiments_2.ipynb](https://github.com/chopeen/CORD-19/blob/master/train_experiments_2.ipynb).
# 
# ### Base model `en_core_sci_lg`
# 
# | Iteration  | Datasets ([data/annotated/](https://github.com/chopeen/CORD-19/tree/master/data/annotated)) | Best F-score  |
# |------------|-------------------------------------------------|---------------|
# | 1          | `cord_19_rf_sentences`                          |   53.333      |
# | 2          | above + `cord_19_rf_sentences_correct`          | **75.630**    |
# | 3          | above + `cord_19_rf_sentences_correct_2`        |   74.894      |
# | 4          | above + `cord_19_rf_sentences_correct_3`        |   68.770      |
# 
# ### Base model `en_core_sci_md`
# 
# | Iteration  | Datasets ([data/annotated/](https://github.com/chopeen/CORD-19/tree/master/data/annotated)) | Best F-score  | Download |
# |------------|-------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------|
# | 1          | `cord_19_rf_sentences`                          |   57.778      | [en_ner_rf_i1_md](https://kagglecord19.blob.core.windows.net/risk-factor-ner/en_ner_rf_i1_md-0.0.1.tar.gz) |
# | 2          | above + `cord_19_rf_sentences_correct`          | **74.380**    | [en_ner_rf_i2_md](https://kagglecord19.blob.core.windows.net/risk-factor-ner/en_ner_rf_i2_md-0.0.1.tar.gz) |
# | 3          | above + `cord_19_rf_sentences_correct_2`        |   74.236      | [en_ner_rf_i3_md](https://kagglecord19.blob.core.windows.net/risk-factor-ner/en_ner_rf_i3_md-0.0.1.tar.gz) |
# | 4          | above + `cord_19_rf_sentences_correct_3`        |   69.725      | [en_ner_rf_i4_md](https://kagglecord19.blob.core.windows.net/risk-factor-ner/en_ner_rf_i4_md-0.0.1.tar.gz) |
# 
# Using a smaller base model (`md` instead of `lg`) results in significantly smaller model, while the F-score
# moves in both directions depending on the iteration.
# 
# ## Packaged models
# 
# Medium models for iterations 1..4 can be installed using the download links from the table above.
# 
# The directory [test/](https://github.com/chopeen/CORD-19/tree/master/test) contains a demo of the models in action (separate Conda environment + notebook).
# 
# ## Key files and resources
# 
# - Data preprocessing: [Kaggle notebook](https://www.kaggle.com/cezaryszulc/kaggle-covid-19-competition)
# - Training of *tok2vec* layers: [Kaggle notebook](https://www.kaggle.com/chopeen/spacy-with-gpu-support)
# - Full set of annotations:
#   - [cord_19_rf_sentences_merged.jsonl](https://github.com/chopeen/CORD-19/blob/master/data/annotated/cord_19_rf_sentences_merged.jsonl) (dump of the Prodigy dataset)
#   - [cord_19_rf_sentences_merged.json](https://github.com/chopeen/CORD-19/blob/master/data/annotated/cord_19_rf_sentences_merged.json) (spaCy JSON format)
# - Log of all experiments (including data annotation and model training): [train_experiments_2.ipynb](https://github.com/chopeen/CORD-19/blob/master/train_experiments_2.ipynb)
# - Early experiments: [train_experiments_1.ipynb](https://github.com/chopeen/CORD-19/blob/master/backup/early_experiments/train_experiments_1.ipynb)
# 
# ## Challenges
# 
# - Detailed discussion posted at the Kaggle forum:
#   [Custom NER model to recognize risk factor names](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/140451)
# - Question posted to the Prodigy support forum:
#   [Annotating compound entity phrases](https://support.prodi.gy/t/annotating-compound-entity-phrases/2796)
# 
# ## Tools
# 
# - [Prodigy](https://prodi.gy/) - text annotation
# - [spaCy](https://spacy.io/) - NLP and model training
# - [scispaCy](https://allenai.github.io/scispacy/) - specialized spaCy models for biomedical text processing
# - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) - environment setup (you can use
#   `conda env create -f environment.yml` to set up the Python environment with all packages and models)
# 
# ## Dataset citation
# 
# COVID-19 Open Research Dataset (CORD-19). 2020. Version 2020-03-13.  
# Retrieved from https://pages.semanticscholar.org/coronavirus-research.  
# Accessed 2020-03-26. doi:10.5281/zenodo.3715506
# 
# ## Notes
# 
# 1. [When to reject when annotating text for NER?](https://support.prodi.gy/t/when-to-reject-in-ner-manual-or-ner-make-gold/892/2)
# 1. [When should I press accept, reject or ignore?](https://prodi.gy/docs/named-entity-recognition#manual-accept-reject)
# 1. [`batch-train` is deprecated](https://prodi.gy/docs/recipes#deprecated)

# # CODE

# In[ ]:


get_ipython().system('pip install -U spacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')


# In[ ]:


from __future__ import unicode_literals, print_function
from pathlib import Path
from spacy.util import minibatch, compounding
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import itertools
import json
import nltk.data
import numpy as np
import os
import pandas as pd
import random
import spacy


# In[ ]:


# CONFIG

# Data
DIR_DATA_INPUT = os.path.join('/kaggle', 'input', 'CORD-19-research-challenge')
DIR_BIORXIV = os.path.join(DIR_DATA_INPUT, 'biorxiv_medrxiv', 'biorxiv_medrxiv', 'pdf_json')
DIR_COMM = os.path.join(DIR_DATA_INPUT, 'comm_use_subset', 'comm_use_subset', 'pdf_json')
DIR_CUSTOM = os.path.join(DIR_DATA_INPUT, 'custom_license', 'custom_license', 'pdf_json')
DIR_NONCUSTOM = os.path.join(DIR_DATA_INPUT, 'noncomm_use_subset', 'noncomm_use_subset', 'pdf_json')

DIR_DATA_OUTPUT = os.path.join('/kaggle', 'working')
PATH_AGG_JSON = os.path.join(DIR_DATA_OUTPUT, 'agg_data.json')


# In[ ]:


def extract_jsons_to_list(folder):
    """
    Extracting 4 fields ('abstract', 'text', 'paper_id', 'title') from orginal Json file
    :folder String, to location with Jsons
    :return: Lists, with selected params
    """
    results = []

    files = os.listdir(folder)
    for filename in tqdm(files, f'parsing {folder}'):
        json_file = os.path.join(folder, filename)
        file = json.load(open(json_file, 'rb'))
        agg_abstract_file = ' '.join(
            [abstract['text'] for abstract in file['abstract']])
        text = ' '.join(
            [text['text'] for text in file['body_text']])
        results.append({
            'abstract': agg_abstract_file,
            'text': text,
            'paper_id': file['paper_id'], 
            'title': file['metadata']['title']
        })

    return results


def save_json(file_to_save, path_to_save):
    """
    Save in relevant Json format
    :file_to_save DataFrame, file to save
    :path_to_save String, lacation to save a file
    """
    df = pd.DataFrame(file_to_save)
    
    df['json_output'] = df.apply(lambda x: {
        'text': x.text, "meta":{'paper_id':x.paper_id, 'title': x.title}
    }, axis=1)
    df['json_output'].to_json(path_to_save, orient='records', lines=True)
    

def filtr_covid_and_risk_factor(file_to_save, path_to_save):
    """
    List filtering in abstact and text (filters: 'COVID-19' or 'SARS-CoV-2')
    :file_to_save List, file to save
    :path_to_save String, lacation to save a file
    :return: DataFrame, valid data
    """
    df = pd.DataFrame(file_to_save)
    mask = df['abstract'].str.contains('COVID-19') | df['text'].str.contains('COVID-19')      | df['abstract'].str.contains('SARS-CoV-2') | df['text'].str.contains('SARS-CoV-2')
    
    abstracts = text_2_sentance(df[mask], 'abstract')
    text = text_2_sentance(df[mask], 'text')
    abstracts.extend(text)

    save_json(abstracts, path_to_save)
    
    return df


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def text_2_sentance(df, column):
    """
    Save 3 senctance before and after sentance which contains `risk factor` expression
    :df DataFrame, with text data
    :column String, column name to process
    :return: List, valid sentance
    """
    df['sentances'] = df.apply(lambda x: tokenizer.tokenize(x[column]), axis = 1)
    
    valid_sentance = []
    for _, row in tqdm(df.iterrows()):
        sentance_range = set()
        for index, singiel_sentance in enumerate(row['sentances']):
            if 'risk factor' in singiel_sentance.lower():
                sentance_range.update(
                    range(index-3, index+4))
        for valid_index in sentance_range:
            if valid_index >=0 and valid_index < len(row['sentances']):
                valid_sentance.append({
                    'text': row['sentances'][valid_index],
                    'paper_id': row['paper_id'], 
                    'title': row['title']
                })
                
    return valid_sentance


# In[ ]:


# Generate Json for Marek

bio = extract_jsons_to_list(DIR_BIORXIV)
comm = extract_jsons_to_list(DIR_COMM)
cus = extract_jsons_to_list(DIR_CUSTOM)
non = extract_jsons_to_list(DIR_NONCUSTOM)

list_agg = bio + comm + cus + non
results = filtr_covid_and_risk_factor(list_agg, PATH_AGG_JSON)


# # Download data for training

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/chopeen/CORD-19/master/data/annotated/cord_19_rf_sentences_merged.json')
get_ipython().system('ls -1')


# # Split dataset for train and test sets

# In[ ]:


new_list = []
file = json.load(open('cord_19_rf_sentences_merged.json', 'rb'))

df = pd.DataFrame(file)

X_train, X_test = train_test_split(
    df, test_size=0.2, random_state=42)

X_train.to_json('train_abstract_teach.json', orient='records')
X_test.to_json('test_abstract_teach.json', orient='records')


# # Train NER model

# In[ ]:


get_ipython().system('spacy train en models/ train_abstract_teach.json test_abstract_teach.json --pipeline ner --base-model en_core_sci_lg  --replace-components')

