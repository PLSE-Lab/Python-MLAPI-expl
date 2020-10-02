#!/usr/bin/env python
# coding: utf-8

# # Demo of using Custom NER model on COVID-19 dataset 
# 
# **This is the first draft and we will use Med7 model to extract clinical named entities. Feedback most welcome!**
# 
# Extracting structured information from unstructured text such as EHRs and medical literature has always been a challenging task. Recent advancements in machine learning take advantage of the large text corpora available in scientific literature as well as medical and pharmaceutical web sites and train systems which can be leveraged for several NLP tasks ranging from text mining to question answering. Along with progress in the research space, there has been significant progress in the libraries and tools available for industry use. We will use a custiom trained NER model to identify entities like Chemical, Symptom, Dosage, Disease - use these entities to show prevelance from the released dataset.
# 
# Based on our paper at: https://arxiv.org/abs/1910.11241
# 
# Dataset available at: https://pages.semanticscholar.org/coronavirus-research
# 
# By Dattaraj J Rao (Persistent Systems) - https://www.linkedin.com/in/dattarajrao
# 
# ![COVID-19](https://upload.wikimedia.org/wikipedia/commons/0/09/Covid-19-4855688_640.png)

# In[ ]:


# install required packages - we will use sciSpacy
get_ipython().system('pip install -U spacy > silent.txt')
get_ipython().system('pip install scispacy > silent.txt')
get_ipython().system('pip install https://med7.s3.eu-west-2.amazonaws.com/en_core_med7_lg.tar.gz > silent.txt')


# ### We will use a named entity recognition (NER) model to extract entities like Drug from the text based on sentence structure. Then understand how the drugs in research literature are distributed.

# In[ ]:


# decide which text corpus to choose for analysis
JSON_PATH = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'


# In[ ]:


import os
import json
import warnings
warnings.simplefilter('ignore')

import spacy
nlp = spacy.load("en_core_med7_lg")

# create distinct colours for labels
col_dict = {}
list_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1']
for label, colour in zip(nlp.pipe_labels['ner'], list_colours):
    col_dict[label] = colour
options = {'ents': nlp.pipe_labels['ner'], 'colors':col_dict}

# fixed path
json_files = [pos_json for pos_json in os.listdir(JSON_PATH) if pos_json.endswith('.json')]
# all json files
'''json_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.json'):
            json_files.append(os.path.join(dirname, filename))
'''

# initialize entities dict - drugs
drugs = {}

# loop through the files
for jfile in json_files[::]:
    # for each file open it and read as json
    with open(os.path.join(JSON_PATH, jfile)) as json_file:
        covid_json = json.load(json_file)
        # read paper id
        #print('-'*50)
        #print("Paper: ", covid_json['metadata']['title'])
        #print('-'*50)
        # read abstract
        for item in covid_json['abstract']:
            text = item['text']
            doc = nlp(text)
            spacy.displacy.render(doc, style='ent', jupyter=True, options=options)
            # get list of drugs
            for ent in doc.ents:
                if ent.label_ == "DRUG":
                    # if drug exists increment, else add
                    if ent.text in drugs.keys():
                        drugs[ent.text] += 1
                    else:
                        drugs[ent.text] = 1
        # read body_text
        #for item in covid_json['body_text']:
        #    print(item['text'])


# ### Show Chart with list of drugs and their frequecies of occurance
# 
# The Named Entity Recognition models built using deep learning techniques extract entities from text sentences by not only identifying the keywords but also by leveraging the context of the entity in the sentence. Furthermore, with language model pre-trained embeddings, the NER models leverage the proximity of other words which appear along with the entity in domain-specific literature.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import operator

to_plot = dict(sorted(drugs.items(), key=operator.itemgetter(1),reverse=True)[:25])
print(to_plot)

plt.style.use('seaborn-white')
plt.figure(figsize=(20,10))
plt.title('Top DRUG names featuring in COVID-19 literature', fontsize=25)
plt.bar(range(len(to_plot)), list(to_plot.values()), align='center')
plt.xticks(range(len(to_plot)), list(to_plot.keys()), rotation=90, fontsize=20)

plt.show()

