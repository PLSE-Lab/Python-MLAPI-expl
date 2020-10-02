#!/usr/bin/env python
# coding: utf-8

# ## Causal Relation extraction : from unstructured text data in biorxiv abstract
# 
# - We want to identify risk factors for the COVID-19 corona virus.  How they co-exist and how they interact. These risk factors can be psychological, demographical, environmental as well as genetic. The following are the algorithms we will be using to get closer to the goal. 
# 
# - We are Extracting  biomedical named entities using sciSpacy. Identifying cause-effect relationships among various entities using the causal-relation-prediction model [BERT-based causal relation extraction](https://github.com/wang-h/bert-relation-classification).This is an attempt to identify bioNER that has cause-effect with coronavirus or similar viruses. 
# 
# - Finally, this notebook part of the puzzle not end to end solution. 

# > 

# In[ ]:





# * ### Movinng pretrained R-Bert model to working directory after adding them to data: search for `causalcovidupdated`  : it is made public 

# In[ ]:


## copied the causal model file in working directory 

get_ipython().system('cp -r ../input/causalcovidupdated/models ../working/models')
get_ipython().system('cp -r ../input/causalcovidupdated/bert_relation/.* ../working/')


# * ### Let's download the requrie models and dependencies

# In[ ]:





# In[ ]:


get_ipython().system('pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html')
   
get_ipython().system('pip install pytorch-transformers==1.1')
get_ipython().system('pip install scispacy')
get_ipython().system('pip install negspacy')

get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz ## scispacy model')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz ## scispacy model ')


# > ### Let's load the needed libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from pprint import pprint
from copy import deepcopy

import argparse
from argparse import ArgumentParser
import glob
import logging
import os
import os.path as p

import sys
import random
import torch.nn as nn
import numpy as np
import torch
import socket



import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from tqdm import tqdm, trange



pd.set_option('display.max_colwidth', 500)
pd.set_option("display.width", 1000)



## importaing library to clean and tokenize the dataset 
from negspacy.negation import Negex
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
import spacy


## loading more library from downloaded libraries  
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# from tensorboardX import SummaryWriter
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule
import torch.nn.functional as F



# In[ ]:





# In[ ]:


## loading files for causal relation extraction : imported data folder in the working directory 

import bert
from utils import *

from config import Config ## from working directory 
#load the configuration file
config1 = Config("config.ini")

additional_special_tokens = []


from model import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("models")
tokenizer = BertTokenizer.from_pretrained("models", do_lower_case=True, additional_special_tokens=additional_special_tokens)

logger = logging.getLogger(__name__)
#additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]

#additional_special_tokens = ["e11", "e12", "e21", "e22"]


# * ### Loading the dataset and saving as df: just reading biorxiv in this notebook

# In[ ]:


import pandas as pd
## Let's start with the metadata csv file
df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv",
                        na_values=[], keep_default_na=False)
df_metadata.head()



# In[ ]:




df_citation = pd.read_csv("/kaggle/input/citation-network-output/ConsolidatedDfwithScore.csv", na_values=[], keep_default_na=False)
df_citation.head()


# In[ ]:


# df_citation.merge(df_biorxiv, on = "paper_id").head().sort(ScoreApproach2)


# In[ ]:





# ### More functions to get dafaframe from provided json

# In[ ]:


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)









def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df


# In[ ]:


biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames),filenames)


# In[ ]:


all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)


# In[ ]:



file = all_files[0]
print("Dictionary keys:", file.keys())


# In[ ]:


cleaned_files = []

for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
        format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    
    cleaned_files.append(features)


# In[ ]:


col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]

df_biorxiv = pd.DataFrame(cleaned_files, columns=col_names)
df_biorxiv.head()


# In[ ]:





# In[ ]:





# ### Cleaning the dataset for input to the causal model 
# 

# In[ ]:


# import sys, os
# import os.path as p

## a function to find teh downloaded model path in environment

def find_model(model_name):
    path_to_env = p.abspath(p.join(sys.executable, "../.."))
    path_to_modules = p.join(path_to_env, f"lib/python{sys.version[:3]}/site-packages")
    path_to_model = p.join(path_to_modules, model_name)
    if not p.exists(path_to_model):
        raise FileNotFoundError(path_to_model)
    model_dir = [d for d in os.listdir(path_to_model) if d.startswith(model_name)][0]
    return p.join(path_to_model, model_dir)


# ### NLP parsers 
# * we are using two sciSpacy models to extract biomedical entities from the provided abstract : `en_ner_bionlp13cg_md` and `en_ner_bc5cdr_md`
# * negspacy for negation detection

# In[ ]:





nlp_path = find_model("en_ner_bionlp13cg_md")
nlp2_path = find_model("en_ner_bc5cdr_md")

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load(nlp_path)
#nlp = spacy.load("/usr/local/lib/python3.6/dist-packages/en_core_sci_lg/en_core_sci_lg-0.2.4")
nlp2 = spacy.load(nlp2_path)
negex = Negex(nlp, language = "en_clinical_sensitive")
nlp.add_pipe(negex)
#linker = UmlsEntityLinker(resolve_abbreviations=True)
#nlp.add_pipe(linker)
nlp2.add_pipe(negex)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Functions to clean and annotate the dataset : for pretrained causal BERT based model

# In[ ]:


## Functions to clean the dataset 


import re
def remove_braces(text):
    text = re.sub(r" ?\([^)]+\)", "", text)
    return text


#######################################################
def clean_sentence(clean_sen):
  clean_sen = clean_sen.replace("\n"," ")
  clean_sen = remove_braces(clean_sen)
  return clean_sen


#######################################################
def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result






#######################################################
def extract_all_relations(doc,chunks):
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    relations = []
    for entity in filter(lambda w: w.text in chunks, doc):
        if entity.dep_ in ("attr", "dobj"):
            subject = [w for w in entity.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, entity))
        elif entity.dep_ == "pobj" and entity.head.dep_ == "prep":
            relations.append((entity.head.head, entity))
    return relations
#######################################################



def parse_sentence(sentence_text):
  tokens = nlp(sentence_text)
  tokens2 = nlp2(sentence_text)
  noun_chunks = [chunk.text for chunk in tokens.ents]
  noun_chunks2 = [chunk.text for chunk in tokens2.ents]
  
  ## what this negs doing 
  negs = [chunk._.negex for chunk in tokens.ents]
  negs2 = [chunk._.negex for chunk in tokens2.ents]

    
    ## why we are combining these like this....
  for index in range(len(noun_chunks2)):
    if (noun_chunks2[index] not in noun_chunks):
      noun_chunks.append(noun_chunks2[index])
      negs.append(negs2[index])

  return noun_chunks,negs

#################################################################

def annotate_sentence(cleaned_text,cleaned_nounchunks,negs):
  l_sentences = []
  l_start_nodes = []
  l_end_nodes = []
  l_negs = []
  size = len(cleaned_nounchunks)
  if (size >= 2):
    for x in range(0,size):
      for y in range(x,size):
        if (x != y): # not relevant
          mod_sen = cleaned_text.replace(cleaned_nounchunks[x],"#"+cleaned_nounchunks[x]+"# ")
          mod_sen = mod_sen.replace(cleaned_nounchunks[y],"$"+cleaned_nounchunks[y]+"$ ")
          if ((mod_sen.count('#') == 2) and (mod_sen.count('$') == 2)):
              #if (((mod_sen.count('#') % 2) == 0) & (((mod_sen.count('$') % 2) == 0))):
              #tokens_a = tokenizer.tokenize(mod_sen)
              #if (("#" in tokens_a) & ("$" in tokens_a)):
              #if ((len(tokens_a) >= 8) and ((tokens_a.count('#') == 2) and (tokens_a.count('$') == 2))):
              l_sentences.append(mod_sen)
              l_start_nodes.append(cleaned_nounchunks[x])
              l_end_nodes.append(cleaned_nounchunks[y])
              if (negs[x] | negs[y]):
                    l_negs.append(True)
              else:
                    l_negs.append(False)
  return l_sentences, l_start_nodes, l_end_nodes, l_negs



########################################################################

def prepare_annotationset_biorxiv(frame_df):

    dataset_sentences_paperid = []
    dataset_sentences = []
    dataset_start_nodes = []
    dataset_end_nodes = []
    dataset_negs = []

    #print(bio_df.index)
    for ind in frame_df.index:
         text = frame_df['abstract'][ind]
         #print(text)
         if ((text != None) & (type(text) == str)):
             doc = nlp(text)
             for sentence in doc.sents:
               sen = clean_sentence(sentence.text)
               noun_chunks, negs = parse_sentence(sen)
#                dataset_sentences_paperid.extend(frame_df['paper_id'][ind])
               dataset_sentences_paperid.extend([frame_df.at[ind,'paper_id']])
                
               annotated_sen, start_node, end_node, node_neg = annotate_sentence(sen,noun_chunks,negs)
               dataset_sentences.extend(annotated_sen)
               dataset_start_nodes.extend(start_node)
               dataset_end_nodes.extend(end_node)
               dataset_negs.extend(node_neg)
    return dataset_sentences_paperid, dataset_sentences, dataset_start_nodes, dataset_end_nodes, dataset_negs

##############################################################################



def write_annotation_set(sentences,output_file):
  ofile = open(output_file,"w+") 
  count = 8001
  for val in sentences:
    ofile.write(str(count)+"\t"+val+"\t"+"6"+"\n")
    count = count+1
  ofile.close()

################################################################################


def read_result():
  relations_found = []
  result_file = open("eval/sem_res.txt","r")
  lines = result_file.readlines()
  for line in lines:
    split_vals = line.split("\t")
    relations_found.append(split_vals[1].rstrip("\n"))
  return relations_found



################################################################################

def write_causal_set(sentences,output_file):
  ofile = open(output_file,"w+") 
  count = 1
  for val in sentences:
    ofile.write(str(count)+"\t"+val+"\n")
    count = count+1
  ofile.close()

###############################################################################




# 

# In[ ]:


## trying above defined functions to parse and annotate a single sentence 
entities, negs= parse_sentence("Developing a deep learning-based model for automatic COVID-19 detection on chest CT is helpful to counter the outbreak of SARS-CoV-2 ")
print(entities)
print(negs)


# In[ ]:


# ## removing rows with abstract == NAN
# df_causal = df_merged[df_merged['abstract'].notna()]


# ## taking top-k abstract -- as ranked by citation network

# k = 1000
# df_causal = df_causal[:k]
# df_causal.head()


# In[ ]:


df_biorxiv.shape


# ### I will be usign biorxiv dataset for further investigation 

# In[ ]:


## annotating the sentences and saving to data folder in workign directory: this data folder is part of improrted stuffs from Input directory

dataset_sentences_paperid, sentences , start_nodes, end_nodes, all_negs = prepare_annotationset_biorxiv(df_biorxiv) #df_biorxiv -- dataframe constructed usign biorxiv json files
len(sentences)
write_annotation_set(sentences,"data/dev.tsv")


# ## causal model on Annotated sentences 

# In[ ]:


#load the configuration file,  trained model and vocabulary are loaded earlier -- they are in workign directory 

# set up details for the type of device
if config1.local_rank == -1 or config1.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not config1.no_cuda else "cpu")
    config1.n_gpu = torch.cuda.device_count()
else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(config1.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    config1.n_gpu = 1
config1.device = device
model.to(config1.device)




# In[ ]:


## bert.evaluate(config1,model,tokenizer)
bert.evaluate(config1,model,tokenizer)


# In[ ]:


COVID19_SYM_FILTER = ['covid',
                    'sars-cov-2',
                    'covid-19',
                    'covid19',
                    'coronavirus',
                    'corona',
                    'coronavirus disease']


# In[ ]:


relations_found = read_result()
len(relations_found)


# In[ ]:


len(all_negs), len(end_nodes), len(start_nodes), len(sentences)


# In[ ]:


relations_found = read_result()
causal_sentences = []

column_names = ["paper_id", "risk_factor", "sentence"]

# causal_for_covid_df = pd.DataFrame(columns = column_names)

list_of_list = []

## we can 
for rel_index in range(len(relations_found)):
  if ("Cause-Effect" in relations_found[rel_index]):
      risk_factor = ""

      
      if ((start_nodes[rel_index] in COVID19_SYM_FILTER) or (end_nodes[rel_index] in COVID19_SYM_FILTER)):
        if (sentences[rel_index] and (all_negs[rel_index] == False)):
            risk_factor = (start_nodes[rel_index], end_nodes[rel_index])
            list_of_list.append([str(dataset_sentences_paperid[rel_index]),  str(risk_factor), str(sentences[rel_index])])
            
#           if (start_nodes[rel_index] not in COVID19_SYM_FILTER):
#             risk_factor = end_nodes[rel_index]
#           elif (end_nodes[rel_index] not in COVID19_SYM_FILTER):
#             risk_factor = start_nodes[rel_index]
#           else:
#             risk_factor = None
#           if (risk_factor != None):
#             list_of_list.append([str(dataset_sentences_paperid[rel_index]),  str(risk_factor), str(sentences[rel_index])])
#           else:
#             print("tada")
#             print([str(dataset_sentences_paperid[rel_index]),  str(risk_factor), str(sentences[rel_index])])


# In[ ]:





# In[ ]:


causal_for_covid_df = pd.DataFrame(list_of_list, columns=column_names)


# In[ ]:


# causal_for_covid_df = causal_for_covid_df[causal_for_covid_df.risk_factor not in COVID19_SYM_FILTER]

# write_causal_set(causal_sentences,"causal.txt")
causal_for_covid_df.to_csv("causal.csv") 


# In[ ]:





# In[ ]:


causal_for_covid_df = causal_for_covid_df.groupby('paper_id').agg({'risk_factor': ', '.join, 'sentence':'.'.join }).reset_index()
causal_for_covid_df.head()                


# In[ ]:





# ## Conclusion: 
# - we are able to extract risk-factors and sentences with risk-factors in varioius abstract.
# - In the dataframe (causal_for_covid_df):
#     * risk-factor column: a list of tuples -->. listing risk-factors having cause-effect relationship with coronavirus/Covid-19, etc.
#     * sentences column: the sentences in the abstract -->> with these risk factors 

# In[ ]:





# In[ ]:





# In[ ]:


# relations_found = read_result()
# causal_sentences = []

# column_names = ["paper_id", "risk_factor", "sentence"]

# causal_for_covid_df = pd.DataFrame(columns = column_names)

# # print(causal_for_covid_df)

# ## we can 
# for rel_index in range(0,len(relations_found)):
# #   print("\n")
#   if ("Cause-Effect" in relations_found[rel_index]):
#       print("\n")
#       risk_factor = ""
    
    
#       if ((start_nodes[rel_index] in COVID19_SYM_FILTER) or (end_nodes[rel_index] in COVID19_SYM_FILTER)):
#         if (sentences[rel_index] and (all_negs[rel_index] == True)):
#             # we are ignoring this for now but could be useful later
#             print(start_nodes[rel_index]+" not caused "+end_nodes[rel_index]+ " in : "+sentences[rel_index])
#             causal_sentences.append(start_nodes[rel_index]+" not caused "+end_nodes[rel_index]+ " in : "+sentences[rel_index])
#         else:  
#             print(start_nodes[rel_index]+" caused "+end_nodes[rel_index]+ " in : "+sentences[rel_index])
            
#             if (start_nodes[rel_index] not in COVID19_SYM_FILTER):
#                 risk_factor = end_nodes[rel_index]
#             elif (end_nodes[rel_index] not in COVID19_SYM_FILTER):
#                 risk_factor = start_nodes[rel_index]
#             else:
#                 risk_factor = None
            
#             if (risk_factor != None):
#                 print(dataset_sentences_paperid[rel_index])
                
#                 df_row = pd.DataFrame(dataset_sentences_paperid[rel_index], risk_factor, sentences[rel_index])
#                 causal_for_covid_df.append(df_row)
                
#             causal_sentences.append(start_nodes[rel_index]+" caused "+end_nodes[rel_index]+ " in : "+sentences[rel_index])

# write_causal_set(causal_sentences,"causal.txt")
# causal_for_covid_df.to_csv("causal.csv") 


# In[ ]:





# In[ ]:




# import re
    
# space_or_dash_regex = r"(?:\s+|\s*-\s*)"
# space_or_comma_regex = r"(?:\s+|\s*[,;]\s*)"
# space_or_colon_regex = r"(?:\s+|\s*:\s*)"
# int_number_regex = r"[+-]?\d+"
# real_number_regex = int_number_regex + r"(?:\.\d+)?"
# or_value_regex = r"OR\s+" + real_number_regex
# real_interval_regex = r"(?:" + real_number_regex + r"\s*-\s*" + real_number_regex + r"|\[" + real_number_regex +\
#                       r"\s+-\s+" + real_number_regex + r"\])"
# ci_regex = r"CI" + space_or_colon_regex + real_interval_regex
# or_ci_regex = or_value_regex + r"(?:" + space_or_comma_regex + r"\(?\s*" + real_number_regex + r"\s*%\s*" + ci_regex +\
#               r"(?:\s*\))?)?"
# comparator_or_space_regex = r"(?:\s*[=<>]\s*|\s+)"
# p_value_regex = r"p(?:" + space_or_dash_regex + r"value)?" + comparator_or_space_regex + real_number_regex
# severe_regex = r"(?:" + or_ci_regex + ")"
# design_regex = r"|".join([
#     r"case study",
#     r"cross" + space_or_dash_regex + "sectional(?:\s+case" + space_or_dash_regex + "control)?",
#     r"meta" + space_or_dash_regex + r"analysis",
#     r"(?:matched\s+)?case" + space_or_dash_regex + "control",
#     r"medical\s+records\s+review",
#     r"(?:non" + space_or_dash_regex + ")?randomized(?:\s+control)?\s+trial",
#     r"prospective\s+case" + space_or_dash_regex + "control",
#     r"prospective\s+cohort",
#     r"retrospective\s+cohort",
#     r"seroprevalence\s+survey",
#     r"syndromic\s+surveillance"
# ])
# sample_regex = r"(?:(?:sample|population)(?:\s+size)?(?:\s+of)?\s+" + int_number_regex + r"|" +\
#                int_number_regex + r"\s+(?:sample|population)(?:\s+size)?" + r")"

# severe_pattern = re.compile(severe_regex)
# design_pattern = re.compile(design_regex)
# sample_pattern = re.compile(sample_regex)






# In[ ]:


# df_relevant_information = causal_for_covid_df.copy()
# df_relevant_information['severe'] = None
# df_relevant_information['severe_significant'] = None
# df_relevant_information['design'] = None
# df_relevant_information['sample'] = None

# for text in df_relevant_information['text'].values:    
#     severe = severe_pattern.findall(text)
#     design = design_pattern.findall(text)
#     sample = sample_pattern.findall(text)
#     if len(severe) > 0:
#         df_relevant_information.loc[id, 'severe'] = severe[0]
        
#     if len(design) > 0:
#         df_relevant_information.loc[id, 'design'] = design[0]
#     if len(sample) > 0:
#         df_relevant_information.loc[id, 'sample'] = sample[0]
        


# In[ ]:





# In[ ]:




