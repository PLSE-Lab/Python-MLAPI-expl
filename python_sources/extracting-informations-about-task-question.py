#!/usr/bin/env python
# coding: utf-8

# ## Summary
# 
# This notebook is the result of the collaborative work of a group of engineers at Atos BDS France.
# 
# In this notebook we treat the different questions of the COVID-19 Open Research Dataset Challenge task: **What is known about transmission, incubation, and environmental stability?**
# 
# Our objective was to **quickly identify sentences/excerpts that are most likely to answer specific questions** we find in the several tasks and the articles they are coming from.

# ### Example of outputs:
# 
# #### This example is taken from the final execution of the notebook to give the reader an idea of what to expect as a final result
# 
# ##### Given question: 
# > **What we know about incubation period ?**
# 
# ##### Outputs:
# 
# **In the paper: Epidemiological characteristics of 1212 COVID-19 patients in Henan, China. medRxiv**
#     
# First excerpt :
# > the following findings are obtained: 1) covid-19 patients in henan show gender (55% vs 45%) and age (81% aged between 21 and 60) preferences, possible causes were explored; 2) statistical analysis on 483 patients reveals that the estimated average, mode and median **incubation periods** are 7.4, 4 and 7 days; incubation periods of 92% patients were no more than 14 days; 3) the epidemic of covid-19 in henan has undergone three stages and showed high correlations with the numbers of patients that recently return from wuhan; 4) network analysis on the aggregate outbreak phenomena of covid-19 revealed that 208 cases were clustering infected, and various people's hospital are the main force in treating patients
# 
# **In the paper: Is a 14-day quarantine period optimal for effectively controlling coronavirus disease 2019 (COVID-19)?**
# 
# First excerpt :
# > the full range of **incubation periods** of the covid-19 cases ranged from 0 to 33 days among 2015 cases.
# 
# Second excerpt :
# > this cohort contained 4 transmission generations, and **incubation periods** of the cases between generations were not significantly different, suggesting that the virus has not been rapidly adapted to human beings.
# 
# Third excerpt :
# > interestingly, **incubation periods** of 233 cases (11.6%) were longer than the who-established quarantine period (14 days).
# 

# ## Approach
# Our approach is divided into three parts:
# 
# 1. For a given question we start by selecting the most similar papers to this question. Those papers are likely those which can contain answers for the questions:
# 
#     *In this step we will start by importing and preprocess the different documents then we will use **Glove** which is a nlp pretrained model on wikipedia corpus that will provide us we embedding for the abstract of each document and an other embedding for the question. Those two vecoriel resprenstation (embeddings) will be used to calculate the similarity between each document and the given question. then  we keep the most similar paper with the question (those paper which has the a score of similarity greater than a given threshold).*
#     
# 2. Extract keywords and key sentences from the given question (using a nlp model).
# 
#     *In this step we will extract keywords and key sentences form the question. For that we will use the Spacy en_core_web_md model which is an english multi-task CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl, to extract keywords. We will also use nltk_rake to extract key sentences form the question.*
# 
# 3. Output sentences
# 
#     *This step will provide us with the output of the notebook. 
#     For each keyword or key sentence and each similar paper we will return the sentences from the pape's body which contain the keyword/key sentence if and only if the paper contains together this keyword and the word covid19 or coronavirus.*
# 

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
import collections
from collections import Counter 
from typing import List, Dict, Any, Generator
from collections import OrderedDict
import gensim.downloader as api
from IPython.display import Markdown, display
from rake_nltk import Rake
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
import en_core_web_md
word_vectors = api.load("glove-wiki-gigaword-100")
nlp = en_core_web_md.load()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


#for dirname, _, filenames in os.walk('/kaggle/input/'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Methods and example of keywords and key sentences extraction from the task question using spacy and nltk_rake Packages

# In[ ]:


def get_keyword(task:str):
    # Use Rake from nltk to extract keyword from task question
    stops_words=stopwords.words('english')+['virus', 'know', 'known']
    r = Rake(stopwords=stops_words, min_length=2) # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(task)
    return r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
    
    
def get_keyword2(task:str):
    # Use spacy model to to extract keyword from task question
    result = []
    pos_tag = ['PROPN', 'NOUN'] 
    nlp.vocab["virus"].is_stop = True
    nlp.vocab["disease"].is_stop = True
    doc = nlp(task.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation or token.is_stop):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
                
    return result     


# In[ ]:


task='what we know about Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.'
print('key sentences extraction using rake_nltk: ')
print(get_keyword(task))

print('\nkeywords extraction using spacy: ')
print(get_keyword2(task))


# ## Utils functions for Data Proccessing
# The following python methods are written to help us to process, load, parse the json file in each folder of data.

# In[ ]:


def get_file_paths(data_path: str) -> List[str]:
    """ Return all JSON file from a folder """
    file_paths = [os.path.join(
        data_path, file_name) for file_name in os.listdir(
            data_path) if file_name.endswith(".json")]
    print(f"Found {len(file_paths)} files.")
    return file_paths

def read_file(file_path: str) -> Dict[str, Any]:
    """ Open JSON file and return dict() data """
    print(f"Reading file {file_path}.")
    with open(file_path, "r") as handler:
        json_data = json.loads(handler.read(), object_pairs_hook=OrderedDict)
    return json_data


# In[ ]:


def get_paths(json_data: Dict[str, Any]) -> List[List[str]]:
    """ Return all paths defined by JSON keys """
    paths = []
    if isinstance(json_data, collections.MutableMapping):
        for k, v in json_data.items():
            paths.append([k])
            paths += [[k] + x for x in get_paths(v)]
    elif isinstance(json_data, collections.Sequence) and not isinstance(json_data, str):
        for i, v in enumerate(json_data):
            paths.append([i])
            paths += [[i] + x for x in get_paths(v)]
    return paths

def extract_section_text_from_paths(paths: List[List[str]], section: str) -> Generator[List[str], None, None]:
    """ Yield paths of abstract, body_text, ref_entries or back_matter """
    section_paths = (path for path in paths if path[0] == section and path[-1] == "text")
    return section_paths

def extract_text_from_path(paths: List[str], data: Dict[str, Any]) -> Generator[str, None, None]:
    """ Use paths (list of keys to follow) to yield texts from JSON """
    for path in paths:
        node_data = data
        for key in path:
            node_data = node_data[key]
        if len(node_data) > 20:
            yield(node_data)


# In[ ]:


def preprocess_paragraphs(dict_of_paragraphs: Dict[str, Generator[str, None, None]]):

    print("TEXT PRE-PROCESSINGTO BE IMPLEMENTED")
    abstract=[]
    body=[]
    legends=[]
    #print(dict_of_paragraphs)
    for k, l in dict_of_paragraphs.items():  
        #print(f"== {k} ==")  # Abstract, body or figures legend as keys
        for s in l:  # Generatos of paragraphs as values
            if k=='abstract':
                abstract.append(s) 
            elif k=='body':
                body.append(s)
            else:
                legends.append(s)
                
    #print(len(abstract))  # Sentences as str
    #print(len(body))   
    #print(len(legends))
            
    return [abstract, body, legends]


# ## Loading all files
# We load all the abstract, body_text, and legends in all existing file in the five  five folder:
# 1. biorxiv_medrxiv/pdf_json
# 2. comm_use_subset/pdf_json
# 3. comm_use_subset/pmc_json
# 4. noncomm_use_subset/pdf_json
# 5. noncomm_use_subset/pmc_json
# 
# In total we have 25004 files

# In[ ]:


data_path=['/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',
'/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',
'/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json',
'/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json',
'/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json']
files_paths=[]
for fold in data_path:
    files_paths= files_paths+ get_file_paths(fold) 

print(len(files_paths))    


# ## Creating Pandas Data Frame containing all previous loaded files
# We create a Data frame including the paper_id, the title,  the absctract, the body and the legends for each paper

# In[ ]:



list_file=[]
list_titles=[]
list_abstract=[]
list_body=[]
list_legends=[]
for file_path in files_paths:

# Read JSON and extract key paths from JSON root to leaves
json_file_content: Dict[str, Any] = read_file(file_path)
list_titles.append(json_file_content['metadata']['title'])     
paths: List[List[str]] = get_paths(json_data=json_file_content)

# Extract paths in JSON data leading to a text from different sections
abstracts_paths = extract_section_text_from_paths(paths=paths, section="abstract")
body_paths = extract_section_text_from_paths(paths=paths, section="body_text")
figures_legends_paths = extract_section_text_from_paths(paths=paths, section="ref_entries")

# Get paragraphs (defined by the JSON structure) from these parts
paragraphs = {
    "abstract": extract_text_from_path(paths=abstracts_paths, data=json_file_content),
    "body": extract_text_from_path(paths=body_paths, data=json_file_content),
    "legends": extract_text_from_path(paths=figures_legends_paths, data=json_file_content)}

# Preprocess them
preprocessed_paragraphs: Dict[str, List[str]] = preprocess_paragraphs(dict_of_paragraphs=paragraphs)
list_file.append(file_path.split('/')[-1].split('.')[0])
list_abstract.append(' '.join([str(elem) for elem in preprocessed_paragraphs[0]]))
list_body.append(' '.join([str(elem) for elem in preprocessed_paragraphs[1]]))
list_legends.append(' '.join([str(elem) for elem in preprocessed_paragraphs[2]]))    


# In[ ]:


paper_table = pd.DataFrame(list(zip(list_file, list_titles, list_abstract, list_body, list_legends)), 
               columns =['ID', 'title','abstract','body', 'legends'])
del(list_file)
del(list_titles)
del(list_abstract)
del(list_body)
del(list_legends)
paper_table.head()


# ## Extract Information About Task question
# The following methods will use the Glove pretrained model to filter papers that are more similar to the question and the match the body of the paper with the extracted key words and sentences in order to provide us with the part of the body that likely answer the question as expalined in the Approach part. 

# In[ ]:


def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

def filter_similar_paper(df, task):
    simil_dict={}
    for id in df['ID']:
        #print(id)
        sentence=df[df['ID']==id]['abstract'].values[0]
        simil_dict[id]= word_vectors.wmdistance(sentence, task)
    simil_dict=filterTheDict(simil_dict, lambda elem: elem[1] <2)
        
    return df[df['ID'].isin(list(simil_dict.keys())) ]


# In[ ]:


def get_task_information(task:str, paper_table):
    # function that display for each extract word  the part of the paper_id and 
    #the part of the abstract that contains that word
    
    paper_table= filter_similar_paper(paper_table, task)
    print(paper_table.shape)
    
    #key_terms=get_keyword(task)+get_keyword2(task)
    key_terms=get_keyword(task)
    
    index=0
    for abstract in paper_table.iloc[:,3]:
        term_in=False
        for terms in key_terms:
            #print(terms)
            terms2=terms
            if terms2 in abstract.lower() and ("covid-19" in abstract.lower() or "coronavirus" in abstract.lower()): 
                term_in=True
        if term_in:
            if len(paper_table.iloc[index,1])>0:
                display(Markdown('<b>'+"In the paper:"+' ' + paper_table.iloc[index,1]+'<b>'))
            else:
                display(Markdown('<b>'+"In the paper:"+' ' + paper_table.iloc[index,0]+'<b>'))
            
            #display(Markdown(df.iloc[index,2]))        
            #display(Markdown('<b>'+'In conclusion:'+'<b>'))
            sentence3="<ul>"
            for sentence in abstract.split('. '):
                sentence2=sentence.lower()
                matched2=False
                for terms in key_terms: 
                    terms2=terms
                    if terms2 in sentence2: 
                        matched2=True
                        sentence2=sentence2.replace(terms, '<b>'+terms+'</b>')
                        #sentence2='<b>'+sentence2+'<b>'
                if matched2: 
                    sentence3=sentence3+"<li>"+sentence2.replace("\n","")+"</li>"
            display(Markdown(sentence3+"</ul>"))       
        
        index+=1


# Here we choose the question **What is known about incubation Periods?** from the task 1

# In[ ]:


get_task_information("what we know about  incubation periods", paper_table)

