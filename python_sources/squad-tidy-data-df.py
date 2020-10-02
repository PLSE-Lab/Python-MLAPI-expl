#!/usr/bin/env python
# coding: utf-8

# # Converting from JSON to Tidy Data Pandas Dataframe
# 
# In this notebook I share a script to convert the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) dataset from JSON format to a [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) Pandas Dataframe.
# 
# **About the SQuAD dataset**
# 
# The dataset is used in natural language processing (NLP) research in the field of machine learning and reading comprehension. The datset consists of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 
# 

# In[ ]:


import json
import pandas as pd
from pandas.io.json import json_normalize
from pprint import pprint


# ### Load data

# In[ ]:


# verify files are present
# !ls ../input

# path to dev dataset file
datapath = "../input/dev-v2.0.json"

# path to train dataset file
# datapath = "../input/train-v2.0.json"

# load data (json)
with open(datapath) as file:
    json_dict = json.load(file)
pprint(json_dict)


# ### Explore JSON file

# In[ ]:


# Inspect Nested Keys
print('top-level-keys: {}'.format(list(json_dict.keys())))
print('data keys: {}'.format(list(json_dict['data'][0].keys())))
print('paragraphs keys: {}'.format(list(json_dict['data'][0]['paragraphs'][0].keys())))
print('qas keys: {}'.format(list(json_dict['data'][0]['paragraphs'][0]['qas'][0].keys())))
print('answers keys: {}'.format(list(json_dict['data'][0]['paragraphs'][0]['qas'][0]['answers'][0].keys())))


# In[ ]:


# Count Corpora
print('Nbr Corpora: {}'.format(len(json_dict['data'])))


# In[ ]:


# Print Corpora Titles
print(list(json_normalize(json_dict,'data')['title']))


# ### Convert to Tidy DF

# In[ ]:


def convert_squad_to_tidy_df(json_dict, corpus):
    """This function converts the SQuAD JSON data to a Tidy Data Pandas Dataframe.
    
    :param obj json_dict: squad json data
    :param str corpus: name of squad corpora to select subset from json object
    
    :returns: converted json data
    :rtype: pandas dataframe
    
    """
    data = [c for c in json_dict['data'] if c['title']==corpus][0]
    df = pd.DataFrame()
    data_paragraphs = data['paragraphs']
    for article_dict in data_paragraphs:
        row = []
        for answers_dict in article_dict['qas']:
            for answer in answers_dict['answers']:
                row.append((article_dict['context'], 
                            answers_dict['question'], 
                            answers_dict['id'],
                            answer['answer_start'],
                            answer['text']
                           ))
        df = pd.concat([df, pd.DataFrame.from_records(row, columns=['context', 'question', 'id', 'answer_start', 'text'])], axis=0, ignore_index=True)
        df.drop_duplicates(inplace=True)
    return df


# In[ ]:


corpus = 'Normans' # only in dev dataset
# corpus = 'Culture' # only in train dataset

df = convert_squad_to_tidy_df(json_dict, corpus)
df.head()

