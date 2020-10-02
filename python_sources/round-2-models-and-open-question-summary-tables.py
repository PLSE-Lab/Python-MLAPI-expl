#!/usr/bin/env python
# coding: utf-8

# # Round 2 - summary tables that address models and open questions related to COVID-19
# 
# ![](https://sportslogohistory.com/wp-content/uploads/2018/09/georgia_tech_yellow_jackets_1991-pres-1.png)
# 
# **GOAL:** Create automated article summary tables that address model and open question studies related to COVID-19
# 
# **SCENARIO:** When a new virus is discovered and causes a pandemic, it is important for scientists to get specific information coming from all scientific literature that may help them combat the pandemic.  In an effort to better understand the virus, scientists need to know what data exists for models and open questions of interest.
# 
# **PROBLEM:** The challenege, however, is that the number of scientific papers created is large and the papers are published very rapidly, making it nearly impossible for scientists to digest and understand important data in this sea of data.  When the COVID-19 Challenge started in March 2020, there were about 50,000 documents in the competition corpus.  As of this writing (June 15, 2020), there are about 140,000 documents in the corpus, almost triple in a three month period. Just keeping up with the volume of papers released is a daunting task, not to mention the near impossible task of searching through them trying to isolate answer to open questions.
# 
# **SOLUTION:** Create an unsupervised scientific literature understanding system that can analyze a very large corpus - made up of tens or hundreds of thousands of scientific papers and return specific text excerpts containing answers to open questions.
# 

# In[ ]:


import pandas as pd
import numpy as np
import functools
import re
from IPython.core.display import display, HTML
import string

### BERT QA
import torch
get_ipython().system('pip install -q transformers --upgrade')
from transformers import *
modelqa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


print ('Python packages loaded')
# keep only documents with covid -cov-2 and cov2
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df

# load the meta data from the CSV file
#usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file']
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
print ('ALL CORD19 articles',df.shape)
#fill na fields
df=df.fillna('no data provided')
#drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")
#keep only 2020 dated papers
df=df[df['publish_time'].str.contains('2020')]
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
#show 5 lines of the new dataframe
df=search_focus(df)
print ('Keep only COVID-19 related articles',df.shape)

import os
import json
from pprint import pprint
from copy import deepcopy
import math


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

def format_tables(ref_entries):
    extract=''
    for x in ref_entries.values():
        if 'html' in x:
            start = '<html>'
            end = '</html>'
            x=str(x).lower()
            dat=(x.split(start))[1].split(end)[0]
            extract=extract+' '+dat
    
    return extract

df['tables']='N/A'
for index, row in df.iterrows():
    #print (row['pdf_json_files'])
    if 'no data provided' not in row['pdf_json_files'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files'])==True:
        with open('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']) as json_file:
            #print ('in loop')
            data = json.load(json_file)
            body=format_body(data['body_text'])
            #ref_entries=format_tables(data['ref_entries'])
            #print (body)
            body=body.replace("\n", " ")
            text=row['abstract']+' '+body.lower()
            df.loc[index, 'abstract'] = text
            #df.loc[index, 'tables'] = ref_entries

df=df.drop(['pdf_json_files'], axis=1)
df=df.drop(['sha'], axis=1)
df.head()


# In[ ]:


# extract result excerpt
def extract_result(text,word,focus):
    extracted='N/A'
    if word in text and focus in text:
        res = [i.start() for i in re.finditer(word, text)]
        for result in res:
            extracted=text[result:result+600]
            if focus in extracted:
                return extracted
    return 'N/A'

# extract method excerpt
def extract_method(text,word,focus):
    extracted='N/A'
    if word in text:
        res = [i.start() for i in re.finditer(word, text)]
        for result in res:
            extracted=text[result:result+600]
    return extracted

# extract study design
def extract_design(text):
    words=['retrospective','prospective cohort','retrospective cohort', 'systematic review',' meta ',' search ','case control','case series,','time series','cross-sectional','observational cohort', 'retrospective clinical','virological analysis','prevalence study','literature','two-center']
    study_types=['retrospective','prospective cohort','retrospective cohort','systematic review','meta-analysis','literature search','case control','case series','time series analysis','cross sectional','observational cohort study', 'retrospective clinical studies','virological analysis','prevalence study','literature search','two-center']
    extract=''
    res=''
    for word in words:
        if word in text:
            res = [i.start() for i in re.finditer(word, text)]
        for result in res:
            extracted=text[result-30:result+30]
            extract=extract+' '+extracted
    i=0
    study=''
    for word in words:
        if word in extract:
            study=study_types[i]
        #print (extract)
        i=i+1
    return study

# BERT pretrained question answering module
def answer_question(question,text,model,tokenizer):
    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
    input_ids = tokenizer.encode(input_text)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    # show qeustion and text
    #tokenizer.decode(input_ids)
    answer=answer.replace(' ##','')
    #print (answer)
    return answer

def process_text(df1,focus):
    df_results = pd.DataFrame(columns=['Date', 'Study', 'Study Link','Journal', 'Study Type', 'Method', 'Result', 'Measure of Evidence', 'Added on', 'DOI', 'CORD_UID'])
    for index, row in df1.iterrows():
        method=extract_method(row['abstract'],'method',focus)
        result=extract_result(row['abstract'],'conclusion',focus)
        if result!='N/A':
            study_type=''
            study_type=extract_design(row['abstract'])
            ### get sample size
            sample_q='how many patients cases studies were included collected or enrolled'
            sample=row['abstract'][0:1000]
            moe=answer_question(sample_q,sample,modelqa,tokenizer)
            if '[SEP]' in moe or '[CLS]' in moe:
                moe='-'
            moe=moe.replace(' , ',',')
            
            link=row['doi']
            linka='https://doi.org/'+link
            to_append = [row['publish_time'], row['title'], linka,row['journal'], study_type, method, result, moe, '-', row['doi'], row['cord_uid']]
            df_length = len(df_results)
            df_results.loc[df_length] = to_append
    return df_results

focuses=['immune response','mutat','virus adaptations','phenotyp', 'genetic variation','transmission model', 'serial interval','assessment framework']


for focus in focuses:
    df1 = df[df['abstract'].str.contains(focus)]
    df1 = df[df['abstract'].str.contains('method')]
    df1 = df[df['abstract'].str.contains('conclusion')]
    df_results=process_text(df1,focus)
    df_results=df_results.sort_values(by=['Date'], ascending=False)
    df_table_show=HTML(df_results.to_html(escape=False,index=False))
    display(HTML('<h1>'+focus+'</h1>'))
    display(df_table_show)
    file=focus+'.csv'
    df_results.to_csv(file,index=False)

