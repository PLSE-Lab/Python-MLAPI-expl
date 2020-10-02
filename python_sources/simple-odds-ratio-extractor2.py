#!/usr/bin/env python
# coding: utf-8

# # Simple Odds Ratio Extractor
# 
# The script below will show you a quick and easy way to extract odds ratios from full text articles or abstracts (if in the abstract) for your selected risk factor.  In this example we will use hypertension. Hope it helps - please up vote if it does.
# 
# An odds ratio (OR) is a measure of association between an exposure and an outcome. The OR represents the odds that an outcome will occur given a particular exposure, compared to the odds of the outcome occurring in the absence of that exposure. Odds ratios are most commonly used in case-control studies, however they can also be used in cross-sectional and cohort study designs as well (with some modifications and/or assumptions).[read more](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/)
# 
# In the CORD19 documents OR will often be presented in the text of the document something like this
# 
# hypertension [or=2.3 [95% ci (1.76, 3.00), p<0.01]
# 
# Here is an example of the hypertension OR data from the Kaggle.com contibutions page so you can see what we are looking to accomplish. [see OR example](https://www.kaggle.com/covid-19-contributions#Hypertension)
# 
# Ok let's get started and load the required python packages.

# In[ ]:


import pandas as pd
import numpy as np
import functools
import re
print ('packages loaded')


# # Next we load the full text articles from the CORD19 dataset but keep only those documents related to COVID-19

# In[ ]:


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

# load the meta data from the CSV file using 3 columns (abstract, title, authors),
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file'])
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


for index, row in df.iterrows():
    if ';' not in row['sha'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json')==True:
        with open('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json') as json_file:
            data = json.load(json_file)
            body=format_body(data['body_text'])
            keyword_list=['TB','incidence','age']
            #print (body)
            body=body.replace("\n", " ")
            text=row['abstract']+' '+body.lower()

            df.loc[index, 'abstract'] =text

df=df.drop(['full_text_file'], axis=1)
df=df.drop(['sha'], axis=1)
df.head()


# # Now we will "focus" the dataframe to make sure it has hypertension in the text - leaving 640 articles that mention hypertension

# In[ ]:


focus='hypertension'
df1 = df[df['abstract'].str.contains(focus)]
print (focus,'focused articles',df1.shape)


# # Now we will iterate through the focused dataframe and pass the full text (or absract when no full text) to the extract ratios function.
# The function works as follows:
# - the text and the term we want the OR for (hypertension) is passed to the function
# - using re.finditer, every starting location (integer) of the word hypertension is found in the text and returned as a list
# - then the list containing the starting location is looped through and a str from the starting location of hypertension +75 is returned
# - The the str is tested to see if it contains important characters that lead to finding OR or other important ratios.
# - if '95' in extracted or 'odds ratio' in extracted or 'p>' in extracted or '=' in extracted or 'p<' in extracted or '])' in extracted or '(rr' in extracted:
# - only the strs meeting the requirments are included and passed back.
# 
# Finally a table is created with the publicaitons and the ratios
# 
# *Note - one drawback, if the ratio data is not found in the text, the system ignores the document and some documents may only have the ratios in a table.*
# 
# # Please use it and improve it, if it helps - upvote if you find it useful.

# In[ ]:


from IPython.core.display import display, HTML

def extract_ratios(text,word):
    extract=''
    if word in text:
        res = [i.start() for i in re.finditer(word, text)]
    for result in res:
        extracted=text[result:result+75]
        #print (extracted)
        #if '95' in extracted or 'odds ratio' in extracted or 'p>' in extracted or '=' in extracted or 'p<' in extracted or '])' in extracted or '(rr' in extracted:
        if '95%' in extracted or 'odds ratio' in extracted or '])' in extracted or '(rr' in extracted or '(ar' in extracted or '(hr' in extracted or '(or' in extracted:
            extract=extract+' '+extracted
    #print (extract)
    return extract

focus='hypertension'
df_results = pd.DataFrame(columns=['date','study','link','extracted'])
for index, row in df1.iterrows():
    extracted=extract_ratios(row['abstract'],focus)
    if extracted!='':
        link=row['doi']
        linka='https://doi.org/'+link
        to_append = [row['publish_time'],row['title'],linka,extracted]
        df_length = len(df_results)
        df_results.loc[df_length] = to_append

df_results=df_results.sort_values(by=['date'], ascending=False)
df_table_show=HTML(df_results.to_html(escape=False,index=False))
display(df_table_show)

