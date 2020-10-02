#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter 

from os import listdir
from os.path import isfile, join
import json
from collections import Counter
import geopandas 
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        break
    
    break 
# Any results you write to the current directory are saved as output.

# Corona = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
covResearch = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
paperName = covResearch[['title', 'cord_uid', 'pmcid','pubmed_id']]

#https://www.geeksforgeeks.org/python-pandas-series-str-find/
# dropping null value columns to avoid errors 
paperName.dropna(inplace = True) 
  
# substring to be searched 
sub1 = 'Temperature'
sub2 = 'temperature'


# Alternatively: sub = 'emperature'
start = 2
  
# creating and passsing series to new column 
paperName["Indexes"]= paperName["title"].str.find(sub1, start) + paperName["title"].str.find(sub2, start) 
# display
paperName = paperName.drop(paperName[paperName.Indexes < 0].index)
  
paperName 


# In[ ]:


# https://www.kaggle.com/diamazov/export-usa-names-into-csv
    
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
paper_names = bq_helper.BigQueryHelper(active_project="Temperature-Research-Papers-Covid", dataset_name="paperNames")

# query and export data 
query = """SELECT title, pmcid, Indexes as number FROM `Temperature-Research-Papers-Covid` GROUP BY title, pmcid, Indexes"""
paper_namesfile = paper_names.query_to_pandas_safe(query)
paper_namesfile.to_csv("researchPapersonTemp.csv")


# Papers that has the word temperatures in their title. This might not be fully encompass an entire document but it does highlight papers that focus on the correlation between COVID-19 and temperature. 

# In[ ]:


def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


# In[ ]:


def removeNestings(l): 
    for i in l: 
        if type(i) == list: 
            removeNestings(i) 
        else: 
            output.append(i)  


# In[ ]:


DATA_PATH = "../input/CORD-19-research-challenge/"
folder_paths = {"biorxiv_medrxiv/biorxiv_medrxiv/", "comm_use_subset/comm_use_subset/", "custom_license/custom_license/", "noncomm_use_subset/noncomm_use_subset/"} 
pdf_pmc = {"pdf_json", "pmc_json"}
p = "pmc_json"
pmcID = paperName[['pmcid']]
pmc_list = pmcID.values.tolist()

onlyfiles = []
data_list = []

for fp in folder_paths:
    if (p in listdir(DATA_PATH+fp)):
        onlyfiles = [f for f in listdir(DATA_PATH+fp+p) if isfile(join(DATA_PATH+fp+ p, f))]
        for n in onlyfiles:
            paper_id = re.sub('.xml.json$', '', n)
            #for y in pmc_list:
            if any(paper_id in s for s in pmc_list):
                with open(f'{DATA_PATH}{fp}pmc_json/{n}') as json_data:
                    this_file = json.load(json_data)
                data_list.append(extract_values(this_file, "text"))
                paper_text = data_list
        
    
output = []
removeNestings(data_list)       
#for l in this_file:
                    #if (string pmcID in l):
                        #data_list.append(l)
                        


# Wordclouds

# In[ ]:


#Wordloud: https://www.datacamp.com/community/tutorials/wordcloud-python
#Stopwords: https://programminghistorian.org/en/lessons/counting-frequencies#removing-stop-words
from wordcloud import WordCloud, STOPWORDS # Making wordcloud
from PIL import Image    # Imports PIL module, to see and scan images  

import matplotlib.pyplot as plt 

#def list2string(l2s):
#    textSTR = " "   #making the text into string
#    return (textSTR.join(l2s))

paper_textSTR = ' '.join(map(str, paper_text)) 


filteredWords = set(STOPWORDS)
filteredWords.update(["may","Figure", "Fig", "et al", "showed", "one", "two", "three", "mean", "table", "using", "well", "result", "model"])
                      
wordcloud = WordCloud(stopwords=filteredWords,
                          background_color='white', 
                      max_words=300
                         ).generate(paper_textSTR)


plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Top 

# Counter

# In[ ]:


#https://stackoverflow.com/questions/20723133/remove-a-list-of-stopwords-from-a-counter-in-python
#https://stackoverflow.com/questions/56558006/counter-that-returns-the-10-most-common-words

# This didnt work, Ignore this but an attempt was made and I might want to make it work in the future. this function is not in the paper 

def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

wordCounter = "".join(map(str, paper_textSTR))

wordCounterSTR = removeStopwords(wordCounter, filteredWords)
print(wordCounterSTR)

# split() returns list of all the words in the string 
split_it = wordCounterSTR.split() 
  
# Pass the split_it list to instance of Counter class. 
Counter = Counter(split_it) 

# most_common() produces k frequently encountered 
# input values and their respective counts. 
 = Counter.most_common(10) 
  
#print(most_occur) 


# In[ ]:


DATA_PATH = "../input/CORD-19-research-challenge/"
folder_paths = {"biorxiv_medrxiv/biorxiv_medrxiv/", "comm_use_subset/comm_use_subset/", "custom_license/custom_license/", "noncomm_use_subset/noncomm_use_subset/"} 
pdf_pmc = {"pdf_json", "pmc_json"}

onlyfiles = []
data_list = []

for fp in folder_paths:
    for p in pdf_pmc:
        if (p in listdir(DATA_PATH+fp)):
            onlyfiles = [f for f in listdir(DATA_PATH+fp+p) if isfile(join(DATA_PATH+fp+ p, f))]
            for n in onlyfiles:
                with open(f'{DATA_PATH}{fp}{p}/{n}') as json_data:
                    this_file = json.load(json_data)

                this_file = extract_values(this_file, 'title')
                for l in this_file:
                    if ("Effect of temperature and relative humidity on" in l):
                        data_list.append(l)
                        print(data_list)


output = []
removeNestings(data_list)

