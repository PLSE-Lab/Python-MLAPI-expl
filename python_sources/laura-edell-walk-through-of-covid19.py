#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# Getting started with COVID19 dataset along with several deep learning techniques like unsupervised learning, to help beginner's or advanced ML experts better understand the data provided to us for this Kaggle competition. <br>
# <br>
# Broadcast on https://aka.ms/datasciencediva on March 20, 2020
# <br>
# <br>
# * What is Sars-Cov2 exactly? How is it different from other <b>viruses?</b>
# * Dive into the COVID19 dataset
# * Analysis and Data Visualization of newly, cleansed dataset
# <br>
# <br>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input director


# # 1. What is Sars-CoV2?
# 
# <br>
# What is Coronavirus?
# <br><br>
# The coronavirus is a family of viruses that can cause a range of illnesses in humans including common cold and more severe forms like SARS and MERS which are life-threatening. The virus is named after its shape which takes the form of a crown with protrusions around it and hence is known as coronavirus.
# <br> https://youtu.be/aerq4byr7ps - Video explaining what exactly is Sars-Cov2
# <br><br>
# How Did the recent Outbreak occur?
# <br><br>
# The recent outbreak of coronavirus is believed to have occurred in a market for illegal wildlife in the central Chinese city of Wuhan. 

# Here is a very infromative video about Coronavirus from <a href="https://www.youtube.com/channel/UCsXVk37bltHxD1rDPwtNM8Q">Kurzgesagt </a>
# 

# In[ ]:


from IPython.display import IFrame, YouTubeVideo
YouTubeVideo('BtN-goy9VOY',width=600, height=400)


# # 2. Current Situation and Available Resources On Sars-CoV2

# 
# Here is a list of Resources available on Kaggle to understand the current situation and also to complete the tasks associated with this new Dataset:
# * https://coronavirus.jhu.edu/map.html  --> Interactive Day by Day Analysis of Outbreak By John Hopkins University
# * https://www.healthmap.org/covid-19/
# * https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset --> Day level Information on spread
# * https://www.kaggle.com/imdevskp/covid-19-analysis-viz-prediction-comparisons --> Comprehensive Kernel On visualization
# * https://www.kaggle.com/paultimothymooney/coronavirus-genome-sequence --> Coronavirus Genome Sequence Dataset
# * https://www.kaggle.com/paultimothymooney/coronavirus-genome-sequence --> Repository of CoronaVirus Genomes
# * https://www.kaggle.com/jamzing/sars-coronavirus-accession --> Exploration of Sars Mutation

# # 3. Understanding and Cleaning the Dataset
# 

# In[ ]:


laura_meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')


# In[ ]:


laura_meta_data.head()


# 
# * Source - It is one of the most important columns of dataset its understanding is critical for our analysis
#            CHZ - It stands for CHen Zuckerberg Initiative
#            PMC - PUBMed Central is a free full text archieve of biomedical journal literature At US NIH. More can be learned here - https://en.wikipedia.org/wiki/PubMed_Central
#            BiorVix And MedrVix - are preprint server for biology which are online platform for publication of paperby Cold Spring Harbour University
# * Sha - 17K of the paper records have PDFs and the hash of the PDFs are in 'sha'
# * title - Contains the title of papers
# * DOI - A DOI, or Digital Object Identifier, is a string of numbers, letters and symbols used to permanently identify an article or document and link to it on the web.More can be learned here - https://en.wikipedia.org/wiki/Digital_object_identifier
# * pmcid and pubmed_id - https://nexus.od.nih.gov/all/2015/08/31/pmid-vs-pmcid-whats-the-difference/
# * Licence - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4845398/  This articles explains licensing and copyright for scholary articles. 
# Rest of the columns can be understood by reading the readme file in the datset

# In[ ]:


laura_meta_data.info()


# In[ ]:


laura_meta_data.describe(include='all')


# So after intial analysis of our metadata file we can see the following things:-
# * There are only 24654 unique titles that means we have duplicate files
# * doi.org is missing from some of the values in doi columns 
# * We have articles from 1732 unique SCHOLARLY ARTICLES

# Now Let's go ahead and drop duplicate and preprocess doi column

# In[ ]:


laura_meta_data.drop_duplicates(subset=['title'], inplace=True)
laura_meta_data.info()


# In[ ]:


def doi_url(d):
    if d.startswith('http://'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'


# In[ ]:


laura_meta_data.doi = laura_meta_data.doi.fillna('').apply(doi_url)
laura_meta_data.head()


# Exploring the missing values

# In[ ]:


laura_meta_data.isna().sum()


# * We have 2290 missing abstract values .
# * We also have 1 missing title - Will Drop

# 
# <br><br> Well you can see that we have titles,abstracts and authors available but not the real text of the paper which contains all the information. Once we form the dataset from Json files we can merge both dataset.
# Let's make it interesting by munging it all together.

# Now I will not analyze metadata dataframe because we have beautiful kernels written for the same . I will provide links to them :-
# * https://www.kaggle.com/docxian/cord-19-metadata-evaluation written by Chris X
# <br><br>
# Next Extracting a csv file from all the Json's Files and merge of 2 DF to nget a final working dataFrame

# # Extracting Information From JSON Files
# 
# So using pd.read_json() won't work here because the underlying structure of the json files are very complex.
# <br> The underlying structure can be seen from file name json.schema present in the dataset directory .Thus to extract information from the json files we will have to write our own customized pipeline . Thankfully for us this work has already been done by two great people :-
# * https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv --> You can go through this kernel to learn how to build a pipeline to parse this huge amount of data
# * https://www.kaggle.com/fmitchell259/create-corona-csv-file --> From here you can learn how to get a dataframe out of this specific data . The former kernel is more comprehensive and great for learning. This kernel however just lets you easily get a csv file of all the json files.
# We will use teh 2nd one.

# In[ ]:


import numpy as np
import pandas as pd
import os
import json
import glob
import sys
sys.path.insert(0, "../")


# In[ ]:


## CODE to CHECK JSON STRUCTURE
file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/252878458973ebf8c4a149447b2887f0e553e7b5.json'
with open(file_path) as json_file:
     json_file = json.load(json_file)
json_file['metadata']


# # Analysis of Final Data
# You can add output of https://www.kaggle.com/fmitchell259/create-corona-csv-file this kernel as  ("File" -> "Add or upload data" -> "Kernel Output File" tab -> search the name of this notebook).

# In[ ]:


# metadata
laura_meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

laura_meta_data.drop_duplicates(subset=['sha'], inplace=True)

def doi_url(d):
    if d.startswith('http://'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'

laura_meta_data.doi = laura_meta_data.doi.fillna('').apply(doi_url)


# In[ ]:


Json = pd.read_csv("../input/create-corona-csv-file/kaggle_covid-19_open_csv_format.csv")
Json = Json.iloc[1:, 1:].reset_index(drop=True)

# merge frames
cols_to_use = laura_meta_data.columns.difference(Json.columns)
all_data = pd.merge(Json, laura_meta_data[cols_to_use], left_on='doc_id', right_on='sha', how='left')

all_data.title = all_data.title.astype(str) # change to string, there are also some numeric values

all_data.head(2)


# Below are some of the best written kernels so far , for this datset:
# * https://www.kaggle.com/maksimeren/covid-19-literature-clustering ---> Explains all the things that can be done using this data
# * https://www.kaggle.com/ratan123/cord-19-understanding-papers-with-textanalytics --> Some very great visualization for the data
# * https://www.kaggle.com/danielwolffram/topic-modeling-finding-related-articles --> Example of how topic modeling is done
# * https://www.kaggle.com/shahules/eda-find-similar-papers-easily --> Kernel exploring EDA and clustering
# 
# Here are some important discussions which will help you gain more insights into data:
# * https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/137256
# * https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/137041
# * https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/136935
# * https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/136907

#  I will publish more exploring and applying different NLP and unsupervised techniques very soon.

# <b>If You liked my efforts , please upvote my kernel :) Super excited to help and work with you! 
