#!/usr/bin/env python
# coding: utf-8

# # Our Team
# 
# This submission is the result of a collaboration between the following:
# 
# * Catherine Doyle ([Catherine Doyle](https://www.kaggle.com/cddata))
# * John Doyle ([JohnDoyle](https://www.kaggle.com/johndoyle))
# * Keith Finnerty ([KeithFinnerty](https://www.kaggle.com/ketihf))
# * Piyush Rumao ([PIYUSH RUMAO](https://www.kaggle.com/piyushrumao))
# 
# We are all members of the Data Insights team in Deutsche Bank. Sitting within the Chief Data Office, the Data Insights team is the global Centre of Excellence for data science and data analytics within Deutsche Bank. Mostly based in Dublin, Ireland with some team members sitting in London and other locations, the Data Insights team comprises approximately forty people and includes data scientists, data analysts, data engineers, data architects, data visualization experts, and programme managers. We have expertise in areas such as advanced analytics, artificial intelligence, machine learning, natural language processing, visualization, dashboards, and software development.  We engage with teams across all areas and functions of the Bank, partnering with them to design and deliver analytics solutions that leverage the large amount of available data in order to create value for the Bank and our clients. Engagements vary between time-boxed Proofs of Concept and longer-term projects, and are conducted using the Scaled Agile Framework (SAFe).  As a Centre of Excellence, we also work to uplift data science and analytics across the Bank, for example by fostering Communities of Practice on different topics and rolling out governance support and resources around best practices for model development and analytics delivery.
# 
# We decided to collaborate on this Kaggle challenge and pool our knowledge and skills with the aim of making a positive impact in the on-going struggle against COVID-19.

# # Key Highlights:
# 
# This notebook is an extension to previous work done on [word embedding](https://www.kaggle.com/piyushrumao/word-embedding-analysis-on-covid-19-dataset) and will be using that as reference to explore detailed findings to discover more relevant research papers that would answer the questions which are presented as Tasks in this competition. Inorder to do that we will be going through following steps:
# 
# 1. Pre-process data and generate combined dataframe from it
# 
# 2. Train Doc2Vec models (currently done externally and added in notebook)
# 
# 3. Use vectors of Doc2Vec model (all input training data) to train a Unsupervised K-Nearest Neighbour model
# 
# 4. Create list of tasks (questions that we need answers for)
# 
# 5. Pass this list to Doc2Vec model first to generate vectors from it and then use this vectors to find its nearest neighbours from the trained KNN model.
# 
# 6. The result will be most relevant documents for our given question.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import re
import errno
import json
import pickle
import glob
import multiprocessing
from time import time  # To time our operations

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
np.random.seed(2018)

from gensim import corpora, models
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.tokenize import word_tokenize

from sklearn.neighbors import NearestNeighbors


import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda

import warnings
warnings.filterwarnings('ignore')

print('starting execution')


# # Data Cleaning and Preprocessing

# In[ ]:


def clean(txt):
    """
    Basic string loading code.

    :param txt:
    :return:
    """
    txt = re.sub(r'.\n+', '. ', txt)  # replace multiple newlines with period
    txt = re.sub(r'\n+', '', txt)  # replace multiple newlines with period
    txt = re.sub(r'\[\d+\]', ' ', txt)  # remove reference numbers
    txt = re.sub(' +', ' ', txt)
    txt = re.sub(',', ' ', txt)
    txt = re.sub(r'\([^()]*\)', '', txt)
    txt = re.sub(r'https?:\S+\sdoi', '', txt)
    txt = re.sub(r'biorxiv', '', txt)
    txt = re.sub(r'preprint', '', txt)
    txt = re.sub(r':', ' ', txt)
    return txt.lower()


# In[ ]:


class document():
    def __init__(self, file_path):
        if file_path:
            with open(file_path) as file:
                data = json.load(file)
                self.paper_id = data['paper_id']
                self.title = data['metadata']['title']
                self.abstract_tripples = {}
                self.text_tripples = {}
                self.key_phrases = ""
                self.abstract = ""
                self.text = ""
                self.entities = {}
                if 'abstract' in data:
                    for section in data['abstract']:
                        self.abstract = self.abstract + "\n" + section["text"]

                for section in data['body_text']:
                    self.text = self.text + "\n" + section['text']

    def clean_text(self):
        self.abstract = clean(self.abstract)
        self.text = clean(self.text)
        self.title =clean(self.title)
        final_data_dict = self.combine_data()
        return final_data_dict

    def combine_data(self):
        self.data = {'paper_id': self.paper_id,
                     'title': self.title,
                     'abstract': self.abstract,
                     'text': self.text,
                     'abstract_tripples': self.abstract_tripples,
                     'text_tripples': self.text_tripples,
                     'key_phrases': self.key_phrases,
                     'entities': self.entities}
        return self.data

    def extract_data(self):

        self.paper_id = self.data['paper_id']
        self.title = self.data['title']
        self.abstract = self.data['abstract']
        self.text = self.data['text']
        self.abstract_tripples = self.data['abstract_tripples']
        self.text_tripples = self.data['text_tripples']
        self.key_phrases = self.data['key_phrases']
        self.entities = self.data['entities']

    def save(self, dir):
        self.combine_data()

        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(dir, 'w') as json_file:
            json_file.write(json.dumps(self.data))

    def load_saved_data(self, dir):
        with open(dir) as json_file:
            self.data = json.load(json_file)
        self.extract_data()


# In[ ]:


desired_dirs=['/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset',
             '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',
             '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',
             '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license']

exp_dir = ['/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv']

noncomm_use_subset=[]
biorxiv_medrxiv = []
comm_use_subset = []
custom_license = []

for individual_dirs in desired_dirs:
    files_list = []
    data = []
    print('############## Directory Working on #################')
    print(individual_dirs)
    print(individual_dirs.split('/')[-1])
    for dirname,_, filenames in os.walk(individual_dirs):
        #print(dirname)
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            files_list.append(os.path.join(dirname, filename))
    print(len(files_list))
    #print(files_list)
    i=0
    for individual_file in files_list:
        try:
            pub = document(individual_file)
            data_dict = pub.clean_text()
            #print(data_dict)
            data.append(data_dict)
            i+=1
        except:
            pass
    print('Now writing back data')
    print('files processed===>'+str(i))
    with open(individual_dirs.split('/')[-1]+'.pickle', "wb") as f:
                pickle.dump(data,f)
    


# In[ ]:


total_dataframe = pd.DataFrame()
models = glob.glob('/kaggle/working/' + "*.pickle")
#models = glob.glob('/kaggle/working/biorxiv_medrxiv.pickle')
# print('models via glob===>'+str(models))
for individual_model in models:
    print(individual_model)
    # open a file, where you stored the pickled data
    file = open(individual_model, 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    print('Showing the pickled data:')
    my_df = pd.DataFrame(data)
    print(my_df.shape)
    my_df = my_df.replace(r'^\s*$', np.nan, regex=True)
    total_dataframe = total_dataframe.append(my_df, ignore_index=True)
    print(my_df.isnull().sum())


# In[ ]:


print('Total data in dataframe')
total_dataframe = total_dataframe.drop(['abstract_tripples', 'text_tripples','entities','key_phrases'], axis=1)
print(total_dataframe.shape)
total_dataframe.head()


# In[ ]:


total_dataframe.isnull().sum()


# In[ ]:


total_dataframe.dtypes


# In[ ]:


tf_df = pd.DataFrame()
tf_df['merged_text'] = total_dataframe['title'].astype(str) +  total_dataframe['abstract'].astype(str) +  total_dataframe['text'].astype(str)
tf_df['paper_id'] = total_dataframe['paper_id']
print(tf_df.shape)
tf_df.head()


# # Training Doc2Vec Model

# In[ ]:


def read_corpus(df, column, tokens_only=False):
    """
    Arguments
    ---------
        df: pd.DataFrame
        column: str 
            text column name
        tokens_only: bool
            wether to add tags or not
    """
    for i, line in enumerate(df[column]):

        tokens = gensim.parsing.preprocess_string(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
            
"""
train_corpus = (list(read_corpus(tf_df, 'merged_text')))
# using distributed memory model
model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=60, min_count=5, epochs=20, seed=42, workers=6)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.save('dm_doc2vec.model')
"""
model_loaded_dm = Doc2Vec.load('/kaggle/input/doc2vec-models/dm_doc2vec.model')


# # Tasks that needs answers

# In[ ]:


def get_doc_vector(doc):
    tokens = gensim.parsing.preprocess_string(doc)
    vector = model_loaded_dm.infer_vector(tokens)
    return vector



Task_1 = "What is known about transmission, incubation, and environmental stability?"
Task_2 = "What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?"
Task_3 = "What do we know about virus genetics, origin, and evolution?"
Task_4 = "What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?"
Task_5 = "What do we know about the effectiveness of non-pharmaceutical interventions?"
Task_6 = "What do we know about diagnostics and surveillance?"
Task_7 = "What has been published about medical care?"
Task_8 = "What has been published concerning ethical considerations for research?"
Task_9 = "What has been published about information sharing and inter-sectoral collaboration?" 
list_of_tasks = [Task_1,Task_2,Task_3,Task_4,Task_5,Task_6,Task_7,Task_8,Task_9]

abstract_vectors = model_loaded_dm.docvecs.vectors_docs
array_of_tasks = [get_doc_vector(task) for task in list_of_tasks]


# In[ ]:


tf_df['abstract_vector'] = [vec for vec in abstract_vectors]
tf_df.head()


# In[ ]:


train_array = tf_df['abstract_vector'].values.tolist()


# In[ ]:


len(train_array)


# # Training Unsupervised Nearest Neighbours on vectors generated from Doc2Vec

# In[ ]:


ball_tree = NearestNeighbors(algorithm='ball_tree', leaf_size=20).fit(train_array)


# In[ ]:


distances, indices = ball_tree.kneighbors(array_of_tasks, n_neighbors=3)


# In[ ]:


print(total_dataframe.shape)
print(tf_df.shape)


# # View Findings

# In[ ]:


for i, info in enumerate(list_of_tasks):
    print("="*80, f"\n\nTask = {info[:100]}\n", )
    df =  total_dataframe.iloc[indices[i]]
    text = df['text']
    titles = df['title']
    dist = distances[i]
    for l in range(len(dist)):
        try:
            print(f" Text index = {indices[i][l]} \n Distance = {distances[i][l]} \n Title: {titles.iloc[l]} \n Text extract: {text.iloc[l][:200]}\n\n")
        except Exception as e:
            print('Error Occured==>'+str(e))

