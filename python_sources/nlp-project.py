#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob # finds all pathnames matching a specified pattern
import json
import nltk

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.


# In[ ]:


# loading the metadata into a dataframe

metadata_path = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

# dtype indicates the type of the specified columns (all ids are strings)
metadata = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})

# head shows the first 5 rows of the dataframe
metadata.head()


# In[ ]:


# the index of the metadata will be the unique cord_uid of the papers

metadata.index = metadata['cord_uid']

metadata


# In[ ]:


# info gives a statistical overview of the dataframe
metadata.info()


# In[ ]:


metadata['abstract'].describe(include = 'all')

# statistics on the abstract column show us that there are some abstract duplicates
# the authors may have submitted their papers to more than 1 journal


# In[ ]:


# dropping the duplicate entries

metadata.drop_duplicates(['abstract'], inplace = True)
metadata.drop_duplicates(['cord_uid'], inplace = True)

metadata['abstract'].describe(include = 'all')


# In[ ]:


# drop all entries with null abstracts

metadata = metadata[metadata.abstract.notnull()]

metadata.info()

# now all our text entries have unique non-null abstracts


# In[ ]:


# data preprocessing

# step 1: specifying the language of each text entry

# installing the langdetect module
get_ipython().system('pip install langdetect')


# In[ ]:


metadata_index = []

for x in metadata.index:
    metadata_index.append(x)


# In[ ]:


from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory

# set seed
DetectorFactory.seed = 0

# holds the language label for each metadata entry
languages = []

# loop through each entry in the metadata and detect the language of its abstract
for x in tqdm(range(len(metadata_index))):
    
    try:
        lang = detect(metadata['abstract'][metadata_index[x]])
        
        languages.append(lang)
          
    except Exception as e:
        languages.append("unknown")


# In[ ]:


# create a dictionary with the number of occurances of each language in our dataset

languages_dict = {}

for lang in set(languages):
    languages_dict[lang] = languages.count(lang)
    
print(languages_dict) # the majority of papers are in english


# In[ ]:


# add a language column to our metadata dataframe
metadata['language'] = languages


# In[ ]:


# drop all entries that are not english

metadata = metadata[metadata['language'] == 'en']


# In[ ]:


# drop the language column

metadata = metadata.drop(['language'], axis = 1)


# In[ ]:


# install spacy; a package for natural language processing

get_ipython().system('pip install spacy')


# In[ ]:


# data preprocessing
# step 2: natural language processing

import spacy 


# In[ ]:


# stop words

from spacy.lang.en.stop_words import STOP_WORDS
import string

# generating lists of punctuations and stop words that are going to be eliminated from the text
punctuations = list(string.punctuation)
stopwords = list(STOP_WORDS)


# In[ ]:


# scientific papers have stopwords that don't contribute to the meaning but are not often used in everyday english
# therefore we need to add these stopwords to our existing list of stopwords

custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]

for word in custom_stop_words:
    if(word not in stopwords):
        stopwords.append(word)


# In[ ]:


from IPython.utils import io
with io.capture_output() as captured:
    get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')


# In[ ]:


import en_core_sci_lg # module for parsing scientific texts

parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000


# In[ ]:


# 1) reducing to lowercase
# 2) removing stop words and punctuation
# 3) lemmatization

from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 

metadata_index = []

for x in metadata.index:
    metadata_index.append(x)

# a new column that is going to carry out processed abstracts
metadata['processed_abstract'] = ''

for i in tqdm(range(len(metadata_index))):
    
    # put all sentences in the abstract into a list
    sentences = list(parser(metadata['abstract'][metadata_index[i]]).sents)

    for x in range(len(sentences)):
        mytokens = parser(str(sentences[x])) # parse each sentence into tokens

        # convert each token to lowercase 
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
        
        # do not include the token if it is a stopword or a punctuation
        mytokens = [word for word in mytokens if str(word) not in stopwords and word not in punctuations]
        
        # lemmatize token
        mytokens = [lemmatizer.lemmatize(str(word)) for word in mytokens]
        
        new_sentence = ""
        for token in mytokens:
            new_sentence = new_sentence + token + " "

        new_sentence = new_sentence[:len(new_sentence)-1]

        sentences[x] = new_sentence

    new_abstract = ""
    for sentence in sentences:
        new_abstract = new_abstract + sentence + ". "

    new_abstract = new_abstract[:len(new_abstract)-1]
    
    metadata['processed_abstract'][metadata_index[i]] = new_abstract


# In[ ]:


# find all papers that don't mention covid-19 in their abstracts

from tqdm import tqdm

# a list that will carry the ids of the papers that do not mention covid-19
not_covid = []

for i in tqdm(range(len(metadata_index))):
    
    sentences = list(parser(str(metadata['processed_abstract'][metadata_index[i]])).sents)

    myList = []
    for sentence in sentences:
        sentence = sentence[:len(sentence)-1]

        for word in sentence:
            myList.append(str(word))
    
    # check if the processed abstract has any mention of the words covid-19 or sars-cov-2 
    if 'covid-19' not in myList and 'sars-cov-2' not in myList:
        not_covid.append(str(metadata['cord_uid'][metadata_index[i]]))


# In[ ]:


# drop all entries that do not mention covid-19 from the metadata

for x in tqdm(range(len(not_covid))):
    try:
        metadata = metadata.drop(not_covid[x], axis = 0)
    except:
        print('cannot find id')


# In[ ]:


metadata_index = []

for index in metadata.index:
    metadata_index.append(index)


# In[ ]:


# write the new, filtered and processed metadata into a csv file
metadata.to_csv('filtered-processed-metadata.csv',index=False)


# In[ ]:


# define a method that generates a list of ngrams for a given text
# only works for unigrams, bigrams and trigrams

def generate_ngram(ngram, text):
 
    # ngrams are generated for each separate sentence in the text
    sentences = text.split(".")
    sentences = [sentence for sentence in sentences if sentence != '']
    
    myList = []
    for sentence in sentences:

        myTokens = sentence.split()
        myTokens = [token for token in myTokens if token != '']

        if(myTokens != []):
            if(ngram == 'unigram'):
                myList.extend(myTokens)
            elif(ngram == 'bigram' and len(myTokens) >= 2): # cannot generate a bigram for less than 2 words
                myList.append(list(nltk.bigrams(myTokens)))
            elif(ngram == 'trigram' and len(myTokens) >= 3): # cannot generate a trigram for less than 3 words
                myList.append(list(nltk.trigrams(myTokens)))
        
    return myList


# In[ ]:


# define a method that runs the unigrams, bigrams and trigrams against each abstract in the matadata
# in our implementation, abstracts that contain at least one of the unigrams, bigrams or trigrams are considered to be relevant

def get_relevant_uids(unigrams, bigrams, trigrams):
    
    # a list containing the unique cord_uids of entries that are considered relevant
    relevant_uids = []

    # prioritize entries that contain trigrams as they are less likely to be irrelevant
    if(trigrams != []):
        for i in tqdm(range(len(metadata_index))):

            myList = generate_ngram('trigram', str(metadata['processed_abstract'][metadata_index[i]]))

            for trigram in trigrams:
                for trigramsList in myList:
                    if trigram in trigramsList:
                        if str(metadata['cord_uid'][metadata_index[i]]) not in relevant_uids:
                            relevant_uids.append(str(metadata['cord_uid'][metadata_index[i]]))

    # prioritize bigrams over unigrams
    if(bigrams != []):
        for i in tqdm(range(len(metadata_index))):

            myList = generate_ngram('bigram', str(metadata['processed_abstract'][metadata_index[i]]))

            for bigram in bigrams:
                for bigramsList in myList:
                    if bigram in bigramsList:
                        if str(metadata['cord_uid'][metadata_index[i]]) not in relevant_uids:
                            relevant_uids.append(str(metadata['cord_uid'][metadata_index[i]]))

    # finally check for the availability of unigrams 
    if(unigrams != []):
        for i in tqdm(range(len(metadata_index))):

            myList = generate_ngram('unigram', str(metadata['processed_abstract'][metadata_index[i]]))

            for unigram in unigrams:
                if unigram in myList:
                    if str(metadata['cord_uid'][metadata_index[i]]) not in relevant_uids:
                        relevant_uids.append(str(metadata['cord_uid'][metadata_index[i]]))

    return relevant_uids    


# In[ ]:


# define a method that calculates the precision @ k that takes the list of relevant uids as input as well as k

def precision_at_k(relevant_documents, k):
    
    return len(relevant_documents)/k


# In[ ]:


# task 1
# Data on potential risks factors 
# Smoking, pre-existing pulmonary disease

# for all tasks, we attempt to focus more on the trigrams, and minimize bigrams and unigrams as they are more prone to give us irrelevant documents

task1_unigrams = ['smoking', 'smoke', 'smoker', 'tobacco', 'cigarettes', 'cigarette', 'asthma']

task1_bigrams = [('heavy', 'smoker')]

task1_trigrams = [('pre-existing', 'pulmonary', 'disease'), ('pre-existing', 'lung', 'disease'), 
                  ('pre-existing', 'respiratory', 'disease'), ('pre-existing', 'pulmonary', 'condition'),
                  ('pre-existing', 'lung', 'condition'), ('pre-existing', 'respiratory', 'condition'),
                  ('pre-existent', 'pulmonary', 'disease'), ('pre-existent', 'lung', 'disease'), 
                  ('pre-existent', 'respiratory', 'disease'), ('pre-existent', 'pulmonary', 'condition'),
                  ('pre-existent', 'lung', 'condition'), ('pre-existent', 'respiratory', 'condition'),
                  ('existing', 'pulmonary', 'disease'), ('existing', 'lung', 'disease'), 
                  ('existing', 'respiratory', 'disease'), ('existing', 'pulmonary', 'condition'),
                  ('existing', 'lung', 'condition'), ('existing', 'respiratory', 'condition')]


# In[ ]:


# task 2
# Data on potential risks factors 
# Co-existing respiratory/viral infections

task2_unigrams = ['co-infection', 'co-infections', 'co-morbidity', 'co-morbidities']

task2_bigrams = [('co-existing', 'infection'), ('co-existent', 'infection'), ('existing', 'infection')]

task2_trigrams = [('co-existing', 'viral', 'infection'), ('co-existing', 'lung', 'infection'),
                  ('co-extisting', 'respiratory', 'infection'),
                  ('existing', 'viral', 'infection'), ('existing', 'lung', 'infection'),
                  ('existing', 'respiratory', 'infection')]


# In[ ]:


# task 3
# Data on potential risks factors 
# Neonates and pregnant women

task3_unigrams = ['baby', 'neonate', 'newborn', 'newborns', 'infant', 
                'pregnant', 'pregnancy']

task3_bigrams = [('pregnant', 'woman'), ('pregnant', 'patient')]

task3_trigrams = []


# In[ ]:


# task 4
# Socio-economic and behavioral factors 
# to understand the economic impact of the virus and whether there were differences

task4_unigrams = []

task4_bigrams = [('socio-economic', 'factor'), ('socioeconomic', 'factor'), ('behavioral', 'factor'), ('economic', 'impact'), 
                 ('social', 'factor'), ('economic', 'factor'), ('social', 'impact')]

task4_trigrams = []


# In[ ]:


# task 5
# Transmission dynamics of the virus
# basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

task5_unigrams = []

task5_bigrams = [('transmission', 'dynamic'), ('reproductive', 'number'), ('incubation', 'period'), 
                 ('serial', 'interval'), ('environmental', 'factor'), ('mode', 'transmission')]

task5_trigrams = [('modes', 'of', 'transmission')]


# In[ ]:


# task 6
# Severity of disease 
# including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

task6_unigrams = ['severity', 'high-risk']

task6_bigrams = [('risk', 'fatality'), ('patient', 'risk'), ('higher', 'risk'), ('high-risk', 'patient'),
                 ('fatality', 'risk'), ('mortality', 'risk'), ('death', 'risk')]

task6_trigrams = [('severity', 'of', 'disease'), ('risk', 'of', 'fatality'), ('patient', 'at', 'risk'), 
                  ('symptomatic', 'hospitalized', 'patients'), ('high', 'risk', 'patient'),
                  ('high', 'fatality', 'risk')]


# In[ ]:


# task 7
# Susceptibility of populations

task7_unigrams = ['susceptibility']

task7_bigrams = [('susceptibility', 'populations'), ('susceptible', 'group'), ('susceptible', 'population'), 
                 ('high-risk', 'population'), ('vulnerable', 'population'), ('vulnerable', 'people')]

task7_trigrams = [('susceptibility', 'of', 'populations'), ('high', 'risk', 'population')]


# In[ ]:


# task 8
# Public health mitigation measures that could be effective for control

task8_unigrams = ['mitigation', 'containment']

task8_bigrams = [('mitigation', 'method'), ('mitigation', 'strategy'), 
                 ('health', 'strategy'), ('prevention', 'method')]

task8_trigrams = [('public', 'health', 'mitigation'), ('public', 'health', 'prevention'), ('public', 'health', 'strategy')]


# In[ ]:


# generate a list of cord_uids for task 1 that are retrieved by our system

task1_retrieved_uids = get_relevant_uids(task1_unigrams, task1_bigrams, task1_trigrams)


# In[ ]:


task1_retrieved_data = pd.DataFrame(task1_retrieved_uids, columns = ['cord_ui'])

task1_retrieved_data.to_csv('task1_retrieved_data', index=False)


# In[ ]:


task2_retrieved_uids = get_relevant_uids(task2_unigrams, task2_bigrams, task2_trigrams)


# In[ ]:


task2_retrieved_data = pd.DataFrame(task2_retrieved_uids, columns = ['cord_ui'])

task2_retrieved_data.to_csv('task2_retrieved_data', index=False)


# In[ ]:


task3_retrieved_uids = get_relevant_uids(task3_unigrams, task3_bigrams, task3_trigrams)


# In[ ]:


task3_retrieved_data = pd.DataFrame(task3_retrieved_uids, columns = ['cord_ui'])

task3_retrieved_data.to_csv('task3_retrieved_data', index=False)


# In[ ]:


task4_retrieved_uids = get_relevant_uids(task4_unigrams, task4_bigrams, task4_trigrams)


# In[ ]:


task4_retrieved_data = pd.DataFrame(task4_retrieved_uids, columns = ['cord_ui'])

task4_retrieved_data.to_csv('task4_retrieved_data', index=False)


# In[ ]:


task5_retrieved_uids = get_relevant_uids(task5_unigrams, task5_bigrams, task5_trigrams)


# In[ ]:


task5_retrieved_data = pd.DataFrame(task5_retrieved_uids, columns = ['cord_ui'])

task5_retrieved_data.to_csv('task5_retrieved_data', index=False)


# In[ ]:


task6_retrieved_uids = get_relevant_uids(task6_unigrams, task6_bigrams, task6_trigrams)


# In[ ]:


task6_retrieved_data = pd.DataFrame(task6_retrieved_uids, columns = ['cord_ui'])

task6_retrieved_data.to_csv('task6_retrieved_data', index=False)


# In[ ]:


task7_retrieved_uids = get_relevant_uids(task7_unigrams, task7_bigrams, task7_trigrams)


# In[ ]:


task7_retrieved_data = pd.DataFrame(task7_retrieved_uids, columns = ['cord_ui'])

task7_retrieved_data.to_csv('task7_retrieved_data', index=False)


# In[ ]:


task8_retrieved_uids = get_relevant_uids(task8_unigrams, task8_bigrams, task8_trigrams)


# In[ ]:


task8_retrieved_data = pd.DataFrame(task8_retrieved_uids, columns = ['cord_ui'])

task8_retrieved_data.to_csv('task8_retrieved_data', index=False)


# In[ ]:


# calculate precision @ k 
# k is the top 10 documents

# task 1
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task1_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# in an attempt to measure the performance of our nlp methods, we took the initiative of manually labeling the first 10 documents that are retrieved by our system

# top 10 retrieved documents:
# ['95fty9yi','6tso9our','8g70j0qw','1gnoo0us','t73oioqv','y778k3hs','5xyt8d5u','emwu5g5f','ygnxmcl1','ygnxmcl1','z0476b47','ubqexcof']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task1_top_10_retrieved = ['95fty9yi', '8g70j0qw', '1gnoo0us', 'y778k3hs', '5xyt8d5u', 'emwu5g5f']

# we realize that this method is very subjective, however, we only took the documents whose main subject was related to our question
# and since the documents aren't labeled, this is the method we resorted to for measurement


# In[ ]:


# precision @ k
task1_precision = precision_at_k(task1_top_10_retrieved, 10)

print(task1_precision)


# In[ ]:


# task 2
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task2_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents:
# ['vqqqip1m','b2vjrskm','fzpz9oaj','1igydu3y','rjqq1lxx','nzjz0q8b','bjykd232','2z5ixtk3','dqs21e0q','50iju57j']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task2_top_10_relevant = ['b2vjrskm', 'fzpz9oaj', '1igydu3y', 'bjykd232', '2z5ixtk3']


# In[ ]:


# precision @ k
task2_precision = precision_at_k(task2_top_10_relevant, 10)

print(task2_precision)


# In[ ]:


# task 3
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task3_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents:
# ['fce2kdzw', 'un1y8r5a', 'c27qh5y7', 's0vo7dlk', '5itjbp2j', '7xe3fvd5', 'vqz6b7zy', '9ae8qhcg', 'n6sq2jz9', 'wncv7qvm']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task3_top_10_relevant = ['fce2kdzw', 'un1y8r5a', 'c27qh5y7', '7xe3fvd5', '5itjbp2j', 'n6sq2jz9']


# In[ ]:


# precision @ k
task3_precision = precision_at_k(task3_top_10_relevant, 10)

print(task3_precision)


# In[ ]:


# task 4
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task4_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents:
# ['dg213uf6', 'ogfwlr4m', 'q114cle2', '1zyeusat', 'd7at4wp4', 's8fvqcel', 'p4q64ksa', 'jtwb17u8', '8ozauxlk', 'pm6ta0qy']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task4_top_10_relevant = ['ogfwlr4m', 'q114cle2', 's8fvqcel', '8ozauxlk']


# In[ ]:


# precision @ k
task4_precision = precision_at_k(task4_top_10_relevant, 10)

print(task4_precision)


# In[ ]:


# task 5
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task5_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents in our last run
# ['zph6r4il', 'lh3x6wat', 'jkyltjap', 'ohrqw3ac', '16rgt4ca', 'efk2jj50', 'dl7vrb5k', '4bzw76lt', 'zd8c1no7', 'vhphlccl']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task5_top_10_relevant = ['lh3x6wat', 'ohrqw3ac', 'efk2jj50', 'dl7vrb5k', 'zd8c1no7', 'vhphlccl']


# In[ ]:


# precision @ k
task5_precision = precision_at_k(task5_top_10_relevant, 10)

print(task5_precision)


# In[ ]:


# task 6
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task6_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents in our last run
# ['a2mqvbj2', '8g70j0qw', 'lh3x6wat', '4bzw76lt', 'v83fgykh', 'mez1k8t0', '142qz8of', 'o3b2yvbs', 'bj142e9b', 'xn8jzzsd']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task6_top_10_relevant = ['8g70j0qw', '4bzw76lt', 'v83fgykh', 'mez1k8t0', 'o3b2yvbs']


# In[ ]:


# precision @ k
task6_precision = precision_at_k(task6_top_10_relevant, 10)

print(task6_precision)


# In[ ]:


# task 7
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task7_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents in our last run
# ['1tkbwk8u', '41cqkmra', 'lp4nhezz', '810fmd8e', 'flzaqw4s', 'b4e3y9u3', 'po1bu4v3', '2ftw85xw', 'c9ecid5e', 'hkm8yspk']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task7_top_10_relevant = ['1tkbwk8u', '41cqkmra', 'lp4nhezz', '810fmd8e', '2ftw85xw', 'hkm8yspk']


# In[ ]:


# precision @ k
task7_precision = precision_at_k(task7_top_10_relevant, 10)

print(task7_precision)


# In[ ]:


# task 8
# the following are the top 10 retrieved documents (cord_uid + abstract)

for x in range(10):
    
    uid = task8_retrieved_uids[x]
    
    for i in range(len(metadata_index)):
        if(uid == metadata['cord_uid'][metadata_index[i]]):
            print(metadata['cord_uid'][metadata_index[i]])
            print(metadata['abstract'][metadata_index[i]])
            print()


# In[ ]:


# top 10 retrieved documents in our last run
# ['nnl1txhi', 'swc2pitd', 'syt4r964', 'qwngx01h', 'v6wt9mhh', '2w0zr9c0', 'gb9rkv9c', '8niqpwvc', '9iwiv5m4']

# the following are the relevant documents that we manually labeled from the top 10 retrieved entires:
task8_top_10_relevant = ['nnl1txhi', 'swc2pitd', 'qwngx01h', '8niqpwvc', '9iwiv5m4']


# In[ ]:


# precision @ k
task8_precision = precision_at_k(task8_top_10_relevant, 10)

print(task8_precision)


# In[ ]:




