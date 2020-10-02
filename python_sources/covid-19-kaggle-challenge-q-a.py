#!/usr/bin/env python
# coding: utf-8

# **Acknowledgement: These results are carried out at Smart City Lab,NCAI, NED University of Engineering and Technology**

# **This Notebook is an attempted to show and know about COVID-19 risk factors.The reason of attepting this challenge to take a part in it and contribute on the COVID-19 Open Research Dataset challenge and to help the research community and also the people who wants to find the answers.******

# Attempted this [Task](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558) as a contribution to the community!!
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Here, I'm working with NL tool kit and playing with the sentences.
# 

# In[ ]:


pip install nltk


# In[ ]:


import nltk
#nltk.download() #use this when you want to download.


# In[ ]:


pip install gensim


# In[ ]:


pip install pattern


# In[ ]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


# In[ ]:


sentence = [("a", "DT"),("clever","JJ"),("fox","NN"),("was","VBP"),   
            ("jumping","VBP"),("over","IN"),("the","DT"),("wall","NN")]


# In[ ]:


print(sentence)
grammar = "NP:{<DT>?<JJ>*<NN>}"
print(grammar)
parser_chunking = nltk.RegexpParser(grammar)
print(parser_chunking)
#parser_chunking.parse(sentence)


# ## Words Tokenization and Post tagging
# few things you need to understand why, where, when to used.
# 1. tokenization is a python's NLTK package that breaks the sentences into chunks.
# 2. it is used when you want to break and know the parts of speech. 

# In[ ]:


sentence2="""Its me Tabarka Rajab"""
tokens = nltk.word_tokenize(sentence2)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
tagged[0:1]


# ### Stopwords
# it uses the english words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence

# In[ ]:


from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))
for word in tokens: 
    if word not in en_stops:
        print(word)


# ###Search a particular word. 

# In[ ]:


import nltk

file = open('../input/nltk-fil/nltk_file.txt', 'r')
read_file = file.read()
text = nltk.Text(nltk.word_tokenize(read_file))

match = text.concordance('diagnose')


# In[ ]:


get_ipython().system('pip install pypdf2')
get_ipython().system('pip install textract')


# Extracting through a Single PDF file.

# In[ ]:


import PyPDF2
import nltk
#open the pdf file
open_file=open('../input/researchpaper/Researh_Paper_version2.pdf','rb')
#print(open_file)
object_file = PyPDF2.PdfFileReader(open_file)
# get number of pages
NumPages = object_file.getNumPages()
outlines=object_file.getDocumentInfo()
print("total number of pages this research paper has:",NumPages)
print("Document information:",outlines)
text = nltk.Text(nltk.word_tokenize((str(object_file))))
print(text)
match = text.concordance(str('Network'))
print(match)


# In[ ]:


get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')


# ### Now working with the dataset files
# * Here i'm going to extract the data from a CSV file through which we will see the text according to our questions.
# * keep your libraries upto-date.

# In[ ]:


import pandas as pd
import scispacy
import spacy
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


nlp = spacy.load("en_core_sci_sm")
#read the csv file from kaggle covid-19 challenge
meta = pd.read_csv("../input/dataset/metadata.csv")


# #### Creating dictionary and play with the TQDM.
# 
# * Tqdm in a short for taqadum in arabic, which means progress.

# In[ ]:


vector_dict = {}
for sha, abstract in tqdm(meta[["sha","abstract"]].values):
    if isinstance(abstract, str):
        vector_dict[sha] = nlp(abstract).vector


# In[ ]:


#initialized the variables keys and values, stored them into the empyty distionary.
keys = list(vector_dict.keys())
values = list(vector_dict.values())
Linguistics = []


# In[ ]:


cosine_sim_matrix = cosine_similarity(values, values)


# In[ ]:


n_sim_articles = 5
#hash file to generate a string file to identify the file.
input_sha = "e3b40cc8e0e137c416b4a2273a4dca94ae8178cc"


# In[ ]:


sha_index = keys.index(input_sha)
sim_indexes = np.argsort(cosine_sim_matrix[sha_index])[::-1][1:n_sim_articles+1]
sim_shas = [keys[i] for i in sim_indexes]
meta_info = meta[meta.sha.isin(sim_shas)]


# In[ ]:


print("-------ABSTRACT-----")
print(meta[meta.sha == input_sha]["abstract"].values[0])


# In[ ]:


print(f"----TOP {n_sim_articles} SIMILAR ABSTRACTS-----")
for abst in meta_info.abstract.values:
    print(abst)
    print("---------")


# ### Now to pass a query/question in order to get the answers.

# In[ ]:


query_statement = input("")


# In[ ]:


#number of top answers 
number_of_return = 5
#passing a query statement here
def passing_argument(query_statement):
    query_vector = nlp(query_statement).vector
    cosine_sim_matrix_query = cosine_similarity(values, query_vector.reshape(1,-1))
    query_sim_indexes = np.argsort(cosine_sim_matrix_query.reshape(1,-1)[0])[::-1][:number_of_return]
    query_shas = [keys[i] for i in query_sim_indexes]
    meta_info_query = meta[meta.sha.isin(query_shas)]
#print(passing_argument(abstract,values))
    print(f"----TOP {number_of_return} SIMILAR ABSTRACTS From a Question-----")
    for abst in meta_info_query.abstract.values:
        print(abst)
        #Linguistics.append(item['text'])
        print("---------")


# ### TASK 2: What do we know about COVID-19 risk factors?
# * this is going to be a answers to the asked questions on kaggle covid-19 task 2 challenge.

# #### Question 1

# In[ ]:


passing_argument("What are the covid-19 risk factors if you do smoking and having pre-existing pulmonary diseases ?")


# Question 2

# In[ ]:


passing_argument("Does the Co-infections and other co-morbidities effect human body in Covid-19?")


# Question 3

# In[ ]:


passing_argument("chances of covid-19 in pregnant women and impact on newborns")


# Question 4

# In[ ]:


passing_argument("What are the Socio-economic and behavioral factors to understand the economic impact of the virus and differences")


# Question 5

# In[ ]:


passing_argument("what are the transmission dynamics,reproductive number, incubation period, serial interval, modes of transmission and environmental factors of the covid-19 virus")


# Question 6
# 

# In[ ]:


passing_argument("how much Severity of disease,risk of fatality among symptomatic hospitalized patients, and high-risk patient groups")


# Question 7

# In[ ]:


passing_argument("Susceptibility of populations in covid-19")


# Question 8

# In[ ]:


passing_argument("what are the Public health mitigation measures effective for control risk factor of covid-19")

