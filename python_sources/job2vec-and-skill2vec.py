#!/usr/bin/env python
# coding: utf-8

# **Import libraries**

# In[ ]:


import numpy as np
import pandas as pd
import gzip
import seaborn as sns
import os
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')

data_dir = '../input/linkedin-crawled-profiles-dataset' #change path where data is save
profiles_path = os.path.join(data_dir, 'linkedin.json/linkedin.json')
print(os.listdir(data_dir))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# **Load data**

# In[ ]:


import json as js
localities = []
industries = []
specialities = []
interests = []
occupations = []
companies = []
majors = []
institutionsList = []

for l in open(profiles_path):
    line = js.loads(l)
    
    #check education exist and from education append the name of Institutions in institutionList
    if 'education' in line: 
        institutionsList.append([exp['name'] for exp in line['education'] if 'name' in exp])
        
    #check locality exist and append locality in localities list
    if 'locality' in line: 
        localities.append([line['locality']])
        
    #check education exist and from education append major in majors list
    if 'education' in line: 
        majors.append([edu['major'] for edu in line['education'] if 'major' in edu])
        
    #check industry exist and append industry in industries list 
    if 'industry' in line: 
        industries.append([line['industry']])
        
    #check specilities exist and append specilities in specilities list
    if 'specilities' in line: 
        specialities.append([s.strip() for s in line['specilities'].split(',')])
    
    #check interests exist and append interests in interests list
    if 'interests' in line: 
        interests.append([s.strip() for s in line['interests'].split(',')])
        
    #check experience exist and from experiance append occupations title in occupations list
    if 'experience' in line: 
        occupations.append([exp['title'] for exp in line['experience'] if 'title' in exp])
        
    #check education exist and from experiance append companies in companies list
    if 'experience' in line:  
        companies.append([exp['org'] for exp in line['experience'] if 'org' in exp])
    


# **View Of Data**

# In[ ]:


#See how data present in list
print(institutionsList[:20],"\n")
print(localities[:20],"\n")
print(industries[:20],"\n")
print(specialities[:20],"\n")
print(interests[:20],"\n")
print(occupations[:20],"\n")
print(companies[:20],"\n")
print(majors[:20])


# **School Model**

# In[ ]:


# model school2vec
school2vec = Word2Vec(institutionsList, min_count=10, iter=100, workers=32)
institutions = list(school2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(institutions[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(school2vec['Columbia University'])


# In[ ]:


# sanity check
print('Columbia University: ', school2vec.wv.most_similar(['Columbia University']))


# **Localities Model**

# In[ ]:


# model localities2vec
localities2vec = Word2Vec(localities, min_count=10, iter=100, workers=32)
localities = list(localities2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(localities[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(localities2vec['United States'])


# In[ ]:


# sanity check
print('United States: ', localities2vec.wv.most_similar(['United States']))


# **Industries  Model**

# In[ ]:


# model industries2vec
industries2vec = Word2Vec(industries, min_count=10, iter=100, workers=32)
industries = list(industries2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(industries[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(industries2vec['Biotechnology'])


# In[ ]:


# sanity check
print('Biotechnology: ', industries2vec.wv.most_similar(['Biotechnology']))


# **Specialities Model**

# In[ ]:


# model specialities2vec
specialities2vec = Word2Vec(specialities, min_count=10, iter=100, workers=32)
specialities = list(specialities2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(specialities[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(specialities2vec['Internet Marketing'])


# In[ ]:


# sanity check
print('Internet Marketing: ', specialities2vec.wv.most_similar(['Internet Marketing']))


# **Interests Model**

# In[ ]:


# model interests2vec
interests2vec = Word2Vec(interests, min_count=10, iter=100, workers=32)
interests = list(interests2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(interests[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(interests2vec['nanotechnology'])


# In[ ]:


# sanity check
print('nanotechnology: ', interests2vec.wv.most_similar(['nanotechnology']))


# **Occupations Model**

# In[ ]:


# model occupations2vec
occupations2vec = Word2Vec(occupations, min_count=10, iter=100, workers=32)
occupations = list(occupations2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(occupations[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(occupations2vec['Senior Scientist'])


# In[ ]:


# sanity check
print('Senior Scientist: ', occupations2vec.wv.most_similar(['Senior Scientist']))


# **Companies Model**

# In[ ]:


# model companies2vec
companies2vec = Word2Vec(companies, min_count=10, iter=100, workers=32)
companies = list(companies2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(companies[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(companies2vec['Albert Einstein Medical Center'])


# In[ ]:


# sanity check
print('Albert Einstein Medical Center: ', companies2vec.wv.most_similar(['Albert Einstein Medical Center']))


# **Majors Model**

# In[ ]:


# model majors2vec
majors2vec = Word2Vec(majors, min_count=10, iter=100, workers=32)
majors = list(majors2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab

print(majors[:100]) ## print the first 100 vocabularies from the list


# In[ ]:


print(majors2vec['Computer Science'])


# In[ ]:


# sanity check
print('Computer Science: ', majors2vec.wv.most_similar(['Computer Science']))


# In[ ]:




