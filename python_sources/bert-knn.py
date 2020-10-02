#!/usr/bin/env python
# coding: utf-8

# # Risk Factors of COVID-19

# ## Introduction
# In this notebook we have answered different questions related to the risk factors of COVID-19 using different machine learning techniques. The core techniques being used are BERT Sentence Embeddings and Approximate Nearest Neighbors. We learn the sentence embeddings of the titles of the papers and cluster them based on the given question. Finally we pick k titles from the cluster nearest to the given question.
# We used [xhlulu](https://www.kaggle.com/xhlulu)'s notebook [cord-19-eda-parse-json-and-generate-clean-csv](https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv) to read the json files and convert them to csv files. It saved us a lot of time.

# ### Data Preparation
# 1. We used  [cord-19-eda-parse-json-and-generate-clean-csv](https://www.kaggle.com/muhammadhassan/cord-19-eda-parse-json-and-generate-clean-csv) notebook to read the json files and convert them to csv files.
# 2. We used [combining-csvfiles](https://www.kaggle.com/massiq/combining-csvfiles) notebook to generate a single csv file of the dataset.

# ### Approach:
# 1. Learn the sentence embeddings of all titles using [bert-sentence-transformer](https://pypi.org/project/sentence-transformers/).
# 2. Use the [Annoy](https://github.com/spotify/annoy) library to build a forest of trees using title embeddings.
# 3. Save the index file of embeddings to the disk.
# 4. Learn the sentence embedding of the given question.
# 5. Load the index file from disk and find k nearest neighbours of questions from title embeddings.

# ### References:
# * [xhlulu/cord-19-eda-parse-json-and-generate-clean-csv](https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv)
# * [muhammadhassan/cord-19-eda-parse-json-and-generate-clean-csv](https://www.kaggle.com/muhammadhassan/cord-19-eda-parse-json-and-generate-clean-csv)

# ### Table of Contents:
# * [Install/Load Packages](#1)
# * [Import Libraries/Packages](#2)
# * [Loading the Data](#3)
# * [Using the GPU](#4)
# * [Downloading the Pretrained SentenceTransformer Model](#5)
# * [Extracting the Sentence Embeddings of all Titles](#6)
# * [Building the Trees of Title Embeddings Using Annoy and Saving the Index File](#7)
# * [Questions](#8)
# * [Loading the Annoy Index File](#9)
# * [COVID-19 and Pregnant Women](#10)
# * [COVID-19 and Neonates](#11)
# * [COVID-19 and Cancer Patients](#12)
# * [Age Groups Vulnerable to COVID-19](#13)
# * [Underlying Diseases in COVID-19 Patients](#14)
# * [COVID-19 and Social Distaning](#15)
# * [Psychological Effects of COVID-19 on Medical Staff](#16)
# * [Controlling the Spread of COVID-19](#17)
# * [Public Health Measures for Control of COVID-19](#18)
# * [Economic Impacts of COVID-19](#19)
# * [Transmission Dynamics of COVID-19](#20)
# * [Public Measures to Control the Spread of COVID-19](#21)

# ### Install/Load Packages <a id= 1></a>

# In[ ]:


get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip install annoy')
get_ipython().system('pip install bert-extractive-summarizer')


# ### Import Libraries/Packages <a id= 2></a>

# In[ ]:


from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import pandas as pd
import torch
import os
import warnings
warnings.filterwarnings("ignore")


# ### Loading the Data <a id= 3></a>

# In[ ]:


input_file = '/kaggle/input/combining-csvfiles/combined_dataset.csv'
data = pd.read_csv(input_file)
data = data.fillna('')
title = data['combined_title']
abstract = data['abstract']
text = data['text']
paper_id = data['paper_id']
authors = data['authors']
print(data.info())


# ### Using the GPU <a id= 4></a>

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ### Downloading the Pretrained SentenceTransformer Model <a id= 5></a>

# In[ ]:


sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_model.to(device)
print()


# ### Extracting the Sentence Embeddings of all Titles <a id= 6></a>

# In[ ]:


sentence_embeddings = sentence_model.encode(title.values.tolist())
print('Dimensionality of the embeddings: ', len(sentence_embeddings[0]))


# ### Building the Trees of Title Embeddings Using Annoy and Saving the Index File <a id= 7></a>

# In[ ]:


embed_dim = 768
tree = AnnoyIndex(embed_dim, "dot")

for i, vec in enumerate(sentence_embeddings):
    tree.add_item(i, vec)

tree.build(20)
tree.save('/kaggle/working/titles_bert_emb.ann')
del sentence_embeddings[:]


# ### Questions <a id= 8></a>

# In[ ]:


questions = {
    'q_1' : ["what are the effects of COVID-19 or coronavirus on pregnant women?"],
    'q_2' : ["what are the effects of COVID-19 or coronavirus on new born babies?"],
    'q_3' : ['what are the effects of COVID-19 or coronavirus on cancer patients?'],
    'q_4' : ['Which age group is more vulnerable to covid-19?'],
    'q_5' : ['what are most common underlying diseases in covid-19 patients?'],
    'q_6' : ['What are the effects of social distancing?'],
    'q_7' : ['What are the psychological effects of covid-19 on medical staff?'],
    'q_8' : ['What are the control strategies to curtail transmission of covid-19?'],
    'q_9' : ['what are the public health mitigation measures that could be effective for control of covid-19?'],
    'q_10' : ['What are the economic and behavioral impacts of covid-19 pandemic or coronavirus, what are different socio-economic and behavioral factors arised as a result of covid-19 that can affect economy? What is the difference between groups for risk for COVID-19 by education level? by income? by race and ethnicity? by contact with wildlife markets? by occupation? household size? for institutionalized vs. non-institutionalized populations (long-term hospitalizations, prisons)?'],
    'q_11' : ['what are the transmission dynamics of the covid-19, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors?'],
    'q_12' : ['what are the public measures to control the spread of covid-19?']
}


# ### Loading the Annoy Index File <a id= 9></a>

# In[ ]:


tree = AnnoyIndex(embed_dim, 'dot')
tree.load('/kaggle/working/titles_bert_emb.ann')


# ### COVID-19 and Pregnant Women <a id= 10></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_1']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_1'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'pregnant_women.csv'
output.to_csv(output_file, index=False)


# ### COVID-19 and Neonates <a id= 11></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_2']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_2'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'neonates.csv'
output.to_csv(output_file, index=False)


# ### COVID-19 and Cancer Patients <a id= 12></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_3']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_3'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'cancer_patients.csv'
output.to_csv(output_file, index=False)


# ### Age Groups Vulnerable to COVID-19 <a id= 13></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_4']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_4'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'vulnerable_groups.csv'
output.to_csv(output_file, index=False)


# ### Underlying Diseases in COVID-19 Patients <a id= 14></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_5']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_5'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'underlying_diseases.csv'
output.to_csv(output_file, index=False)


# ### COVID-19 and Social Distaning <a id= 15></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_6']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_6'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'social_distancing.csv'
output.to_csv(output_file, index=False)


# ### Psychological Effects of COVID-19 on Medical Staff <a id= 16></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_7']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_7'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'medical_staff.csv'
output.to_csv(output_file, index=False)


# ### Controlling the Spread of COVID-19  <a id= 17></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_8']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_8'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'controlling_spread.csv'
output.to_csv(output_file, index=False)


# ### Public Health Measures for Control of COVID-19 <a id= 18></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_9']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_9'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'public_health_measures.csv'
output.to_csv(output_file, index=False)


# ### Economic Impacts of COVID-19 <a id= 19></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_10']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_10'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'economic_impacts.csv'
output.to_csv(output_file, index=False)


# ### Transmission Dynamics of COVID-19 <a id= 20></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_11']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_11'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'transmission_dynamics.csv'
output.to_csv(output_file, index=False)


# ### Public Measures to Control the Spread of COVID-19 <a id= 21></a>

# In[ ]:


output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])
question = questions['q_12']
question_emb = sentence_model.encode(question)
title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results
print('QUESTION: ',questions['q_12'][0])
title_list = title.values.tolist()
for i, o in enumerate(title_output):
  print('-------')
  print(i)
  print('Title: ',title_list[o])
  print('Paper Id: ', paper_id[o])
  print('Authors: ', authors[o])
  print('\n')
  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])
  output = output.append(df, ignore_index=True)

output_file = 'public_measures_controlling_spread.csv'
output.to_csv(output_file, index=False)


# In[ ]:




