#!/usr/bin/env python
# coding: utf-8

# COVID-19 Open Research Dataset (CORD-19) Analysis
# ======
# 
# COVID-19 Open Research Dataset (CORD-19) is a free resource of scholarly articles, aggregated by a coalition of leading research groups, about COVID-19 and the coronavirus family of viruses. The dataset can be found on [Semantic Scholar](https://pages.semanticscholar.org/coronavirus-research) and there is a research challenge on [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
# 
# BERT - Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks: https://github.com/google-research/bert, We use derivation of BERT model as BioBERT to come up with answers for set of questions listed in the challenge, First step is to use sentence embeddings by using FastText in gensim and come up with context for questions.
# 
# 
# Second step is to use that context and feed set of questions and their respective context in BioBERT model

# ### Installing Gensim library which will be used to load FastText embeddings for Sentence vectors to find context, 
# 
# Note: need to maintain version as 3.8.0 had compaitibility issues

# In[ ]:


get_ipython().system(' pip install gensim==3.4.0')


# In[ ]:


import os
import pandas as pd
from gensim.models.fasttext import FastText as FT_gensim
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from uuid import uuid4
import torch
import re
import json
from tqdm import tqdm
import datetime
import pprint
import random
import string
import sys


# ### Indexing all 45,000 research papers into pandas dataframe with Title, Abstract and Full text body

# In[ ]:


# main folder of Covid19 Dataset
dirs = ["biorxiv_medrxiv", "comm_use_subset", "custom_license", "noncomm_use_subset"]
docs = []
base_path = "/kaggle/input/CORD-19-research-challenge"
for d in dirs:
    for file in tqdm(os.listdir(f"{base_path}/{d}/{d}/pdf_json")):
        file_path = f"{base_path}/{d}/{d}/pdf_json/{file}"
        json_file = json.load(open(file_path,"rb"))
        
        title = json_file["metadata"]["title"]
        try : 
            abstract = "\n\n".join([text["text"] for text in json_file["abstract"]])
        except : 
            abstract = ""
        full_text = "\n\n".join([text["text"] for text in json_file["body_text"]])
        docs.append([title, abstract, full_text])
# Pandas Dataframe containing the title, abstract and body text
papers_df = pd.DataFrame(docs, columns = ["title", "abstract", "full_text"])


# ### Eliminate empty papers

# In[ ]:


papers_df = papers_df.dropna()
papers_df


# ### Upload Pretrained FastText Embeddings Model

# In[ ]:


# Upload the model 
model_load_name = 'final_model_gensim.pt'
path = F"/kaggle/input/similaritymodels2new/Similarity/FinalModel/{model_load_name}"
model = FT_gensim.load(path)


# ### Function to campare similary between input query and each paper title or paper abstract

# In[ ]:


def token_similarity(token1, token2): 
    """
    calculate similarity between sentences based on their embeddings
        ----------------
    Args : 
        token1 : String
        token1 : String
        ---------------
    returns:
        float between 0 and 100, representing the percentage of similarity
    """
    try :
        token1 = re.sub('[^a-zA-z0-9\s]', '' , token1).lower()
        token2 = re.sub('[^a-zA-z0-9\s]', '' , token2).lower()
        return model.similarity(token1, token2)
    except : 
        return 0

def get_context(query, search_on, model = model, df = papers_df):
    """
    maps similarity function for given query to either all paper abstracts or to all paper titles to extract the closes paper to answer the query.
        ----------------
    Args : 
        query : String
        search_on : String
        model : Gensim model Object
        df : pandas Dataframe
        ---------------
    returns:
        Tuple containing full_text, title and similarity degree to closeset paper to query.
    """
    
    if search_on in ["title", "abstract"]:
        df["similarity_to_query"] = df[search_on].apply(lambda x : token_similarity(x, query))
        result = df.nlargest(1, ['similarity_to_query']).reset_index(drop = True)
        return result["full_text"][0].replace("\n", " "), result["title"][0], result["similarity_to_query"][0]
    else :
        raise Exception("search_on argument should be in ['title', 'abstract']")


# ### Test the query

# In[ ]:


query = "what are risk factors COVID-19?"
search_on = "title"
import time
t1 = time.time()
context = get_context(query, search_on, model, papers_df)
t2 = time.time()
print(f"query took {t2-t1} seconds")
print(context)


#  # BioBERT - BERT model trained on corpus: 
#  ## This repository provides the code for fine-tuning BioBERT, a biomedical language representation model designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, and question answering, Github code: https://github.com/dmis-lab/biobert
# ### We have used question answering side of BioBERT model to find answer in COVID-19 Challenge

# ### Task-1 Questions to be answered

# In[ ]:



question_list = """ What is known about transmission, incubation, and environmental stability
What do we know about natural history, transmission, and diagnostics for the virus 
What have we learned about infection prevention and control
What is Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.
What is Prevalence of asymptomatic shedding and transmission (e.g., particularly children).
What is Seasonality of transmission.
What is Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
What is Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
What is Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
What is Natural history of the virus and shedding of it from an infected person
What is Implementation of diagnostics and products to improve clinical processes
What is Disease models, including animal models for infection, disease and transmission
What is Tools and studies to monitor phenotypic change and potential adaptation of the virus
What is Immune response and immunity
What is Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
What is Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
What is Role of the environment in transmission
"""

question_list = question_list.split("\n")
qa_dataframe = pd.DataFrame({"Questions": question_list})


# Extracting context from the data ingestion module above using Fast Text model

# In[ ]:


qa_dataframe["Context"] = qa_dataframe["Questions"].apply(lambda x : get_context(x, "title", model, papers_df)[0])


# In[ ]:


qa_dataframe


# ## BIOBERT Fine-tuning and Prediction

# ### Since BERT model was trained using tf1.x, we can not use transfer learning of using pre-trained weights on tensorflow 2.0 so need to installing Tensorflow version 1.x

# In[ ]:


get_ipython().system('pip install tensorflow==1.15.2')


# In[ ]:


import tensorflow as tf
print(tf.version)


# TPU support is not available for Tensorflow version 1.x in Kaggle.

# # Test Data preparation

# In[ ]:


covidQuestions = qa_dataframe


# Create temp directory to store temporary files

# In[ ]:


get_ipython().system(' mkdir temp')


# In[ ]:


covidQuestions.to_json('temp/QnAInput.json')


# Format the dataframe to create a JSON as input to BIOBERT QnA model

# In[ ]:


#Test File

import json
from uuid import uuid4

def restructure_json_Test(input_data):  
    
    """
    Restructure input data to be compatible with BIOBERT Qna test data format.
        ----------------
    Args : 
        input_data : String
        ---------------
    returns:
        formatted data in dictionary
    """
    output = dict()
    data = []
    paragraphs = []
    for i in input_data["Questions"].keys():
        qas = []
        answers = []
      
        qas.append({"id" : str(uuid4()),
                   "question" : input_data["Questions"][i],
                   })
        paragraphs = paragraphs + [{"qas" : qas, "context" : input_data["Context"][i]}]
    data = data + [{"paragraphs" : paragraphs, "title" : "BioASQ6b"}]
    output["data"] = data
    output["version"] = "BioASQ6b"
    return output


# In[ ]:


with open('temp/QnAInput.json',encoding='ISO-8859-1' ) as f:
    input_data = json.load(f)


# Save the restructured data in JSON format

# In[ ]:


outputData = restructure_json_Test(input_data)
with open('temp/QnAInputTest.json', 'w',  encoding='ISO-8859-1') as json_file:
    json.dump(outputData, json_file)


# Verify all files used for prediction

# In[ ]:


get_ipython().system(" ls -l '/kaggle/input/biobertcode/GitRepo/BIOBERT/bioasq-biobert/run_factoid.py'")
get_ipython().system(" ls -l '/kaggle/input/biobertconfig/biobertconfig/vocab.txt'")
get_ipython().system(" ls -l '/kaggle/input/biobertconfig/biobertconfig/bert_config.json'")
get_ipython().system(" ls -l '/kaggle/input/biobertmodel2/BERT-pubmed-1000000-SQuAD2/model.ckpt-14470.index'")
get_ipython().system(" ls -l 'temp/QnAInputTest.json'")


# # Prediction using Fine Tuned BIOBERT model
# 
# BioBERT model has been uploaded to https://www.kaggle.com/varshnes/biobertmodel2

# In[ ]:


get_ipython().system(' python /kaggle/input/biobertcode/GitRepo/BIOBERT/bioasq-biobert/run_factoid.py      --do_train=False      --do_predict=True      --vocab_file=/kaggle/input/biobertconfig/biobertconfig/vocab.txt      --bert_config_file=/kaggle/input/biobertconfig/biobertconfig/bert_config.json      --init_checkpoint=/kaggle/input/biobertmodel2/BERT-pubmed-1000000-SQuAD2/model.ckpt-14470.index      --max_seq_length=512      --max_answer_length=200      --train_batch_size=12      --learning_rate=5e-6      --doc_stride=128      --num_train_epochs=1.0      --do_lower_case=False      --train_file=$BIOASQ_INPUT_DIR/BioASQ-6b/train/Full-Abstract/BioASQ-train-factoid-6b-full-annotated.json      --predict_file=temp/QnAInputTest.json      --output_dir=/kaggle/output/kaggle/factoid_output/prediction/')


# Check the prediction file in output folder

# In[ ]:


ls -l /kaggle/output/kaggle/factoid_output/prediction/predictions.json


# In[ ]:


with open('/kaggle/output/kaggle/factoid_output/prediction/predictions.json',encoding='ISO-8859-1' ) as f:
    output_data = json.load(f)


# In[ ]:


output_data


# In[ ]:


with open('temp/QnAInputTest.json',encoding='ISO-8859-1' ) as f:
    output_data_questions = json.load(f)


# ### Lets go through Task 2 Questions 1 by 1 and predictions are mapped to output side by side answers

# In[ ]:


paragraphs = output_data_questions['data'][0]['paragraphs']

for index, item in enumerate(paragraphs):
    id = item['qas'][0]['id']
    question = item['qas'][0]['question']
    answer = output_data[id]
    print("Question ", str(index + 1), " : " , question ,"\n Answer : ", answer)
    print("\n")


# In[ ]:




