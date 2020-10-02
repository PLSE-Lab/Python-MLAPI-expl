#!/usr/bin/env python
# coding: utf-8

# Created by a [TransUnion](http://www.transunion.com) data scientist that believes that information can be used to change our world for the better. #InformationForGood

# Task Details
# What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# 
# Specifically, we want to know what the literature reports about:
# 
# * Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.
# * Prevalence of asymptomatic shedding and transmission (e.g., particularly children).
# * Seasonality of transmission.
# * Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
# * Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
# * Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
# * Natural history of the virus and shedding of it from an infected person
# * Implementation of diagnostics and products to improve clinical processes
# * Disease models, including animal models for infection, disease and transmission
# * Tools and studies to monitor phenotypic change and potential adaptation of the virus
# * Immune response and immunity
# * Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
# * Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
# * Role of the environment in transmission

# # Question & Answering Model
# 
# Adapted from - https://www.kaggle.com/jonathanbesomi/a-qa-model-to-answer-them-all
# 
# The 3 important aspects of this are - *question*, *context*, *answer*
# 
# Model input - question & context 
# 
# Model output - answer

# For a question & answering model you need a dataset to train the model and the most popular one is SQuAD - The Stanford Question Answering Dataset. 
# 
# For context, we use a search engine to filter the data and get a reduced set of relevant context. Search engine used is Okapi BM25. This is a ranking function used to estimate the relevance of documents given a search query. This ranking function is based on bag-of-words retrieval function.
# 
# BERT (Bidirectional Encoder Representation from Transformers) has shown great performance for a multitude of Natural Language Understanding tasks. The BERT pre-trained model on the SQuAD dataset is used here to obtain the answers based on the question and context.
# 

# # Load dataset & libraries

# In[ ]:


"""
Libraries
"""

get_ipython().system('pip install rank_bm25 -q')

import numpy as np
import pandas as pd 
from pathlib import Path, PurePath

import nltk
from nltk.corpus import stopwords
import re
import string
import torch

from rank_bm25 import BM25Okapi # Search engine


# In[ ]:


"""
Load metadata df
"""

input_dir = PurePath('../input/CORD-19-research-challenge')
metadata_path = input_dir / 'metadata.csv'
metadata_df = pd.read_csv(metadata_path, low_memory=False)
metadata_df = metadata_df.dropna(subset=['abstract', 'title'])                             .reset_index(drop=True)


# In[ ]:


from rank_bm25 import BM25Okapi
nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = list(set(stopwords.words('english')))


# # Covid Search Engine
# 
# The main method prepocesses the metadata (title+abstract) and indexes the data. Once indexed, similar papers can be searched using the search(question) function.

# In[ ]:


class CovidSearchEngine:
    """
    Simple CovidSearchEngine.
    """
    
    def remove_special_character(self, text):
        #Remove special characters from text string
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        # tokenize text
        words = nltk.word_tokenize(text)
        return list(set([word for word in words 
                         if len(word) > 1
                         and not word in english_stopwords
                         and not word.isnumeric() 
                        ])
                   )
    
    def preprocess(self, text):
        # Clean and tokenize text input
        return self.tokenize(self.remove_special_character(text.lower()))


    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus
        self.columns = corpus.columns
        
        raw_search_str = self.corpus.abstract.fillna('') + ' '                             + self.corpus.title.fillna('')
        
        self.index = raw_search_str.apply(self.preprocess).to_frame()
        self.index.columns = ['terms']
        self.index.index = self.corpus.index
        self.bm25 = BM25Okapi(self.index.terms.tolist())
    
    def search(self, query, num):
        """
        Return top `num` results that better match the query
        """
        # obtain scores
        search_terms = self.preprocess(query) 
        doc_scores = self.bm25.get_scores(search_terms)
        
        # sort by scores
        ind = np.argsort(doc_scores)[::-1][:num] 
        
        # select top results and returns
        results = self.corpus.iloc[ind][self.columns]
        results['score'] = doc_scores[ind]
        results = results[results.score > 0]
        return results.reset_index()


# In[ ]:


cse = CovidSearchEngine(metadata_df)


# # Question-Answering Model
# 
# Download the pre-trained BERT model and install the dependencies

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


"""
Download pre-trained QA model
"""

import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_SQUAD = 'bert-large-uncased-whole-word-masking-finetuned-squad'

model = BertForQuestionAnswering.from_pretrained(BERT_SQUAD)
tokenizer = BertTokenizer.from_pretrained(BERT_SQUAD)

model = model.to(torch_device)
model.eval()

print()


# The answer_question() function, based on the question and the context obtained using the search engine passed returns the answer

# In[ ]:


def answer_question(question, context):
    # answer question given question and context
    encoded_dict = tokenizer.encode_plus(
                        question, context,
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_tensors = 'pt'
                   )
    
    input_ids = encoded_dict['input_ids'].to(torch_device)
    token_type_ids = encoded_dict['token_type_ids'].to(torch_device)
    
    start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])
    answer = answer.replace('[CLS]', '')
    return answer


# # Task and questions
# 
# Dictionary to store task and questions

# In[ ]:


# adapted from https://www.kaggle.com/dirktheeng/anserini-bert-squad-for-semantic-corpus-search

covid_kaggle_questions = {
"data":[
          {
              "task": "What is known about transmission, incubation, and environmental stability?",
              "questions": [
                  "Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water?",
                  "How long is the incubation period for the virus?",
                  "Can the virus be transmitted asymptomatically or during the incubation period?",
                  "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV?",
                  "How long can the 2019-nCoV virus remain viable on common surfaces?"
              ]
          },
        {
              "task":  "transmission, incubation, and environmental stability - Part2",
              "questions": [
                  "How long are individual contagious after recovery?",
                  "How does incubation period vary with age?",
                  "How does incubation period vary with health status?",
                  "Can the virus be transmitted asymptomatically or during the incubation period to children?"
              ]
          },
        {
              "task":  "transmission, incubation, and environmental stability - Part3",
              "questions": [
                  "Is there a seasonality for transmission?",
                  "How does the virus persist on surfaces like copper, stainless steel or plastic?",
                  "What are the tools and studies to monitor phenotypic change and potential adaptation of the virus?",
                  "How effective are movement control strategies to prevent secondary transmission in health care and community settings?"
                  "How effective are personal protective equipment (PPE)?"
                  "What is the role of the environment in transmission?"
              ]
          }
      ]
}


# # Compute Answers
# 
# The get_result() function takes in the question and returns the context, answer and start & end index for the answer in the context which can be highlighted.
# 
# The questions in the dictionary are passed to the get_result() in a loop and the answers appended.

# In[ ]:


NUM_CONTEXT_FOR_EACH_QUESTION = 10


def get_all_context(query, num_results):
    # Return ^num_results' papers that better match the query
    
    papers_df = cse.search(query, num_results)
    return papers_df['abstract'].str.replace("Abstract", "").tolist()


def get_all_answers(question, all_contexts):
    # Ask the same question to all contexts (all papers)
    
    all_answers = []
    
    for context in all_contexts:
        all_answers.append(answer_question(question, context))
    return all_answers


def create_output_results(question, 
                          all_contexts, 
                          all_answers, 
                          summary_answer='', 
                          summary_context=''):
    # Return results in json format
    
    def find_start_end_index_substring(context, answer):   
        search_re = re.search(re.escape(answer.lower()), context.lower())
        if search_re:
            return search_re.start(), search_re.end()
        else:
            return 0, len(context)
        
    output = {}
    output['question'] = question
    output['summary_answer'] = summary_answer
    output['summary_context'] = summary_context
    results = []
    for c, a in zip(all_contexts, all_answers):

        span = {}
        span['context'] = c
        span['answer'] = a
        span['start_index'], span['end_index'] = find_start_end_index_substring(c,a)

        results.append(span)
    
    output['results'] = results
        
    return output

    
def get_results(question, 
                summarize=False, 
                num_results=NUM_CONTEXT_FOR_EACH_QUESTION,
                verbose=True):
    # Get results

    all_contexts = get_all_context(question, num_results)
    
    all_answers = get_all_answers(question, all_contexts)
    
    if summarize:
        # NotImplementedYet
        summary_answer = get_summary(all_answers)
        summary_context = get_summary(all_contexts)
    
    return create_output_results(question, 
                                 all_contexts, 
                                 all_answers)


# In[ ]:


all_tasks = []

for i, t in enumerate(covid_kaggle_questions['data']):
    print("Answering questions to task {}. ...".format(i+1))
    answers_to_question = []
    for q in t['questions']:
            answers_to_question.append(get_results(q, verbose=False))
    task = {}
    task['task'] = t['task']
    task['questions'] = answers_to_question
    
    all_tasks.append(task)

all_answers = {}
all_answers['data'] = all_tasks


# # Show Results
# 
# Define helper functions to better visualize.

# In[ ]:


from IPython.display import display, Markdown, Latex, HTML

def layout_style():
    style = """
        div {
            color: black;
        }
        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }
        .answer{
            color: #dc7b15;
        }
        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }      
        div.output_scroll { 
            height: auto; 
        }
    """
    return "<style>" + style + "</style>"

def dm(x): display(Markdown(x))
def dh(x): display(HTML(layout_style() + x))


# In[ ]:


def display_single_context(context, start_index, end_index):
    
    before_answer = context[:start_index]
    answer = context[start_index:end_index]
    after_answer = context[end_index:]

    content = before_answer + "<span class='answer'>" + answer + "</span>" + after_answer

    return dh("""<div class="single_answer">{}</div>""".format(content))

def display_question_title(question):
    return dh("<h2 class='question_title'>{}</h2>".format(question.capitalize()))


def display_all_contexts(index, question):
    
    def answer_not_found(context, start_index, end_index):
        return (start_index == 0 and len(context) == end_index) or (start_index == 0 and end_index == 0)

    display_question_title(str(index + 1) + ". " + question['question'].capitalize())
    
    # display context
    for i in question['results']:
        if answer_not_found(i['context'], i['start_index'], i['end_index']):
            continue # skip not found questions
        display_single_context(i['context'], i['start_index'], i['end_index'])

def display_task_title(index, task):
    task_title = "Task " + str(index) + ": " + task
    return dh("<h1 class='task_title'>{}</h1>".format(task_title))

def display_single_task(index, task):
    
    display_task_title(index, task['task'])
    
    for i, question in enumerate(task['questions']):
        display_all_contexts(i, question)


# In[ ]:


task = 1
display_single_task(task, all_tasks[task-1])


# In[ ]:


task = 2
display_single_task(task, all_tasks[task-1])


# In[ ]:


task = 3
display_single_task(task, all_tasks[task-1])


# # Export Results
# 
# Export the results to a json dump.

# In[ ]:


import json
with open("covid_kaggle_answer_from_qa.json", "w") as f:
    json.dump(all_answers, f)


# In[ ]:




