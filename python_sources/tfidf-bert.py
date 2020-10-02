#!/usr/bin/env python
# coding: utf-8

# # TFIDF - BERT Question Answer Model
# 
# Our submission answers all the task questions by utilising a pretrained BERT Question Answer model - [bert-large-uncased-whole-word-masking-finetuned-squad](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad).
# 
# This work is a collaborative effort between [Nick Sorros](https://www.kaggle.com/nsorros), [Antonio Campello](https://www.kaggle.com/campello) and [Liz Gallagher](https://www.kaggle.com/lizgal).
# 
# 
# ## Methodology
# 
# For each question in the task list we find the 5 more relevant 2020 papers to the question text using TFIDF vectors. Next we split up the text from these papers and apply the pretrained model to predict the answer to the question in each of the text chunks. We output the highest scoring answer found. Thus for each question we output the top answer found for each of the 5 most relevant papers.
# 
# ### Data Filters
# 
# 1. We only use papers which are from 2020.
# 2. We only use English language papers. Languages were predicted by applying the [langdetect](https://pypi.org/project/langdetect/) language detection library to the abstract text in the metadata.
# 3. We deduplicate any papers which had the same pmcid.
# 
# ### Evaluation Data
# 
# We have tagged 150 answers (5 from each question) as to whether we though they were the best answer or not the best answer from this paper, or whether the paper was not relevant to this question.
# 
# Using this data we can calculate 2 metrics of how well the model performs on the answers - the relevance and the good answer ratio:
# 
# relevance = `(best_answer + not_best_answer) / (best_answer + not_best_answer + not_relevant)`
# 
# and
# 
# Good answer ratio = `best_answer / (best_answer + not_best_answer)`
# 
# 
# ## Results
# 
# ### Results version 1
# 
# Data: The data wasn't filtered.
# 
# Model: Predictions were made by separating chunks of text by fullstops.
# 
# | Task | Number of questions | Relevance | Good answer ratio | 
# | --- | --- | --- | --- |
# | What is known about transmission, incubation, and environmental stability? | 5 | 65.22% | 40.00 |
# | What do we know about COVID-19 risk factors? | 8 | 85.00% | 52.94 |
# | What do we know about virus genetics, origin, and evolution? | 5 | 52.63% | 10.00 |
# | What do we know about vaccines and therapeutics? | 2 | 70.00% | 57.14 |
# | What do we know about non-pharmaceutical interventions? | 2 | 85.71% | 66.67 |
# | What has been published about medical care? | 4 | 66.67% | 30.00 |
# | What do we know about diagnostics and surveillance? | 1 | 75.00% | 100.00 |
# | Other interesting questions | 3 | 18.18% | 0.00 |
# 
# 
# 

# %%bash
# pip install -q transformers**

# In[ ]:


get_ipython().system('pip install spacy==2.2.1')
get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')
get_ipython().system('pip install langdetect')


# ## Load all the 2020 publication texts
# 
# - For any JSON file found in the data location collate all the body text found
# - Create a dictionary of the paper ID and its body text
# - Only include publications in the data which are from 2020 (this has to be done by linking to the metadata.csv file)
# - Create key dictionaries of the index to the paper ID (index2paperID), and the index to publication file pathway (index2paperPath)

# In[ ]:


import pandas as pd
from langdetect import detect

import json
import os

meta_path = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

def get_data_texts():
    
    def get_abstract_language(abstract):
        try:
            language = detect(abstract)
        except:
            language = None
        return language

    # Create dict of paper_id and publication year
    meta_data = pd.read_csv(meta_path, low_memory=True)
    paperID2year = {}
    paperID2lang = {}
    sha2pmcid = {}
    for _, meta_row in meta_data.iterrows():
        # Only save information for meta data with parsed text
        if meta_row['has_pmc_xml_parse'] or meta_row['has_pdf_parse']:
            # The paper ID will either be the pmcid or sha
            if pd.notnull(meta_row['pmcid']):
                paperID2year[meta_row['pmcid']] = meta_row['publish_time']
                if pd.notnull(meta_row['abstract']):
                    lang = get_abstract_language(meta_row['abstract'])
                    if lang:
                        paperID2lang[meta_row['pmcid']] = lang
            # There can be muliple sha IDs in the rows
            if pd.notnull(meta_row['sha']):
                lang = None
                if pd.notnull(meta_row['abstract']):
                    lang = get_abstract_language(meta_row['abstract'])
                paper_ids = meta_row['sha'].split('; ')
                for paper_id in paper_ids:
                    if pd.notnull(meta_row['pmcid']):
                        sha2pmcid[paper_id] = meta_row['pmcid']
                    paperID2year[paper_id] = meta_row['publish_time']
                    if lang:
                        paperID2lang[paper_id] = lang
                                
    data_text = {}
    index2paperID = {}
    index2paperPath = {}
    paperpmcids = set()
    i = 0
    for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge'):
        for filename in filenames:
            paper_path = os.path.join(dirname, filename)
            if paper_path[-4:] != 'json':
                continue
            with open(paper_path) as json_file:
                article_data = json.load(json_file)
                # Don't include duplicates (defined from pmcid - if given) in data_text
                if article_data['paper_id'][0:3] == 'PMC':
                    pmcid = article_data['paper_id']
                else:
                    pmcid = sha2pmcid.get(article_data['paper_id'], None)
                if (not pmcid) or (pmcid not in paperpmcids):
                    if pmcid:
                        paperpmcids.add(pmcid)
                    paper_date = paperID2year.get(article_data['paper_id'], None)
                    paper_language = paperID2lang.get(article_data['paper_id'], None)
                    if paper_date:
                        # Only include papers from 2020 and papers in English (or no language given)
                        if (paper_date[0:4] == '2020') and (paper_language == 'en' or not paper_language):
                            data_text[article_data['paper_id']] = ' '.join([d['text'] for d in article_data['body_text']])
                            index2paperID[i] = article_data['paper_id']
                            index2paperPath[i] = paper_path
                            i += 1

    return data_text, index2paperID, index2paperPath


data_text, index2paperID, index2paperPath = get_data_texts()


# In[ ]:


with open('/kaggle/working/data_text.jsonl', "w") as f:
    json.dump(data_text, f)


# In[ ]:


with open('/kaggle/working/index2paperID.jsonl', "w") as f:
    json.dump(index2paperID, f)


# In[ ]:


with open('/kaggle/working/index2paperPath.jsonl', "w") as f:
    json.dump(index2paperPath, f)


# ## Create the QuestionCovid class
# ```
# Arguments:
#     TOKENIZER: A pretrained BertTokenizer
#     MODEL: A pretrained BertForQuestionAnswering model
#     index2paperID: A dictionary of indexes to the paper id as found from get_data_texts
#     index2paperPath: A dictionary of indexes to the paper pathway as found from get_data_texts
#     data_text: A dictionary of paper ids and the collated body text from them
#     question: A single question to ask the papers
#     
# Attributes:
#     fit: Vectorize the body text from data_text using TFIDF
#     predict: Load the text of the top 5 closest TFIDF vectors of the papers to the question.
#         Split the texts for each of these papers up into chunks of 3 sentences, and predict the
#         answers to the question for each chunk of text using the pretrained BERT QA model.
#         Output the answer with the highest score for each of the 5 closest papers.
#         For each of the 5 closest papers using TFIDF, 'predict' yields:
#         1. the paper id,
#         2. the best answer,
#         3. the BERT QA score,
#         4. the text chunk the answer came from,
#         5. and the cosine similarity between the question and the paper TFIDF vectors. 
# ```
# 
# 

# In[ ]:


from transformers import BertTokenizer, BertForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy

from wasabi import msg
import time

class QuestionCovid:

    def __init__(
            self,
            TOKENIZER,
            MODEL,
            index2paperID,
            index2paperPath
            ):
        self.TOKENIZER = TOKENIZER
        self.MODEL = MODEL
        self.index2paperID = index2paperID
        self.index2paperPath = index2paperPath
        self.scispacy = spacy.load("en_core_sci_sm")

    def fit(self, data_text):

        self.TFIDF_VECTORIZER = TfidfVectorizer()
        with msg.loading("   Fitting TFIDF"):
            start = time.time()
            self.TFIDF_VECTORIZER.fit(data_text.values())
        msg.good("   TFIDF fitted - Took {:.2f}s".format(time.time()-start))
        with msg.loading("   Creating Articles matrix"):
            start = time.time()
            self.ARTICLES_MATRIX = self.TFIDF_VECTORIZER.transform(data_text.values())
        msg.good("   Article matrix created - Took {:.2f}s".format(time.time()-start))

    def get_answer(self, text, question):

        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = self.TOKENIZER.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.MODEL(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.TOKENIZER.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        score = round(start_scores.max().item(), 2)

        return answer, score

    def predict(self, question):

        query = self.TFIDF_VECTORIZER.transform([question + ' covid'])
        best_matches = sorted([(i,c) for i, c in enumerate(cosine_similarity(query, self.ARTICLES_MATRIX).ravel())], key=lambda x: x[1], reverse=True)

        for i, tfidf_score in best_matches[:5]:
            best_score = 0 # if score is negative, i consider the answer wrong
            best_answer = "No answer"
            best_text = "No snippet"
            
            paper_path = self.index2paperPath[i]
            with open(paper_path) as json_file:
                article_data = json.load(json_file)
                text = ' '.join([d['text'] for d in article_data['body_text']])
            sentences = [s.text for s in self.scispacy(text).sents]

            def yield_subtext(sentences):
                subtext = ''
                for i in range(len(sentences)):
                    sent = sentences[i]
                    if len(sent) + len(subtext) > 400:
                        yield subtext
                        subtext = sent
                    else:
                        subtext += sent
            #sentences_grouped = ['.'.join(sentences[i:i+n]) for i in range(0, len(sentences), n)]
            for subtext in yield_subtext(sentences):
                answer, score = self.get_answer(subtext, question)
                if score > best_score:
                    best_score = score
                    best_answer = answer
                    best_text = subtext
            yield (self.index2paperID[i], best_answer, best_score, best_text, tfidf_score)


# ## Load the BERT model and fit the QuestionCovid model with the publication texts

# In[ ]:


TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[ ]:


covid_q = QuestionCovid(TOKENIZER, MODEL, index2paperID, index2paperPath)
covid_q.fit(data_text)


# ## Add each question from the Kaggle competition tasks list
# https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks

# In[ ]:


# Credit to https://www.kaggle.com/jonathanbesomi
challenge_tasks = [
  {
      "task": "What is known about transmission, incubation, and environmental stability?",
      "questions": [
          "Is the virus transmitted by aerosol, droplets, food, close contact, fecal matter, or water?",
          "How long is the incubation period for the virus?",
          "Can the virus be transmitted asymptomatically or during the incubation period?",
          "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV?",
          "How long can the 2019-nCoV virus remain viable on common surfaces?"
      ]
  },
  {
      "task": "What do we know about COVID-19 risk factors?",
      "questions": [
          "What risk factors contribute to the severity of 2019-nCoV?",
          "How does hypertension affect patients?",
          "How does heart disease affect patients?",
          "How does copd affect patients?",
          "How does smoking affect patients?",
          "How does pregnancy affect patients?",
          "What is the fatality rate of 2019-nCoV?",
          "What public health policies prevent or control the spread of 2019-nCoV?"
      ]
  },
  {
      "task": "What do we know about virus genetics, origin, and evolution?",
      "questions": [
          "Can animals transmit 2019-nCoV?",
          "What animal did 2019-nCoV come from?",
          "What real-time genomic tracking tools exist?",
          "What geographic variations are there in the genome of 2019-nCoV?",
          "What effors are being done in asia to prevent further outbreaks?"
      ]
  },
  {
      "task": "What do we know about vaccines and therapeutics?",
      "questions": [
          "What drugs or therapies are being investigated?",
          "Are anti-inflammatory drugs recommended?"
      ]
  },
  {
      "task": "What do we know about non-pharmaceutical interventions?",
      "questions": [
          "Which non-pharmaceutical interventions limit tramsission?",
          "What are most important barriers to compliance?"
      ]
  },
  {
      "task": "What has been published about medical care?",
      "questions": [
          "How does extracorporeal membrane oxygenation affect 2019-nCoV patients?",
          "What telemedicine and cybercare methods are most effective?",
          "How is artificial intelligence being used in real time health delivery?",
          "What adjunctive or supportive methods can help patients?"
      ]
  },
  {
      "task": "What do we know about diagnostics and surveillance?",
      "questions": [
          "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV?"
      ]
  },
  {
      "task": "Other interesting questions",
      "questions": [
          "What is the immune system response to 2019-nCoV?",
          "Can personal protective equipment prevent the transmission of 2019-nCoV?",
          "Can 2019-nCoV infect patients a second time?"
      ]
  }
]


# ## Predict the answer to one question using the most relevant paper

# In[ ]:


question = "How long is the incubation period for the virus?"


# In[ ]:


for i, (paper_id, answer, score, snippet, tfidf_score) in enumerate(covid_q.predict(question)):
    print(f"Answer {i}: {answer}")
    print(f"Text segment: {snippet}")
    print(f"Paper id: {paper_id}")


# ## Predict and save the answers to all the task questions
# This takes up to an hour and will save the answers in '/kaggle/working/answers.jsonl'.

# In[ ]:


# possibly better to write as csv
with open('/kaggle/working/answers.jsonl', "w") as f:
    for task_id, task in enumerate(challenge_tasks):
        task_question = task['task']
        msg.text(f"Task {task_id}: {task_question}")

        questions = task['questions']
        for question_id, question in enumerate(questions):
            with msg.loading(f"Answering question: {question}"):
                start = time.time()
                for i, (paper_id, answer, score, snippet, tfidf_score) in enumerate(covid_q.predict(question)):
                    chunk = json.dumps({
                        'task_id': task_id,
                        'task': task_question,
                        'question_id': question_id,
                        'question': question,
                        'paper_id': paper_id,
                        'answer': answer,
                        'snippet': snippet,
                        'bert_score': score,
                        'tfidf_score': tfidf_score
                    })
                    f.write(chunk + '\n')
                    msg.text("\n")
                    msg.text(f"Answer {i}: {answer}")
            time_elapsed = time.time()-start
            msg.good(f"Question {question_id} answered - Took {time_elapsed}s")


# In[ ]:




