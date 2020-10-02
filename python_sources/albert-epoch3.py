#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
import sys
import collections
import json

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


answers_df = pd.read_json("../input/predictions/predictions.json")
answers_df.head()
answers_df["predictions"][1]


# In[ ]:


def df_long_index_score(df):
    answers = []
    cont = 0
    for i in range(len(df)):
        index={}
        if df["predictions"][i]['long_answer_score']>0:
            index['start'] = df["predictions"][i]['long_answer']['start_token']
            index['end'] = df["predictions"][i]['long_answer']['end_token']
            index['score'] = df["predictions"][i]['long_answer_score']
            index = [index]
            answers.append(index) 
        else:
            answers.append([])    
    return answers

def df_short_index_score(df):
    answers = []
    cont = 0
    for i in range(len(df)):
        index={}
        if df["predictions"][i]['short_answers_score']>0 and df["predictions"][i]['short_answers'][0]['start_token'] != -1:
            index['start'] = df["predictions"][i]['short_answers'][0]['start_token']
            index['end'] = df["predictions"][i]['short_answers'][0]['end_token']
            index['score'] = df["predictions"][i]['short_answers_score']
            index = [index]
            answers.append(index) 
        else:
            answers.append([]) 
    return answers


# In[ ]:


def df_example_id(df):
    return df['example_id']


# In[ ]:


# answers_df['answer'] = answers_df['predictions'].apply(df_long_index_score)
answers_df['long_indexes_and_scores']=df_long_index_score(answers_df)
answers_df['short_indexes_and_scores']=df_short_index_score(answers_df)


# In[ ]:


answers_df['example_id'] = answers_df['predictions'].apply(df_example_id)


# In[ ]:


answers_df = answers_df.drop(['predictions'], axis=1)


# In[ ]:


answers_df.head()


# In[ ]:


def create_answer(entry):
    answer = []
    for e in entry:
        answer.append(str(e['start']) + ':'+ str(e['end']))
    if not answer:
        answer = ""
    return ", ".join(answer)


# In[ ]:


answers_df["long_answer"] = answers_df['long_indexes_and_scores'].apply(create_answer)
answers_df["short_answer"] = answers_df['short_indexes_and_scores'].apply(create_answer)
answers_df["example_id"] = answers_df['example_id'].apply(lambda q: str(q))

long_answers = dict(zip(answers_df["example_id"], answers_df["long_answer"]))
short_answers = dict(zip(answers_df["example_id"], answers_df["short_answer"]))

answers_df.head()


# In[ ]:


answers_df = answers_df.drop(['long_indexes_and_scores', 'short_indexes_and_scores'], axis=1)
answers_df.head()


# In[ ]:


sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")

long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)

sample_submission.loc[sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings
sample_submission.loc[sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings


# In[ ]:


import os
path = '/kaggle/working'
os.chdir(path)
os.getcwd()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


sample_submission['PredictionString'][146:180]


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from pandas import DataFrame
answers_df["long_answer"][70:80]

