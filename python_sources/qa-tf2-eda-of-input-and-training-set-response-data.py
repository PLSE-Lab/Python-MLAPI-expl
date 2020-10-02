#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# Any results you write to the current directory are saved as output.
import os
import gc
import matplotlib.pyplot as plt
import json

from collections import Counter


# In[ ]:


# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # File Description
# 
# * simplified-nq-train.jsonl - the training data, in newline-delimited JSON format.
# * simplified-nq-kaggle-test.jsonl - the test data, in newline-delimited JSON format.
# * sample_submission.csv - a sample submission file in the correct format

# # Data fields
# * document_text - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
# * question_text - the question to be answered
# * long_answer_candidates - a JSON array containing all of the plausible long answers.
# * annotations - a JSON array containing all of the correct long + short answers. Only provided for train.
# * document_url - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
# * example_id - unique ID for the sample.

# # Load Data

# In[ ]:


path = '/kaggle/input/tensorflow2-question-answering/'
train_path = 'simplified-nq-train.jsonl'
test_path = 'simplified-nq-test.jsonl'
sample_submission_path = 'sample_submission.csv'


# In[ ]:


train_line_count = 0
with open(path+train_path) as f:
    for line in f:
        line = json.loads(line)
        train_line_count += 1
train_line_count


# The dataset is huge, for exploration purpose we are going perform the exploratory analysis over a sample of the dataset. Let's read the training data and extract a sample (hopefully the dataset is shuffled so that the first records are random)

# In[ ]:


def read_data(path, sample = True, chunksize = 30000):
    if sample == True:
        df = []
        with open(path, 'rt') as reader:
            for i in range(chunksize):
                df.append(json.loads(reader.readline()))
        df = pd.DataFrame(df)
    else:
        df = pd.read_json(path, orient = 'records', lines = True)
        gc.collect()
    return df


# In[ ]:


train = read_data(path+train_path, sample = True, chunksize=100000)
print("train shape", train.shape)
train[:10]


# In[ ]:


test = read_data(path+test_path, sample = False)
print("test shape", test.shape)
test[:10]


# In[ ]:


sample_submission = pd.read_csv(path + sample_submission_path)
print("Sample submission shape", sample_submission.shape)


# The submission file has 692 rows, this means that for each row in the test set (there are 346) we have to predict both the short and long answer.
# The short answer may also be a Yes or No answer.

# In[ ]:


sample_submission[:10]


# # Missing Values
# What columns we have and whether or not we have missing values

# In[ ]:


def missing_values(df):
    df = pd.DataFrame(df.isnull().sum()).reset_index()
    df.columns = ['features', 'n_missing_values']
    return df


# In[ ]:


missing_values(train)


# In[ ]:


missing_values(test)


# Great we don't have missing values.

# # What the train data looks like

# In[ ]:


train.columns


# ### Question text
# The question as a text string. Note that it is not tokenized.

# In[ ]:


# Question text
train.loc[0, 'question_text']


# ### Document text
# 'The text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.'
# This is a huge wikipedia text, where we need to find the answer for the previous question
# Note that they give the tokenization method they expect us to use.

# In[ ]:


train.loc[0, 'document_text']


# In[ ]:


train.loc[0, 'document_text'].split()[:100]


# ### Long answer candidates
# 'A JSON array containing all of the plausible long answers'.
# There seem to be a lot of these. Each one has a start_token and an end_token in the text, to delimit where the long answer is.
# There is also a 'top_level' field.

# In[ ]:


train.loc[0, 'long_answer_candidates'][:10]


# ### Annotations
# 'A JSON array containing all of the correct long + short answers. Only provided for train.'
# This defines the target we are training on.

# In[ ]:


train['annotations'][:10]


# In[ ]:


# Make a dataframe to accumulate answer types
answer_num_annotations = train.annotations.apply(lambda x: len(x))
Counter(answer_num_annotations)


# So there is only ever one annotation in the train data.
# Let us explore how well the annotations are structured. First we will find out what things are in the annotations.

# In[ ]:


train['annotations'][0]


# * This is our target variable. In this case this is telling us that our long answer starts in indices 1952 and end at indices 2019.
# * Also, we have a short answer that starts a indices 1960 and end at indices 1969.
# * In this example we dont have a yes or no answer

# In[ ]:


set().union(*(d[0].keys() for d in train['annotations']))


# So we know about all our keys.

# ## Annotations Analysis
# Annotations is a JSON array containing all of the correct long + short answers. Only provided for train.
# We will build a reformulated data frame around the annotations.

# In[ ]:


answer_summary = train['example_id'].to_frame()


# In[ ]:


answer_summary['annotation_id'] = train.annotations.apply(lambda x: x[0]['annotation_id'])


# In[ ]:


answer_summary[:10]


# ### Yes-No

# In[ ]:


answer_yes_no = train.annotations.apply(lambda x: x[0]['yes_no_answer'])
yes_no_answer_counts = Counter(answer_yes_no)
yes_no_answer_counts


# In[ ]:


ks = [k for k in yes_no_answer_counts.keys()]
vs = [yes_no_answer_counts[k] for k in ks]

plt.bar(ks, vs)


# In[ ]:


percent_yes_no = 1 - yes_no_answer_counts['NONE'] / sum(yes_no_answer_counts.values())
print(percent_yes_no, "of the questions have yes/no answers given")


# Clean column and save in summary

# In[ ]:


answer_yes_no_cleaned = answer_yes_no.apply(lambda x: None if x == 'NONE' else x)
answer_summary['has_yes_no'] = answer_yes_no_cleaned.apply(lambda x: x is not None)
answer_summary['yes_no'] = answer_yes_no_cleaned


# ### Short Answers

# In[ ]:


train.annotations.apply(lambda x: x[0]['short_answers'])[:10]


# In[ ]:


answer_short = train.annotations.apply(lambda x: [(y['start_token'], y['end_token']) for y in x[0]['short_answers']])
num_short_answers = answer_short.apply(lambda x: len(x))
short_answer_counts = Counter(num_short_answers)
short_answer_counts


# In[ ]:


ks = [k for k in short_answer_counts.keys()]
ks.sort()
vs = [short_answer_counts[k] for k in ks]

plt.bar(ks, vs)


# So a large majority of the questions do not have any short answers. Of the remaining, most have only one short answer. The longest list is one question that has 21 answers.

# In[ ]:


percent_short = 1 - short_answer_counts[0] / sum(short_answer_counts.values())
print(percent_short, "of the questions have at least one short answer")


# Clean columns and save in summary

# In[ ]:


answer_summary['has_short_answers'] = num_short_answers.apply(lambda x: x>0)
answer_summary['num_short_answers'] = num_short_answers
answer_summary['answer_short'] = answer_short.apply(lambda x: x if len(x) > 0 else None)


# ### Long Answers

# In[ ]:


train.annotations.apply(lambda x: x[0]['long_answer'])[:20]


# In[ ]:


train.loc[0, 'annotations'][0]['long_answer']


# So we have a start token and an end token, similar to the short answers.
# The candidate index is the index into the list of candidate answers.

# In[ ]:


answer_long = train.annotations.apply(lambda x: (x[0]['long_answer']['start_token'], x[0]['long_answer']['end_token']))
answer_long_cleaned = answer_long.apply(lambda x: x if x != (-1, -1) else None)
num_long_answers = answer_long_cleaned.apply(lambda x: 1 if x else 0)
long_answer_counts = Counter(num_long_answers)
long_answer_counts


# In[ ]:


ks = [k for k in long_answer_counts.keys()]
ks.sort()
vs = [long_answer_counts[k] for k in ks]

plt.bar(ks, vs)


# In[ ]:


percent_long = 1 - long_answer_counts[0] / sum(long_answer_counts.values())
print (percent_long, "of the questions have at least one long answer")


# Update the summary.

# In[ ]:


answer_summary['has_long_answer'] = num_long_answers.apply(lambda x: x>0)
answer_summary['num_long_answers'] = num_long_answers
answer_summary['answer_long'] = answer_long_cleaned


# Also look at the candidate indices.

# In[ ]:


candidate_indices = train.annotations.apply(lambda x: (x[0]['long_answer']['candidate_index']))


# In[ ]:


answer_summary['long_candidate_index'] = candidate_indices


# In[ ]:


answer_summary[:10]


# ### Summary

# How many of the questions have some answer?

# In[ ]:


summary = answer_summary.apply(lambda row: 
                               True if (row['has_yes_no'] or row['has_short_answers'] or row['has_long_answer'])
                               else False, axis=1)
summary[:10]


# In[ ]:


Counter(summary)


# Looks like about half our training set consists of questions for which the document does not contain an answer.

# In[ ]:


answer_summary["summary"] = summary


# Count how many have both a yes-no and at least one short answer; also how many have both short and long answers.

# In[ ]:


answer_summary.groupby(['has_yes_no', 'has_short_answers', 'has_long_answer']).size().reset_index()


# So in the training data there is either a yes-no answer or short answers, but not both.
Here is the final summary dataframe
# In[ ]:


answer_summary[:10]


# In[ ]:




