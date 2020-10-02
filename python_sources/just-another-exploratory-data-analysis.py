#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import json

from IPython.display import display

#local script
from tfutils_py import get_answer, read_sample

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')


# # Dataset Introduction
# Here's an introduction to the dataset based on the info provided here: https://github.com/google-research-datasets/natural-questions/blob/master/README.md.
# 
# This challenge is actually a specific type of challenge called **Natural Question**. When a user asks a question to a search engine, instead of returning the user a whole document that contains the answer to this question, we want to be able to return the answer directly taken from that document. The motivation is that the user doesn't want to have to manually parse through the document looking for his answer.
# 
# ## Long Answer Candidates
# 
# The first task in *Natural Question* problems [...] is to identify the **smallest HTML bounding box** that contains all of the information required to infer the answer to a question. The candidates will therefore be bounded by the following html tags:
# - paragraphs;
# - lists;
#     * list items;
# - tables;
#      * table rows. 
# 
# 
# In this specific problem, **multiple** long-answer-candidates to each question were generated (presumably through an automated process).
# ## Annotations
# It is the goal of the annotators to choose the *best long answer candidate* to answer the question. This *best candidate* is the one that verifies the condition: **The smallest long-answer-candidate that answers the question.** Furthermore, if applicable, the annotators also have to specify the answer (yes or no) in case it is a yes/no question and lastly also specify set of short answers which should answer the question in less words than the **best long-answer-candidate**, if these can be found in the text.
# 
# ## Challenge
# 
# The train set essentially contains a lot of manual annotations, in this challenge we want to train a model with those annotations so it learns how to annotate by itself on future questions.

# # EDA
# 
# 

# I wrote an utility script to make this kernel cleaner, it can be found here: https://www.kaggle.com/snovaisg/tfutils-py

# In[ ]:


df = read_sample(n=3)
df.head()


# # document_text

# This is a sample of the first document in the dataset

# In[ ]:


df = read_sample(n=10)
df.loc[0,'document_text'][:50]


# Let's check the distribution of document size

# In[ ]:


df = read_sample(n=1000)
doc_text_words = df['document_text'].apply(lambda x: len(x.split(' ')))
plt.figure(figsize=(12,6))
sns.distplot(doc_text_words.values,kde=True,hist=False).set_title('Distribution of text word count of 1000 docs')


# ## Conclusions
# 
# - The text is in html
# - Most texts in this dataset contain about 5k words.

# # long_answer_candidates

# These are candidates which were selected prior to the annotators giving their answer. They contain the *start* and *end* tokens of the html text which will **always** be html tags.

# In[ ]:


df = read_sample(n=3)
df.long_answer_candidates[0][:5]


# We can take some conclusions already:
# - **start_token**: The token position in the text where the answer begins;
# - **end_token**: The token position in the text where the answer ends;
# - **top_level**: Whether this answer is contained inside another answer in the text.
# 
# Furthermore,
# - There can be multiple candidate-answers to a single question

# In[ ]:


# sample answer
sample = df.iloc[0]
get_answer(sample.document_text, sample.long_answer_candidates[0])


# # Annotations
# We can think of these as our target variables. There are few possibilities for these:
# 
# - Long answer;
# - yes or no answer;
# - short answer;
# - Answer doesn't exist in the text;
# 
# 
# Let's simply understand the distribution of these in our dataset. We'll start by the long_answers because when there isn't a long answer for a question, we can conclude there doesn't exist any form of answer to the question on the wikipedia page.
# 
# ## long_answers_distribution

# In[ ]:


def preprocess(n=10):
    df = read_sample(n=n,ignore_doc_text=True)
    df['yes_no'] = df.annotations.apply(lambda x: x[0]['yes_no_answer'])
    df['long'] = df.annotations.apply(lambda x: [x[0]['long_answer']['start_token'], x[0]['long_answer']['end_token']])
    df['short'] = df.annotations.apply(lambda x: x[0]['short_answers'])
    return df
df = preprocess(5000)


# In[ ]:


display(df.long.apply(lambda x: "Answer Doesn't exist" if x[0] == -1 else "Answer Exists").value_counts(normalize=True))


# About half of the questions don't have an answer, that's surprising!

# In[ ]:


# let's keep a mask of the answers that exist
mask_answer_exists = df.long.apply(lambda x: "Answer Doesn't exist" if x == -1 else "Answer Exists") == "Answer Exists"


# ## Distribution of Yes and No Answers

# In[ ]:


yes_no_dist = df[mask_answer_exists].yes_no.value_counts(normalize=True)
yes_no_dist


# Essentially most of the answerable questions aren't a yes/no type.
# 
# ## Distribution of short answers

# In[ ]:


short_dist = df[mask_answer_exists].short.apply(lambda x: "Short answer exists" if len(x) > 0 else "Short answer doesn't exist").value_counts(normalize=True)
plt.figure(figsize=(8,6))
sns.barplot(x=short_dist.index,y=short_dist.values).set_title("Distribution of short answers in answerable questions")


# About 60% of the questions that are answerable don't have a short answer. There can also be multiple short answers to a question so let's check the distribution of that!

# In[ ]:


short_size_dist = df[mask_answer_exists].short.apply(len).value_counts(normalize=True)
short_size_dist_pretty = pd.concat([short_size_dist.loc[[0,1,],], pd.Series(short_size_dist.loc[2:].sum(),index=['>=2'])])
short_size_dist_pretty = short_size_dist_pretty.rename(index={0: 'No Short answer',1:"1 Short answer",">=2":"More than 1 short answers"})
plt.figure(figsize=(12,6))
sns.barplot(x=short_size_dist_pretty.index,y=short_size_dist_pretty.values).set_title("Distribution of Number of Short Answers in answerable questions")


# Conclusions:
# - About 60% of the answerable questions don't have a short answer
# - About 30% of the answerable questions have 1 short answer
# - About 10% of the answerable questions have more than 1 possible short answer

# In[ ]:




