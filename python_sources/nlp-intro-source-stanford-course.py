#!/usr/bin/env python
# coding: utf-8

# **Natural Language Processing**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Natural Language Processing Use Cases**:
# * Question Answering Example IBM Watson
# * Information Extraction Example Automatically create a calendar entry from mail
# * Sentimental Analysis Example Automatically determine attributes about a product negative or positive
# * Machine Tranlation Example Chinese to English
#  

# **Language Technology**
# 1. Mostly Solved
#     * Spam Detection
#     * Part of Speech Tagging  
#     * Named Entity Recognition
# 2. Making Good Progress
#     * Sentimental Analysis
#     * Word Sense Disambiguation
# 3. Really Hard
#     * Paraphrase
#     * Summarization    

# **What makes NLP hard?**
# 
# Ambiguity makes NLP hard because ambuiguity creates various interpretations. 
# 
# 1. Syntactic Ambiguity
# Example:- Teacher strikes idle kids.
#         Machine Interpretation :- Teacher strikes + idle(verb) + kids   ( teacher striking makes kids idle)
#         Human Interpretation :- Teacher  +strikes + idle kids           ( teacher strikes idle kids)
#         
# 2. Ambiguity is pervasive. Phrase structure and dependency structure between words make different interpretations. More and more ambiguity.
# 
# 3. Segmenatation issues , idioms , new words that has never seen before and many more

# **Regular Expressions**
# It is a formal language to specifying text strings.
# 1. Disjunctions  :- [A-Z] Any letter between this range , [ !] will match all non- alphanumeric alphabets.
# 2. Negations  :-  [^A-Z] Not a capital letter  ,  [^e^] Neither e nor ^ ,^[b-d] start of line , [a-z]$ matches end of line , use escape(\\) to match special symbol , period(.) matches everything
# 3. Pipe symbol  :-  [Mm]ayank|[Yy]adav  disjunction without square bracket
# ![image.png](attachment:image.png)
# 
# **Regular Expressions are used as features in classifiers. Can be useful in capturing generalizations.**

# **Errors**
# * Two kinds of errors:-
#     1. Type 1 error (False Positives) :- Matching strings that shouldn't have been matched. Reducing False positive -> Increasing accuracy
#     2. Type 2 error (False Negatives) :- Not matching things that should have matched. Reducing False negatives  -> Increasing Recall

# **Word tokenization and Normalization**
# * Lemma :- same stem , part of speech , rough word sense Example :- Cat and cats same lemma but differnt word forms ( the full inflected surface form)

# In[ ]:




