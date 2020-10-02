#!/usr/bin/env python
# coding: utf-8

# # Quora Insincere Questions Exploratory Data Analysis
# 
# We will begin exploring the training data in order to come up with insights and a plan for modeling.

# In[ ]:


# import packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
import nltk
import re

import string
from nltk.corpus import stopwords
from collections import Counter

# print any variable/statement on its own line (not just the last one!)
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

np.random.seed(27)


# In[ ]:


# setting up default plotting parameters
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = [20.0, 7.0]
plt.rcParams.update({'font.size': 22,})

sns.set_palette('viridis')
sns.set_style('white')
sns.set_context('talk', font_scale=0.8)


# In[ ]:


# load in data and print shape/head/tail
raw_data = pd.read_csv('../input/train.csv')
print(raw_data.shape)
raw_data.head()


# In[ ]:


# using seaborns countplot to show distribution of questions in dataset
fig, ax = plt.subplots()
g = sns.countplot(raw_data.target, palette='viridis')
g.set_xticklabels(['Sincere', 'Insincere'])
g.set_yticklabels([])

# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Distribution of Questions', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
fig.savefig('classes.png')
plt.show()


# In[ ]:


# print percentage of questions where target == 1
(len(raw_data.loc[raw_data.target==1])) / (len(raw_data.loc[raw_data.target == 0])) * 100


# ### Class Imbalance
# Imbalanced classes are a common problem in machine learning classification where there are a disproportionate ratio of observations in each class.  With just 6.6% of our dataset belonging to the target class, we can definitely have an imbalanced class!
# 
# This is a problem because many machine learning models are designed to maximize overall accuracy, which especially with imbalanced classes may not be the best metric to use.  Classification accuracy is defined as the number of correct predictions divided by total predictions times 100.  For example, if we simply predicted that all questions are sincere, we would get a classification acuracy score of 93%!
# 
# This competition uses the F1 score which balances precision and recall.
#  - Precision is the number of true positives divided by all positive predictions.  Precision is also called Positive Predictive Value.  It is a measure of a classifier's exactness.  Low precision indicates a high number of false positives.
#  - Recall is the number of true positives divided by the number of positive values in the test data.  Recall is also called Sensitivity or the True Positive Rate.  It is a measure of a classifier's completeness.  Low recall indicates a high number of false negatives.

# In[ ]:


# printing out a random sample of questions labeled insincere
import random

index = random.sample(raw_data.index[raw_data.target == 1].tolist(), 5)
for i in index:
    print(raw_data.iloc[i, 1])


# In[ ]:


# taking a sample of the training data to speed up processing
df = raw_data.sample(frac=0.3)
df.shape


# In[ ]:


# tokenize with spacy
nlp = spacy.load('en')

df['tokens'] = [nlp(text, # disable parts of the language processing pipeline we don't need here to speed up processing
                    disable=['ner', # named entity recognition
                                   'tagger', # part-of-speech tagger
                                   'textcat', # document label categorizer
                                  ]) for text in df.question_text]
df.sample(5)


# In[ ]:


df['num_tokens'] = [len(token) for token in df.tokens]
df.sample(5)


# In[ ]:


# using seaborns boxplot to visualize number of tokens per question
fig, ax = plt.subplots()
g = sns.boxplot(x=df.target, y=df.num_tokens, palette='viridis')
g.set_xticklabels(['Sincere', 'Insincere'])
g.set_yticklabels([])

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Number of Tokens per Question', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
fig.savefig('tokens.png')
plt.show()


# In[ ]:


# get number of sentences per question
print(list(df.iloc[0,3].sents))

sents = [list(x.sents) for x in df.tokens]
df['num_sents'] = [len(sent) for sent in sents]
df.sample(5)


# In[ ]:


# plotting number of sentences per question
fig, ax = plt.subplots()
g = sns.countplot(df.num_sents, hue=df.target, palette='viridis')
#g.set_xticklabels(['Sincere', 'Insincere'])
g.set_yticklabels([])

# using log scale on y-axis so we can better see the questions with more sentences
ax.set(yscale='log')

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Number of Sentences per Question', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
fig.savefig('sentences.png')
plt.show()


# In[ ]:


# Finding most common words
# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
punctuations = string.punctuation
stop_words = set(stopwords.words("english"))

def cleanup_text(docs):
    texts = []
    for doc in docs:
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = nlp(doc, disable=['ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# In[ ]:


# Grab all text associated with insincere questions
insincere_text = [text for text in df[df['target'] == 1]['question_text']]
insincere_clean = cleanup_text(insincere_text)
insincere_clean = ' '.join(insincere_clean).split()


# In[ ]:


# Count all unique words
insincere_counts = Counter(insincere_clean)
# get words and word counts
insincere_common_words = [word[0] for word in insincere_counts.most_common(20)]
insincere_common_counts = [word[1] for word in insincere_counts.most_common(20)]

# plot 20 most common words in insincere questions
sns.barplot(insincere_common_words, insincere_common_counts, palette='viridis')
sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Insincere Questions Common Words', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
fig.savefig('insincere_words.png')
plt.show()


# In[ ]:


# Grab all text associated with sincere questions
sincere_text = [text for text in df[df['target'] == 0]['question_text']]
sincere_clean = cleanup_text(sincere_text)
sincere_clean = ' '.join(sincere_clean).split()


# In[ ]:


# Count all unique words
sincere_counts = Counter(sincere_clean)
# get words and word counts
sincere_common_words = [word[0] for word in sincere_counts.most_common(20)]
sincere_common_counts = [word[1] for word in sincere_counts.most_common(20)]

# plot 20 most common words in sincere questions
sns.barplot(sincere_common_words, sincere_common_counts, palette='viridis')
sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Sincere Questions Common Words', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
fig.savefig('sincere.png')
plt.show()


# In[ ]:




