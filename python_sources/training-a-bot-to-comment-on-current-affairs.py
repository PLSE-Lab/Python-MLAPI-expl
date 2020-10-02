#!/usr/bin/env python
# coding: utf-8

# # Training Markov Chain Model on NYT Comments using `markovify` and `spaCy`

# The comments on New York Times articles are very well written, rich in vocabulary and mostly grammatically correct, yet they are close to lay person's conversational language. This makes the [dataset of ***2 million comments*** on NYT articles]() a good candidate for generating automated comments. This kernel is an attempt to make a bot comment meaningfully on topics of current affairs by generating comments similar to those on the NYT articles.
# 
# ***To see this kernel in action, check out twitter bot [@OnAffairs](https://twitter.com/OnAffairs) trained on NYT comments' dataset [[code](https://github.com/AashitaK/CurrentOnAffairs/blob/master/README.md)].***
# 

# Python package `markovify` is used for Markov chain generator. To improve the sentence structure for the generated comments, high performance NLP package `spaCy` is used for parts of speech tagging and functions from the package `markovify` are overriden.

# ## Steps:
# * Preparing text from comments for training the generator.
# * Training a simple Markov chain generator using the comments' text and using it to generate some comments.
# * Training an improved Markov chain generator with POS-Tagged text and using it to generate more comments.

# 

# ### Loading required packages and data

# We use the python package [`markovify`](https://github.com/jsvine/markovify), that has an in-built Markov chain generator for the automated text generation.

# In[ ]:


import pandas as pd
import markovify 
import spacy
import re

import warnings
warnings.filterwarnings('ignore')

from time import time
import gc


# In[ ]:


curr_dir = '../input/'
df1 = pd.read_csv(curr_dir + 'CommentsJan2017.csv')
df2 = pd.read_csv(curr_dir + 'CommentsFeb2017.csv')
df3 = pd.read_csv(curr_dir + 'CommentsMarch2017.csv')
df4 = pd.read_csv(curr_dir + 'CommentsApril2017.csv')
df5 = pd.read_csv(curr_dir + 'CommentsMay2017.csv')
df6 = pd.read_csv(curr_dir + 'CommentsJan2018.csv')
df7 = pd.read_csv(curr_dir + 'CommentsFeb2018.csv')
df8 = pd.read_csv(curr_dir + 'CommentsMarch2018.csv')
df9 = pd.read_csv(curr_dir + 'CommentsApril2018.csv')
comments = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])
comments.drop_duplicates(subset='commentID', inplace=True)
comments.head(3)


# In[ ]:


comments.shape


# In[ ]:


comments.sectionName.value_counts()[:5]


# We select the section *'Politics'* for articles so that the comments generated stay on the topic and clean up the comments' text:

# In[ ]:


def preprocess(comments):
    commentBody = comments.loc[comments.sectionName=='Politics', 'commentBody']
    commentBody = commentBody.str.replace("(<br/>)", "")
    commentBody = commentBody.str.replace('(<a).*(>).*(</a>)', '')
    commentBody = commentBody.str.replace('(&amp)', '')
    commentBody = commentBody.str.replace('(&gt)', '')
    commentBody = commentBody.str.replace('(&lt)', '')
    commentBody = commentBody.str.replace('(\xa0)', ' ')  
    return commentBody


# In[ ]:


commentBody = preprocess(comments)
commentBody.shape


# Freeing up memory:

# In[ ]:


del comments, df1, df2, df3, df4, df5, df6, df7, df8
gc.collect()


# A random comment from the dataset:

# In[ ]:


commentBody.sample().values[0]


# ### Training the Markov chain generator using the NYT comments:

# For Markov chains, future is independent of the past and depends only on the present. For the text generation, the Markov chain generator focuses on the current word (or words depending on state size) and randomly (weighted by the transition probabilities) find the next word. These transition probabilities are trained from the input text by calculating how frequently words follows other words (or phrases). Here, we have used a pretty high state size of 5, which means the generator consider the current 5 words phrase to find the next one.

# In[ ]:


start_time = time()
comments_generator = markovify.Text(commentBody, state_size = 5)
print("Run time for training the generator : {} seconds".format(round(time()-start_time, 2)))


# ### Generating comments:

# In[ ]:


# Print randomly-generated comments using the built model
def generate_comments(generator, number=10, short=False):
    count = 0
    while count < number:
        if short:
            comment = generator.make_short_sentence(140)
        else:
            comment = generator.make_sentence()
        if comment:
            count += 1
            print("Comment {}".format(count))
            print(comment)
            print()
    


# In[ ]:


generate_comments(comments_generator)


# ### Improving Markov chain generator using `spaCy` for POS-Tagging:

# The comments generated above are pretty good, but the sentence structure can be improved by using parts of speech tagging. Here we use high-performance library `Spacy` for this purpose and override the relevant functions in the `markovify` module.

# In[ ]:


nlp = spacy.load("en")

class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ["::".join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence


# The POS-Tagging somewhat slows down the training of the generator model, so this time we use a smaller training set consisting of comments from April 2018.

# In[ ]:


commentBody = preprocess(df9)
commentBody.shape


# Freeing up memory:

# In[ ]:


del comments_generator, df9
gc.collect()


# ### Generating more comments:

# In[ ]:


# start_time = time()
# comments_generator_POSified = POSifiedText(commentBody, state_size = 2)
# print("Run time for training the generator : {} seconds".format(round(time()-start_time, 2)))


# In[ ]:


# generate_comments(comments_generator_POSified)


# References:
# * [Markovify package](https://github.com/jsvine/markovify)
# * https://www.kaggle.com/nulldata/nlg-for-fun-automated-headlines-generator
