#!/usr/bin/env python
# coding: utf-8

# With the advent of social media forums, people are becoming more vulnerable to prejudices, hatred, attacks, and insults etc.
# 
# So, it is inevitable for social media forums to identity topics intend to destroy peace and harmony among people. 
# 
# In this competition quora has given opportunity to kagglers to classify questions asked on its platform as sincere or insincere.
# 
# 

# **Notebook Objective:**
# 
# Objective of the notebook is to extract topics people are discussing in insincere category

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# gensim
import gensim
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_df = pd.read_csv("../input/train.csv")


# In[ ]:


train_df_insincere = train_df[train_df.target==1]


# In[ ]:


# select observations that belong to genre of insincere questions
insincere_questions = train_df_insincere['question_text']


# **Text cleaning**
# 
# Remove stop words
# 
# Remove special characters such as punctuations

# 
# 

# In[ ]:


# tokenize words and cleaning up punctuations and so
def sent_to_words(insincere_questions):
    for question in insincere_questions:
        yield(gensim.utils.simple_preprocess(str(question), deacc=True))  # deacc=True removes punctuations

question_words = list(sent_to_words(insincere_questions))


# In[ ]:


# remove stopwords
stop_words = set(stopwords.words("english"))
questions_without_stopwords = [[word for word in simple_preprocess(str(question)) 
                                if word not in stop_words] for question in question_words]


# Bigrams are referred to as two words frequently occuring together such as Donald Trump, United states, north korea. 
# 
# Let's now create bigrams. 
# 

# In[ ]:


# Form Bigrams
bigram = Phrases(questions_without_stopwords, min_count=5, threshold=100) 
# higher value of the params min_count and threshold will result in less bigrams. 
# You can playaround with these parameters to get better bigrams

bigram_mod = Phraser(bigram)
bigrams = [bigram_mod[question] for question in questions_without_stopwords]


# In[ ]:


# Create Dictionary of words - This creates id for each word/ phrase
id2word = Dictionary(bigrams) 
print("Word at 0th id: ", id2word[0])

# create corpus - Convert a list of words into the bag-of-words forma
corpus = [id2word.doc2bow(text) for text in bigrams]
print("First element of corpus: ", corpus[0])


# Before creating topic model, Let's find the optimal number of topics. 

# 
# 

# In[ ]:


# this is bit computation intensive and may take time
coherence_values = []
model_list = []
range_num_topics = range(2,10)
for num_topics in range_num_topics:
    model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                            random_state=100, update_every=1,
                                            chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    model_list.append(model)
    coherencemodel = CoherenceModel(model=model, texts=bigrams, dictionary=id2word, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())


# In[ ]:


# Print the coherence scores
for num_topic, coherence_value in zip(range_num_topics, coherence_values):
    print("Number of Topics : ", num_topic, " . Coherence Value: ", round(coherence_value, 3))


# In[ ]:


# Plot coherence score graph
plt.plot(range_num_topics, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend("coherence_values", loc='best')
plt.show()


# Model with the highest coherence value is the best fit for the corpus. It gives intuition of what corpus looks like.
# 
# But if coherence value keeps on increasing, pick the model that gives highest coherence score before flattening out.
# 
# Providing higher number of topics to model usually causes overlapping between topics i.e same keyword may become part of various topic. 
# 
# Thus, it may not give any clear intuition of what the particular topic is trying to convey.
# 
# So, it is really important to find optimal number of topics.

# Let's visualize model with 3 number of topics. This is experimental . 
# 
# You can try out different number of topics and visulaize overlapping of topic segments

# In[ ]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model_list[1], corpus, id2word)
vis


# From the above visualization, we can visualize what is each topic is trying to convey.
# 
# Each bubble represents topic and hovering mouse over each bubble highlight words. 
# 
# The words are salient keywords that constitute the particular topic.
# 
# 
# * Topic1 talks more about Donald trump, women, americans.
# * Topic2 talks more about muslims, indians, christians. It seems to be talking more on religion 
# * Topic3 talks more about quora.

# Data preprocessing to remove unwanted words, phrases that do not account much in prediction will result 
# 
# in better formation of topic clusters and better coherence score.
