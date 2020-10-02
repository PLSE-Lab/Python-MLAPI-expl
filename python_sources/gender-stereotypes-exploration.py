#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook we explore some of gender stereotypes which are automatically learned by word embedings algorithms. We'll use ideas from [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)  paper. 
# 
# Word embedings algorithms has a property of learning stereotypes from text they are trained on. Sometimes we want to get rid of them sometimes they can be usefull. 
# 
# In this experiment we'll use  [GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/) word mebedings and  [gensim](https://radimrehurek.com/gensim/index.html) as library.

#  Load gensim libary

# In[ ]:


import gensim

# Set path to word embedings file

word2vecpath = "../input/GoogleNews-vectors-negative300.bin.gz"


# Load word vectors (word2vec) into model

# In[26]:


word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecpath, binary=True)


# We need a function which takes as input a sentence and returns similarity to male and female directions. For this task it will use list of words stongly associated with gender like **she**, **he**, **man**, **woman**.

# In[30]:


def get_gender_scores(sentence):
    # split sentence into words
    words = sentence.lower().split()
    
    # compute similarity for male and female direction.
    male_score = word2vec.n_similarity(['he', 'his', 'man', 'himself', 'son'], words)
    female_score = word2vec.n_similarity(['she', 'her', 'woman', 'herself', 'daughter'], words)
    
    return male_score, female_score


# In[34]:


words = ['security', 'doctor', 'nurse', 'health insurance', 'english teacher', 'math',          'programmer', 'housekeeping', 'driver', 'plumber']
for word in words:
    male_score, female_score = get_gender_scores(word)
    stereotype = 'male' if male_score > female_score else 'female'
    print('"%s" is "%s" (male score: %f, female score: %f)' % (word, stereotype, male_score, female_score))


# It is a very basic example and it doesn't handle situations when words are missing from dictionary, but it is a good start for stereotypes exploring. One way of improving algorithm performance could be adding more gender specific wordsto gender vectors (like girl, boy, mother, father, etc).
# 

# In[ ]:




